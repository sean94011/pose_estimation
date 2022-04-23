import cv2
import numpy as np
import ros_numpy
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from sklearn.linear_model import LinearRegression
from tf_bodypix.api import BodyPixModelPaths, download_model, load_model

from functions import HTM, pixel_ratio


class ImageConverter:

    def __init__(self, pub=None, show_result=False):
        ## load predictor model
        ## TODO replace all deprecated API, follow the warning message
        # load model (once)
        self.bodypix_model = load_model(download_model(
            BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16
        ))
        self.depth_thres = 0.8
        self.half_shoulder_length = 0.70/2
        self.half_chest_thickness = 0.60/2


        self.color_image = None
        self.color_image_timestamp = -1

        self.depth_image = None
        self.depth_image_timestamp = -1

        self.cv_bridge = CvBridge()
        self.publisher = pub
        self.trunk_msg = None
        self.output = {}
        self.output["radius"] = self.half_shoulder_length
        self.output["angle"] = None
        self.output["center_point"] = None
        self.output["rectangle_tips"] = None


    def depth_image_callback(self, data):
        try:
            rospy.logdebug('new depth_image, timestamp %d', data.header.stamp.secs)
            
            self.depth_image_timestamp = data.header.stamp.secs
            self.depth_image = ros_numpy.numpify(data)

        except CvBridgeError as e:
            rospy.logerr('depth_image_callback error: "%s"', e)

        if self.depth_image_timestamp == self.color_image_timestamp:
            self.pose_estimation()

    def color_image_callback(self, data):
        try:
            rospy.logdebug('new color_image, timestamp %d', data.header.stamp.secs)
            self.color_image_timestamp = data.header.stamp.secs
            self.color_image = ros_numpy.numpify(data)

        except CvBridgeError as e:
            rospy.logerr('color_image_callback error: "%s"', e)

        if self.color_image_timestamp == self.depth_image_timestamp:
            self.pose_estimation()

    def pose_estimation(self):
        rospy.logdebug('pose_estimation, timestamp %d', self.color_image_timestamp)

        rospy.logdebug(
            'depth_image.shape=%s, color_image.shape=%s', 
            self.depth_image.shape, 
            self.color_image.shape
        )
        
        result = self.bodypix_model.predict_single(self.color_image)

        # simple mask
        mask = result.get_mask(threshold=0.75)

        trunk = result.get_scaled_part_segmentation(mask, part_names=['torso_front'])
        cv2.imwrite("img2.png", trunk)

        trunk_mask = np.where(trunk > self.depth_thres, 1 , 0)
        if np.sum(trunk_mask) == 0:
            rospy.loginfo('No User Detected...')
            return
        else:
            rospy.loginfo('trunk extracted...')
        masked_depth_image = self.depth_image* trunk_mask

        # saving the trunk_mask image
        pre_trunk = 255*trunk_mask
        cv2.imwrite("img.png", 255*pre_trunk)

        

        non_zero_parts = np.array(np.nonzero(masked_depth_image))
        
        trunk_dict = {}
        for i in range(non_zero_parts.shape[1]):
            if non_zero_parts[0,i] not in trunk_dict:
                trunk_dict[non_zero_parts[0,i]] = []
            trunk_dict[non_zero_parts[0,i]].append(non_zero_parts[1,i])

        trunk_center = 0
        # TODO we should just iterate over elements instead of doing keys then __getitem__
        for j in trunk_dict.keys():
            trunk_center += np.mean(np.array(trunk_dict[j]))

        if len(trunk_dict) == 0:
            rospy.logdebug("... no human found")
            # TODO should we keep return value immutabl as empty dict?
            return
        trunk_center /= len(trunk_dict.keys())
            
        data_point_thres = 100
        angle = []
        # TODO numpy has other linear regression methods that are lightweight than LR here
        regression_model = LinearRegression()
        for row in trunk_dict:
            if len(trunk_dict[row]) <= data_point_thres:
                continue
            y = np.array(trunk_dict[row])
            x = np.arange(len(y)).reshape(-1,1)
            regression_model.fit(x,y)
            slope = regression_model.coef_
            angle.append(np.arctan(slope))
        angle = np.mean(np.array(angle))
        rospy.logdebug("angle: %f" , np.rad2deg(angle))
        y_coor = np.mean(non_zero_parts[1,:])
        rospy.logdebug("y coordinate: %f", y_coor)
        pix_real_ratio = pixel_ratio(y_coor)
        img_center = self.color_image.shape[1]/2
        x_coor = (trunk_center - img_center)*pix_real_ratio
        rospy.logdebug("x coordinate: %f", x_coor)
        center_point = np.array([x_coor/10,y_coor/1000])
        cur_HTM = HTM(angle, x_coor, y_coor)
        rectangle_tips = np.array([np.matmul(cur_HTM, np.array([-1*self.half_shoulder_length/pix_real_ratio, -1*self.half_chest_thickness/pix_real_ratio, 0, 1]).T), 
                                   np.matmul(cur_HTM, np.array([   self.half_shoulder_length/pix_real_ratio, -1*self.half_chest_thickness/pix_real_ratio, 0, 1]).T),
                                   np.matmul(cur_HTM, np.array([-1*self.half_shoulder_length/pix_real_ratio,    self.half_chest_thickness/pix_real_ratio, 0, 1]).T), 
                                   np.matmul(cur_HTM, np.array([   self.half_shoulder_length/pix_real_ratio,    self.half_chest_thickness/pix_real_ratio, 0, 1]).T)])[:,0:2]
        
        # print("User Rectangle Model: ", rectangle_tips)
        self.output["angle"] = angle
        self.output["center_point"] = center_point
        self.output["rectangle_tips"] = rectangle_tips

        # r1, r2 = 0.35/2, 0.3/2 # r1 -> turtlebot radius r2 -> human radius hardcore
        human_pos, r2 = self.output['center_point'], self.output['radius']
        if human_pos is None:
            return
        x_, y_ = human_pos
        r1 = 0.35/2
        footprint_point_array = np.array([[0,r1],[x_,r2],[x_+r2,r2],[x_+r2,-r2],[x_,-r2],[0,-r1],[-r1,-r1],[-r1,r1]]).T
        transform_theta = np.arctan2(y_,x_)
        transform_array = np.array([[np.cos(transform_theta),-np.sin(transform_theta)],[np.sin(transform_theta),np.cos(transform_theta)]])
        result_footprint = transform_array @ footprint_point_array
        rospy.set_param('footprint',result_footprint.tolist())
        print("x, y, r2: ", x_, y_, r2)
        return


def start_node():
    rospy.init_node('pose_estimator', log_level=rospy.INFO)
    rospy.loginfo('pose_estimator node started')
    
    converter = ImageConverter(show_result=False)

    # RGB image
    #   /camera/color/image_raw
    # depth image
    #   /camera/depth/image_raw
    rospy.Subscriber('/camera/color/image_raw', Image, converter.color_image_callback)
    rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, converter.depth_image_callback)
    
    # hold till system ends
    rospy.spin()

    
if __name__ == '__main__':
    try:
        start_node()
    except rospy.ROSInterruptException:
        pass
