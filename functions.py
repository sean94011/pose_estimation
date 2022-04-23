from math import cos, sin
import numpy as np


def pixel_ratio(depth):
    slope = 1.0864224505928857e-05
    intercept = -4.2662924901173197e-07
    return slope*depth + intercept

def HTM(theta_z, x, y):
    return np.array([[cos(theta_z), -1*sin(theta_z), 0, x],
                     [sin(theta_z),    cos(theta_z), 0, y],
                     [           0,            0,    1, 0],
                     [           0,            0,    0, 1]])
