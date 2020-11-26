#! /usr/bin/env python
import rospy
import sys
import cv2
import numpy as np

import homography

from geometry_msgs.msg import Twist
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from scipy.ndimage import center_of_mass
from math import isnan

LINE_THRESHOLD = 240
INTENSITY = 255
CAMERA_LENGTH = 1280
CAMERA_HEIGHT = 720

LIMIT = 710
LEFT_BOUND = 480
RIGHT_BOUND = 800

# Call this method before we move to check for pedestrian.
def is_at_crosswalk(colored_img):

    # get bottom row of frame
    bottom_row = colored_img[LIMIT:, LEFT_BOUND:RIGHT_BOUND]
    hsv = cv2.cvtColor(bottom_row, cv2.COLOR_BGR2HSV)

    # boundaries for red
    lower_red = np.array([0,100,100])
    upper_red = np.array([0,255,255])

    # mask the red pixels and compute the average amount of red
    red_mask = cv2.inRange(hsv,lower_red,upper_red)
    avg_red = np.sum(red_mask) / ((CAMERA_HEIGHT - LIMIT)*(RIGHT_BOUND-LEFT_BOUND))

    return avg_red > 0.8*255


def generate_com(grayscale_img, prev_com):
    _, threshed_img = cv2.threshold(grayscale_img,
                                    LINE_THRESHOLD, 
                                    INTENSITY, 
                                    cv2.THRESH_BINARY) 
                                    
    # Compute center of mass of the threshed image
    com = center_of_mass(threshed_img)

    if isnan(com[0]) or isnan(com[1]):
        com_loc = prev_com
    else:
        com_loc = (int(com[1]), int(com[0]))
        prev_com = com_loc

    return com_loc[0], com_loc[1], prev_com
