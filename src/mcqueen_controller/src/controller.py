#! /usr/bin/env python

import roslib
roslib.load_manifest('mcqueen_controller')

import rospy
import sys
import cv2
import numpy as np

import homography as hm
import callback_handling as ch

from homography import Homography
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from scipy.ndimage import center_of_mass
from math import isnan


class ImageConverter:

    def __init__(self, rm):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.callback)
        self.prev_com = (640, 360)
        self.rm = rm
  
        self.plate_num = 0
        #self.hm = hm.Homography()

    def callback(self, image):
        try:
            grayscale_img = self.bridge.imgmsg_to_cv2(image, 'mono8')
            colored_img = self.bridge.imgmsg_to_cv2(image, 'bgr8')
        except CvBridgeError as e:
            print(e)

        # if self.hm.detect_features(grayscale_img, self.plate_num):
        #     self.plate_num += 1

        # If the red bar of crosswalk is detected, check for pedestrian
        # if ch.is_at_crosswalk(colored_img):
        #     self.rm.stop_robot()
        #     rospy.sleep(5)
        #     # TODO: Manual control of robot here based on homography/color masking of pedestrians
        #     self.rm.move_robot(x=0.05)

        x, y, self.prev_com = ch.generate_com(grayscale_img[650:,900:], self.prev_com)

        # Control conditions
        if x < 120:
            self.rm.move_robot(x=0.075, z=1)
        elif 120 <= x and x <= 235:
            self.rm.move_robot(x=0.3, z=0)
        else:
            self.rm.move_robot(x=0.075, z=-1)


class RobotMovement:

    def __init__(self):
        self.move_pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)

    # Publishes a move command.
    def move_robot(self, x=0, y=0, z=0):
        move = Twist()
        move.linear.x = x
        move.linear.y = y
        move.angular.z = z

        self.move_pub.publish(move)

    # Stops the robot.
    def stop_robot(self):
        self.move_robot()

    # Gets us into the outer loop.
    def init(self):
        self.move_robot(x=0.15)
        rospy.sleep(2.5)
        self.move_robot(x=0,z=0.85)
        rospy.sleep(2.2)
        self.stop_robot()


class PlateReader:

    # TODO: Instantiate a Homography class in here, 
    # which will detect homographies and predict plates

    def __init__(self):
        self.plate_pub = rospy.Publisher('/license_plate', String, queue_size=1)
        self.start_string = '12345678,mcqueen,0,ABCD'
        self.stop_string = '12345678,mcqueen,-1,ABCD'

    def begin_comp(self):
        self.plate_pub.publish(self.start_string) # TODO: We should not start the competition from within PR

    def stop_comp(self):
        self.plate_pub.publish(self.stop_string)


def main(args):
    # TODO: We can consider moving and instantiating all classes in a higher level control class.
    # TODO: make sure that our code executes within the competition timing blocks.

    rm = RobotMovement()
    pr = PlateReader() 

    # Initialize the node.
    rospy.init_node('controller')
    init_rate = rospy.Rate(1)

    # Begin the competition.
    init_rate.sleep()
    pr.begin_comp()
    
    # # start movin bruh
    rospy.sleep(1)
    rm.init()
    ic = ImageConverter(rm)
    #hm = Homography()
    
    rospy.sleep(600)

    # Stop the robot and the competition.
    rm.stop_robot()
    pr.stop_comp()
    rospy.sleep(1)


if __name__ == '__main__':
    main(sys.argv)
