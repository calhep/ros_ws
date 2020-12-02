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
        self.image_sub = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.callback, queue_size=1, buff_size=1000000)
        self.prev_com = (640, 360)
        self.rm = rm

        self.lower_blue = np.array([100,0,0])
        self.upper_blue = np.array([150,255,255])

        # pedestrian flags
        self.crosswalk = False
        self.waiting_for_blue = False
        self.leaving = False
        self.stop = False

    def callback(self, image):
        try:
            grayscale_img = self.bridge.imgmsg_to_cv2(image, 'mono8')
            colored_img = self.bridge.imgmsg_to_cv2(image, 'bgr8')
        except CvBridgeError as e:
            print(e)

        if not self.crosswalk and not self.leaving:
            self.crosswalk = ch.is_at_crosswalk(colored_img)
            if self.crosswalk:
                print("Initial waiting.")
                self.waiting_for_blue = True

        if self.crosswalk and not self.leaving:
            frame = colored_img[400:460,540:740]
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            blue_mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)

            cv2.imshow('v', blue_mask)
            cv2.waitKey(1)

            current_blue = np.sum(blue_mask)
            if current_blue > 5000:
                # we have seen blue, wait for blue to disappear
                print("blue in frame, please wait for cross")
                self.stop = True
                self.waiting_for_blue = False
                self.rm.stop_robot()

            if self.waiting_for_blue:
                print("waiting for blue")
                self.stop = True
                self.rm.stop_robot()
            else:
                print("blue has appeared, wait.")
                if current_blue == 0:
                    print("no blue detected go")
                    self.stop = False
                    self.crosswalk = False
                    self.rm.move_robot(x=0.15)
                    rospy.sleep(4)
                    self.leaving = True
        
        x, y, self.prev_com = ch.generate_com(grayscale_img[:,800:], self.prev_com)
        # displayed_img = cv2.circle(grayscale_img, (x+750,y), 50, (255,0,0))
        # cv2.imshow('g',displayed_img)
        # cv2.waitKey(3)

        # Control conditions
        if not self.stop:
            if x < 220:
                self.rm.move_robot(x=0.05, z=0.7)
                self.leaving = False
            elif x > 250:
                self.rm.move_robot(x=0.05, z=-0.7)
            else:
                self.rm.move_robot(x=0.1, z=0)
       


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
        rospy.sleep(2.3)
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
