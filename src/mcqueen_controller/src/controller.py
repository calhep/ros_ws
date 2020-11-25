#! /usr/bin/env python

import roslib
roslib.load_manifest('mcqueen_controller')

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


class ImageConverter:

    THRESHOLD = 88
    INTENSITY = 255
    CAMERA_LENGTH = 1280
    CAMERA_HEIGHT = 720
    RED = (255,0,0)

    def __init__(self, rm):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.callback)
        self.prev_com = (640, 360)
        self.rm = rm
    
    def callback(self, image):
        try:
            camera_img = self.bridge.imgmsg_to_cv2(image, 'mono8') # image grayscaled
        except CvBridgeError as e:
            print(e)

        # Threshold a vertical slice of the camera feed on the right side
        img = camera_img[:,900:] # 640 for half the cam img
        _, threshed_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY) 

        # Compute center of mass of the threshed image
        my_com = self.generate_com(threshed_img)
        x = my_com[0]
        y = my_com[1]
        print(my_com)

        displayed_img = cv2.circle(threshed_img, my_com, 50, (255,0,0))
        cv2.imshow('guh', displayed_img)
        cv2.waitKey(3)

        # Control conditions
        if x < 125:
            self.rm.move_robot(x=0.05, z=0.6)
        elif 125 <= x and x <= 255:
            self.rm.move_robot(x=0.1, z=0)
        else:
            self.rm.move_robot(x=0.05, z=-0.6)


    # Returns center of mass of a threshed image as a tuple,
    # with the x coordinate in the 0th index and y coordinate in the 1st index
    def generate_com(self, image):
        com = center_of_mass(image)

        if isnan(com[0]) or isnan(com[1]):
            com_loc = self.prev_com
        else:
            com_loc = (int(com[1]), int(com[0]))
            self.prev_com = com_loc

        return com_loc


class RobotMovement:

    # The current design I'm thinking of would essentially
    # make it so that move_robot would be private to the client.
    # e.g. the caller would only be able to use straight(), turn(), fork()

    TURN_TIME = 2.18
    TURN_SPD = 0.85
    
    def __init__(self):
        self.move_pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)

    def move_robot(self, x=0, y=0, z=0):
        move = Twist()
        move.linear.x = x
        move.linear.y = y
        move.angular.z = z

        self.move_pub.publish(move)

    def stop_robot(self):
        self.move_robot()

    def init(self):
        self.move_robot(x=0.15)
        rospy.sleep(2.7)
        self.move_robot(x=0,z=0.85)
        rospy.sleep(2.2)
        self.stop_robot()

    # TODO: this gets us out of the starting fork for now.
    def drive(self):    
        self.move_robot(x=0.15)
        rospy.sleep(2.4)
        self.turn_left()

    # Drives the distance of the outer straight, then stops.
    def straight(self):
        self.move_robot(x=0.2)
        rospy.sleep(12.227)
        
    # Drives to where the fork in the straight is, then stops.
    def half(self):
        self.move_robot(x=0.2)
        rospy.sleep(6.125)

    # Turns 90 degrees left, then stop.
    def turn_left(self):
        self.move_robot(x=0, z=self.TURN_SPD)
        rospy.sleep(self.TURN_TIME)
        self.stop_robot()

    # Turns 90 degrees right, then stop.
    def turn_right(self):
        self.move_robot(x=0, z=-self.TURN_SPD)
        rospy.sleep(self.TURN_TIME)
        self.stop_robot()


    ### TODO: remove this, you won't need it anymore
    def move_to_com(self, com):
        print(com)
        if com[1] <= 200:
            self.move_robot(x=0.2)
        else:
            self.move_robot(x=0.2)
            if com[0] >= 740:
                self.move_robot(x=0, z=-0.7)
            elif com[0] <= 440:
                self.move_robot(x=0, z=0.7)


class PlateReader:

    ### TODO: Instantiate a Homography class in here, 
    # which will detect homographies and predict plates

    def __init__(self):
        self.plate_pub = rospy.Publisher('/license_plate', String, queue_size=1)
        self.start_string = '12345678,mcqueen,0,ABCD'
        self.stop_string = '12345678,mcqueen,-1,ABCD'

    def begin_comp(self):
        self.plate_pub.publish(self.start_string) # TODO: We should not start the competition from within PR

    def stop_comp(self):
        self.plate_pub.publish(self.stop_string)


def drive(rm):
    # a control sequence that moves the robot around the track.
    rm.drive() # get out of the fork
    rm.half()
    rm.turn_left()
    rm.straight()
    rm.turn_left()
    rm.straight()
    rm.turn_left()
    rm.straight()
    rm.turn_left()
    rm.straight()

def main(args):
    # TODO: Is it good practice to "start" the competition from the PR class?
    # We can consider moving and instantiating all classes in a higher level control class.
    rm = RobotMovement()
    pr = PlateReader() 

    # Initialize the node.
    rospy.init_node('controller')
    init_rate = rospy.Rate(1)

    # Begin the competition.
    init_rate.sleep()
    pr.begin_comp()
    
    # start movin bruh
    rospy.sleep(2.5)
    rm.init()
    ic = ImageConverter(rm)
    
    rospy.sleep(45)
    # Stop the robot and the competition.
    # rm.stop_robot()
    # pr.stop_comp()

    init_rate.sleep()


if __name__ == '__main__':
    main(sys.argv)