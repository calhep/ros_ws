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
            camera_image = self.bridge.imgmsg_to_cv2(image, 'mono8') # image grayscaled
        except CvBridgeError as e:
            print(e)

        # Rectangular strip coordinates
        corner_tl = (480, 520)
        corner_br = (960, 520)
        color = self.RED

        img = cv2.rectangle(camera_image, corner_tl, corner_br, color, -1)
        
        cv2.imshow('guh', img)
        cv2.waitKey(3)
       

    def mask_frame(self, image, threshold, intensity, mask_type):
        _, masked_frame = cv2.threshold(image, threshold, intensity, mask_type)
        
        return masked_frame

    def generate_com(self, image):
        com = center_of_mass(image)

        if isnan(com[0]) or isnan(com[1]):
            com_loc = self.prev_com
        else:
            com_loc = (int(com[1]), int(com[0]) + 620)
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
    ic = ImageConverter(rm)

    # Initialize the node.
    rospy.init_node('controller')
    init_rate = rospy.Rate(1)

    # Begin the competition.
    init_rate.sleep()
    pr.begin_comp()

    #drive(rm)
    # rm.turn_left()
    # rospy.sleep(5)
    # rm.turn_right()
    
    rospy.sleep(30)
    # Stop the robot and the competition.
    rm.stop_robot()
    pr.stop_comp()

    init_rate.sleep()


if __name__ == '__main__':
    main(sys.argv)