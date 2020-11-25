#! /usr/bin/env python

import roslib
roslib.load_manifest('mcqueen_controller')

import rospy
import sys
import cv2
import numpy as np
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from scipy.ndimage import center_of_mass
from math import isnan


class ImageConverter:

    THRESHOLD = 88
    INTENSITY = 255

    ### TODO: Reevaluated whether we will need the CoM functions

    def __init__(self, rm):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/R1/pi_camera/image_raw', Image,self.callback)
        self.prev_com = (640, 360)
        self.rm = rm
    
    def callback(self, image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image, 'mono8') # image grayscaled
        except CvBridgeError as e:
            print(e)

        def f(x):
            if x < 80:
                return 255
            else:
                return x
        
        v_func = np.vectorize(f)
        threshed_image = np.float32(v_func(cv_image))
        
        display_image = self.mask_frame(threshed_image, self.THRESHOLD,self.INTENSITY, cv2.THRESH_BINARY_INV)
        center_of_mass = self.generate_com(display_image[-100:])

        self.rm.move_to_com(center_of_mass)

        cv2.imshow("Image window", cv2.circle(display_image, center_of_mass, 50, (0, 0, 255)))
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


    TURN_TIME = 2.178
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
        rospy.sleep(12.1)
        
    # Drives to where the fork in the straight is, then stops.
    def half(self):
        self.move_robot(x=0.2)
        rospy.sleep(6.05)

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

    # shut up and drive. this is where something
    # like a control loop would be. 
    # (Or a method that calls a control loop)
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
    
    

    # Stop the robot and the competition.
    rm.stop_robot()
    pr.stop_comp()

    init_rate.sleep()


if __name__ == '__main__':
    main(sys.argv)