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

    LINE_THRESHOLD = 240
    INTENSITY = 255
    CAMERA_LENGTH = 1280
    CAMERA_HEIGHT = 720
    RED_BOUNDARY = ([17,15,100],[50,56,200])

    def __init__(self, rm):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.callback)
        self.prev_com = (640, 360)
        self.rm = rm
    

    def callback(self, image):
        try:
            grayscale_img = self.bridge.imgmsg_to_cv2(image, 'mono8')
            colored_img = self.bridge.imgmsg_to_cv2(image, 'bgr8')
        except CvBridgeError as e:
            print(e)

        # Threshold a vertical slice of the camera feed on the right side
        # camera_img[y:,x:] (row,col)
        _, threshed_img = cv2.threshold(grayscale_img[650:,900:],
                                        self.LINE_THRESHOLD, 
                                        self.INTENSITY, 
                                        cv2.THRESH_BINARY) 

        # Compute center of mass of the threshed image
        my_com = self.generate_com(threshed_img)
        x = my_com[0]
        y = my_com[1]
        #print(my_com)

        # If the red bar of crosswalk is detected, check for pedestrian
        self.crosswalk(colored_img)

        # Control conditions
        if x < 120:
            self.rm.move_robot(x=0.0, z=.45)
        elif 120 <= x and x <= 235:
            self.rm.move_robot(x=0.125, z=0)
        else:
            self.rm.move_robot(x=0., z=-.45)


    # Call this method before we move to check for pedestrian.
    def crosswalk(self, colored_img):

        LIMIT = 710
        LEFT_BOUND = 480
        RIGHT_BOUND = 800
        LAMBDA = 0.8
        
        # get bottom row of frame
        bottom_row = colored_img[LIMIT:, LEFT_BOUND:RIGHT_BOUND]
        hsv = cv2.cvtColor(bottom_row, cv2.COLOR_BGR2HSV)

        # boundaries for red
        lower_red = np.array([0,100,100])
        upper_red = np.array([0,255,255])

        # mask the red pixels and compute the average amount of red
        red_mask = cv2.inRange(hsv,lower_red,upper_red)
        avg_red = np.sum(red_mask) / ((self.CAMERA_HEIGHT - LIMIT)*(RIGHT_BOUND-LEFT_BOUND))

        # If we are more than 80% red, and we are not currently in a crosswalk,
        # that means stop and wait for pedestrian to cross, then go.
        if (avg_red > 0.8*255):
            self.rm.stop_robot()
            rospy.sleep(5)
            # TODO: Manual control of robot here based on homography/color masking of pedestrians
            self.rm.move_robot(x=0.05)


    # Returns CoM as a tuple, x coordinate in the 0th index and y coordinate in the 1st index
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
    
    rospy.sleep(300)

    # Stop the robot and the competition.
    rm.stop_robot()
    pr.stop_comp()
    rospy.sleep(1)


if __name__ == '__main__':
    main(sys.argv)
