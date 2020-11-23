#! /usr/bin/env python

import roslib
roslib.load_manifest('mcqueen_controller')

import rospy
import sys
import cv2
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from scipy.ndimage import center_of_mass
from math import isnan

class image_converter:

    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/R1/pi_camera/image_raw', Image,self.callback)
        self.threshold = 88
        self.intensity = 255
        self.prev_com = (160, 120)
    
    def callback(self, image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image, 'mono8') # image grayscaled
        except CvBridgeError as e:
            print(e)
        
        display_image = self.mask_frame(cv_image, self.intensity)
        masked_image = self.mask_frame(cv_image, 1)
        center_of_mass = self.generate_com(masked_image[-100:])

        cv2.imshow("Image window", display_image)
        cv2.waitKey(3)

    def mask_frame(self, image, intensity):
        _, masked_frame = cv2.threshold(image, self.threshold, intensity, cv2.THRESH_BINARY_INV)
        
        return masked_frame

    def generate_com(self, image):
        com = center_of_mass(image)

        if isnan(com[0]) or isnan(com[1]):
            com_loc = self.prev_com
        else:
            com_loc = (int(com[1]), int(com[0]) + 140)
            self.prev_com = com_loc

        return com_loc


class robot_movement:

    def __init__(self):
        self.move_pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)

    def move_robot(self, x=0, y=0, z=0):
        move = Twist()
        move.linear.x = x
        move.linear.y = y
        move.angular.z = z

        self.move_pub.publish(move)

    def stop_robot(self):
        self.move_robot(x=0,y=0,z=0)

class robot_timer:

    def __init__(self):
        self.timer_sub = rospy.Subscriber('/clock', String, queue_size=1)

class plate_reader:

    def __init__(self):
        self.plate_pub = rospy.Publisher('/license_plate', String, queue_size=1)
        self.start_string = '12345678,mcqueen,0,ABCD'
        self.stop_string = '12345678,mcqueen,-1,ABCD'

    def begin_comp(self):
        self.plate_pub.publish(self.start_string)

    def stop_comp(self):
        self.plate_pub.publish(self.stop_string)

def main(args):
    rm = robot_movement()
    # rt = robot_timer()
    pr = plate_reader()
    ic = image_converter()

    rospy.init_node('controller')
    init_rate = rospy.Rate(1)

    init_rate.sleep()
    pr.begin_comp()

    # turn left onto main road go straight till corner
    rm.move_robot(x=0.15)
    rospy.sleep(2.7)
    rm.move_robot(x=0,z=0.85)
    rospy.sleep(2.2)
    rm.move_robot(x=0.2, z=0)
    rospy.sleep(6)

    #turn left go straight
    rm.move_robot(x=0, z=0.85)
    rospy.sleep(2.15)
    rm.move_robot(x=0.2, z=0)
    rospy.sleep(12.25)

    #turn left go straight
    rm.move_robot(x=0, z=0.85)
    rospy.sleep(2.2)
    rm.move_robot(x=0.2, z=0)
    rospy.sleep(12.25)

    #turn left go straight
    rm.move_robot(x=0, z=0.85)
    rospy.sleep(2.21)
    rm.move_robot(x=0.2, z=0)
    rospy.sleep(12.45)

    #turn left go straight
    rm.move_robot(x=0, z=0.85)
    rospy.sleep(2.22)
    rm.move_robot(x=0.2, z=0)
    rospy.sleep(12.45)

    rm.stop_robot()
    pr.stop_comp()

    init_rate.sleep()

if __name__ == '__main__':
    main(sys.argv)