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

class image_converter:

    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/R1/pi_camera/image_raw', Image,self.callback)
        self.threshold = 88
        self.intensity = 255
    
    def callback(self, image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image, 'mono8') # image grayscaled
        except CvBridgeError as e:
            print(e)
        
        masked_image = self.mask_frame(cv_image)

        cv2.imshow("Image window", masked_image)
        cv2.waitKey(3)

    def mask_frame(self, image):

        _, masked_frame = cv2.threshold(image, self.threshold, self.intensity, cv2.THRESH_BINARY_INV)
        
        return masked_frame


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

    # rm.move_robot(x=0.15)
    # rospy.sleep(2.7)
    # rm.move_robot(x=0,z=0.85)
    # rospy.sleep(2.2)
    # rm.move_robot(x=0.2, z=0)
    # rospy.sleep(6)
    rospy.sleep(10)

    # rm.stop_robot()
    pr.stop_comp()

    init_rate.sleep()

if __name__ == '__main__':
    main(sys.argv)