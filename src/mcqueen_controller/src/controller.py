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

# class image_converter:

#     def __init__(self):
#         self.bridge = CvBridge()
#         self.image_sub = rospy.Subscriber('/robot/camera1/image_raw',Image,self.callback)
    
#     def callback(self, image):
#         try:
#             cv_image = self.bridge.imgmsg_to_cv2(image, 'mono8') # image grayscaled
#         except CvBridgeError as e:
#             print(e)

#         cv2.imshow("Image window", cv_image)
#         cv2.waitKey(3)

class robot_movement:

    def __init__(self):
        self.move_pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)

    def move_robot(self):
        move = Twist()
        move.linear.x = 0.1

        self.move_pub.publish(move)

def main(args):
    rm = robot_movement()
    rospy.init_node('controller')
    rate = rospy.Rate(2)
    
    while not rospy.is_shutdown():
        rm.move_robot()
        rate.sleep()


if __name__ == '__main__':
    main(sys.argv)