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

    rospy.init_node('controller')
    init_rate = rospy.Rate(1)
    turn_rate = rospy.Rate(2)

    init_rate.sleep()
    pr.begin_comp()

    start_time = rospy.get_time()

    rm.move_robot(x=0.15)
    init_rate.sleep()
    init_rate.sleep()
    init_rate.sleep()
    rm.move_robot(x=0,z=0.85)
    init_rate.sleep()
    init_rate.sleep()

    while rospy.get_time() < start_time + 20:
        
        rm.move_robot(x=0.1, z=0)
        init_rate.sleep()

    rm.stop_robot()
    pr.stop_comp()

    init_rate.sleep()

if __name__ == '__main__':
    main(sys.argv)