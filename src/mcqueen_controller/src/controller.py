#! /usr/bin/env python

import rospy
from std_msgs.msg import String


def callback(data):
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)


def listener():
    rospy.init_node('clock_listener', anonymous=True)
    rospy.Subscriber('/clock', String, callback)


if __name__ == '__main__':
    rate = rospy.Rate(2)
    listener()
    rospy.spin()
