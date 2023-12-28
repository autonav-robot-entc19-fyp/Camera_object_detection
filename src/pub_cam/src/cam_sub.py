#!/usr/bin/env python3

import rospy
from std_msgs.msg import String

def callback(data):
    rospy.loginfo("Detected Object: %s", data.data)

def listener():
    rospy.init_node('object_detection_listener', anonymous=True)

    rospy.Subscriber("/detected_objects", String, callback)

    rospy.spin()

if __name__ == '__main__':
    listener()
