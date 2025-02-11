#!/usr/bin/env python3

import rospy,os,time,sys
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
from threading import Thread

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.color_msg import ColorMsg

class MultiCameraPublisher:
    def __init__(self):
        rospy.init_node('hand_arc_to_range', anonymous=True)
        self.subscriber = rospy.Subscriber("/cb_right_hand_control_cmd_arc",JointState,self.arc_to_range, queue_size = 1)
        self.publisher_hand_command_right_joint = rospy.Publisher("/cb_right_hand_control_cmd", JointState, queue_size=1)
    def arc_to_range(self, msg):
        #print(msg.position)
        zero = (0.0, 0.0, 0.0, 0.0)
        hand_range_r = [0.0] * (len(msg.position) + 4)
        #print(hand_range_r)
        hand_range = msg.position[:11] + zero + msg.position[11:]
        #print(hand_range)
        hand_range = list(hand_range)
        #print(hand_range)
        print((len(msg.position) + 4))
        for i in range((len(msg.position) + 4)):
            if 11 <= i <= 14 and (len(msg.position) + 4) == 20: continue
            val_r = is_within_range(hand_range[i], l20_r_min[i], l20_r_max[i])
            if l20_r_derict[i] == -1:
                hand_range_r[i] = scale_value(val_r, l20_r_min[i], l20_r_max[i], 255, 0)
            else:
                hand_range_r[i] = scale_value(val_r, l20_r_min[i], l20_r_max[i], 0, 255)
        
        #hand_range_r[10:15] = zero
        #hand_range_r = hand_range_r[:11] + zero + hand_range_r[11:]
        #print(hand_range_r)
        
        self.msg_pub.position = [0.0] * (len(msg.position) + 4)
        #print(msg_pub)
        self.msg_pub.position = hand_range_r
        print(self.msg_pub)
        self.publisher_hand_command_right_joint.publish(self.msg_pub)
if __name__ == '__main__':
    try:
        multi_cam_publisher = MultiCameraPublisher()
        rospy.spin()  # Keep the node running
    except rospy.ROSInterruptException:
        pass
   
