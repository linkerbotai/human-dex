#! /usr/bin/env python3
import rospy,sys,os
import numpy as np
import time
from collections import deque
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from constants import DT,LEFT_HAND_JOINT,RIGHT_HAND_JOINT,LEFT_ARM_JOINT,RIGHT_ARM_JOINT


import IPython
e = IPython.embed

class ImageRecorder:
    def __init__(self,camera_name, init_node=True, is_debug=False):
        from collections import deque
        import rospy
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image
        self.is_debug = is_debug
        self.bridge = CvBridge()
        self.camera_names = ['cam_top', 'cam_right', 'cam_front']
        self.camera_names = camera_name
        if init_node:
            rospy.init_node('image_recorder', anonymous=True)
        for cam_name in self.camera_names:
            setattr(self, f'{cam_name}_image', None)
            setattr(self, f'{cam_name}_secs', None)
            setattr(self, f'{cam_name}_nsecs', None)
            if cam_name == 'cam_high':
                callback_func = self.image_cb_cam_high
            elif cam_name == 'cam_low':
                callback_func = self.image_cb_cam_low
            elif cam_name == 'cam_left':
                callback_func = self.image_cb_cam_left
            elif cam_name == 'cam_right':
                callback_func = self.image_cb_cam_right
            elif cam_name == 'cam_top':
                callback_func = self.image_cb_cam_top
            elif cam_name == 'cam_front':
                callback_func = self.image_cb_cam_front
            elif cam_name == 'cam_hand':
                callback_func = self.image_cb_cam_hand
            else:
                raise NotImplementedError
            rospy.Subscriber(f"/camera/rgb/image_raw/{cam_name}", Image, callback_func)
        time.sleep(0.5)
    
    def image_cb(self, cam_name, data):
        setattr(self, f'{cam_name}_image', self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough'))
        setattr(self, f'{cam_name}_secs', data.header.stamp.secs)
        setattr(self, f'{cam_name}_nsecs', data.header.stamp.nsecs)
        
    def image_cb_cam_high(self, data):
        cam_name = 'cam_high'
        return self.image_cb(cam_name, data)

    def image_cb_cam_low(self, data):
        cam_name = 'cam_low'
        return self.image_cb(cam_name, data)

    def image_cb_cam_left(self, data):
        cam_name = 'cam_left'
        return self.image_cb(cam_name, data)

    def image_cb_cam_right(self, data):
        cam_name = 'cam_right'
        return self.image_cb(cam_name, data)
    def image_cb_cam_top(self, data):
        cam_name = 'cam_top'
        return self.image_cb(cam_name, data)
    def image_cb_cam_front(self, data):
        cam_name = 'cam_front'
        return self.image_cb(cam_name, data)
    def image_cb_cam_hand(self, data):
        cam_name = 'cam_hand'
        return self.image_cb(cam_name, data)
    def get_images(self):
        image_dict = dict()
        for cam_name in self.camera_names:
            image_dict[cam_name] = getattr(self, f'{cam_name}_image')
        return image_dict
    
class Recorder:
    def __init__(self, side, init_node=True, is_debug=False):
        from collections import deque
        import rospy
        from sensor_msgs.msg import JointState
        # from interbotix_xs_msgs.msg import JointGroupCommand, JointSingleCommand
        self.side = side
        self.secs = None
        self.nsecs = None
        self.name = None
        self.qpos = None
        self.effort = None
        self.arm_command = None
        self.gripper_command = None
        self.left_hand_name = None
        self.left_hand_qpos = None
        self.left_hand_qvel = None
        self.left_hand_effort = None
        self.right_hand_name = None
        self.right_hand_qpos = None
        self.right_hand_qvel = None
        self.right_hand_effort = None
        self.arm_action = None
        self.left_hand_action = None
        self.right_hand_action = None
        self.is_debug = is_debug

        if init_node:
            rospy.init_node('recorder', anonymous=True)
        # state
        rospy.Subscriber("/joint_states", JointState, self.joint_state_cb, queue_size=10)
        rospy.Subscriber("/cb_left_hand_state", JointState, self.left_hand_state_cb,queue_size=10)
        rospy.Subscriber("/cb_right_hand_state", JointState, self.right_hand_state_cb,queue_size=10)
        # action
        rospy.Subscriber("/cb_arm_control_cmd", JointState, self.arms_action_cb,queue_size=10)
        rospy.Subscriber("/cb_left_hand_control_cmd", JointState, self.left_hand_action_cb,queue_size=10)
        rospy.Subscriber("/cb_right_hand_control_cmd", JointState, self.right_hand_action_cb,queue_size=10)

        time.sleep(0.1)
    # state
    def joint_state_cb(self, data):
        self.name = data.name
        self.qpos = data.position
        self.qvel = data.velocity
        self.effort = data.effort
        self.data = data


    def left_hand_state_cb(self, data):
        tmp_data = data
        self.left_hand_name = tmp_data.name[:LEFT_HAND_JOINT]
        self.left_hand_qpos = tmp_data.position[:LEFT_HAND_JOINT]
        self.left_hand_qvel = tmp_data.velocity[:LEFT_HAND_JOINT]
        self.left_hand_effort = tmp_data.effort[:LEFT_HAND_JOINT]

    def right_hand_state_cb(self, data):
        tmp_data = data
        self.right_hand_name = tmp_data.name[:RIGHT_HAND_JOINT]
        self.right_hand_qpos = tmp_data.position[:RIGHT_HAND_JOINT]
        self.right_hand_qvel = tmp_data.velocity[:RIGHT_HAND_JOINT]
        self.right_hand_effort = tmp_data.effort[:RIGHT_HAND_JOINT]

    # action
    def arms_action_cb(self, data):
        self.arm_action = data.position

    def left_hand_action_cb(self, data):
        self.left_hand_action = data.position[:LEFT_HAND_JOINT]

    def right_hand_action_cb(self, data):
        self.right_hand_action = data.position[:RIGHT_HAND_JOINT]

    def clean_recorder(self):
        self.secs = None
        self.nsecs = None
        self.name = None
        self.qpos = None
        self.effort = None
        self.arm_command = None
        self.gripper_command = None
        self.left_hand_name = None
        self.left_hand_qpos = None
        self.left_hand_qvel = None
        self.left_hand_effort = None
        self.right_hand_name = None
        self.right_hand_qpos = None
        self.right_hand_qvel = None
        self.right_hand_effort = None
        self.arm_action = None
        self.left_hand_action = None
        self.right_hand_action = None

        