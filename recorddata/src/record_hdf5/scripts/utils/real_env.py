#! /usr/bin/env python3
import time,os,sys
import numpy as np
import collections
import matplotlib.pyplot as plt
import dm_env
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from color_msg import ColorMsg
from robot_utils import Recorder, ImageRecorder
from pyquaternion import Quaternion

from constants import DT,LEFT_HAND_JOINT,RIGHT_HAND_JOINT,LEFT_ARM_JOINT,RIGHT_ARM_JOINT
import pyagxrobots


import IPython
e = IPython.embed

class RealEnv:
    def __init__(self):
        self.recorder_arms = Recorder('left', init_node=False)
        self.image_recorder = ImageRecorder(init_node=False)
        #self.recorder_right = Recorder('right', init_node=False)

    def reset(self, fake=False,t="aloha"):
        if not fake:
            pass
        if t=="aloha":
            ob = self.get_aloha_observation()
        else:
            ob = self.get_observation()
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=0,
            discount=None,
            observation=ob)
    
    def step(self, action, base_action=None, get_tracer_vel=False, get_obs=True,t="aloha"):
        if get_obs:
            if t=="aloha":
                obs = self.get_aloha_observation()
            else:
                obs = self.get_observation()
        else:
            obs = None
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=0,
            discount=None,
            observation=obs)
    # hummanplus是手与臂数据分开
    def get_qpos(self):
        arm = self.recorder_arms.qpos
        l_qpos = arm[:6]
        r_qpos = arm[7:13]
        return np.concatenate([l_qpos, r_qpos])
    # aloha是手与臂数据和在一起
    def get_aloha_qpos(self):
        arm = self.recorder_arms.qpos
        l_qpos = arm[:6]
        r_qpos = arm[7:13]
        return np.concatenate([l_qpos, r_qpos, self.recorder_arms.left_hand_qpos,  self.recorder_arms.right_hand_qpos])
    def get_qvel(self):
        arm_qvel = [0]*12
        return np.array(arm_qvel)
    def get_aloha_qvel(self):
        arm_qvel = [0]*12
        hand_qvel = [0]*LEFT_HAND_JOINT+[0]*RIGHT_HAND_JOINT
        return np.concatenate([arm_qvel,hand_qvel])
    def get_effort(self):
        t = [0]*LEFT_ARM_JOINT+[0]*RIGHT_ARM_JOINT+[0]*LEFT_HAND_JOINT+[0]*RIGHT_HAND_JOINT
        return np.array(t)
    def get_images(self):
        return self.image_recorder.get_images()
    def get_base_vel(self):
        t = [0]*LEFT_ARM_JOINT+[0]*RIGHT_ARM_JOINT+[0]*LEFT_HAND_JOINT+[0]*RIGHT_HAND_JOINT
        return np.array(t)
    def get_observation(self, get_tracer_vel=False):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos()
        obs['qvel'] = self.get_qvel()
        obs['effort'] = self.get_effort()
        obs['images'] = self.get_images()
        # obs['base_vel_t265'] = self.get_base_vel_t265()
        obs['base_vel'] = self.get_base_vel()
        return obs
    def get_aloha_observation(self):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_aloha_qpos()
        obs['qvel'] = self.get_aloha_qvel()
        obs['effort'] = self.get_effort()
        obs['images'] = self.get_images()
        # obs['base_vel_t265'] = self.get_base_vel_t265()
        obs['base_vel'] = self.get_base_vel()
        return obs
    def get_action(self):
        arms_action = self.recorder_arms.arm_action
        left_hand_action = self.recorder_arms.left_hand_action
        right_hand_action = self.recorder_arms.right_hand_action
        action = arms_action+left_hand_action+right_hand_action
        return action
    def clean_data(self):
        self.recorder_arms.clean_recorder()