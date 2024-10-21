import time,sys,os,rospy
import numpy as np
import collections
import matplotlib.pyplot as plt
import dm_env
from pyquaternion import Quaternion
from sensor_msgs.msg import Image,JointState
from std_msgs.msg import Header, String
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from HIT.constants import DT, LEFT_HAND_JOINT, RIGHT_HAND_JOINT, LEFT_ARM_JOINT, RIGHT_ARM_JOINT
from robot_utils import Recorder, ImageRecorder
#from robot_utils import setup_master_bot, setup_puppet_bot, move_arms, move_grippers

import pyagxrobots


import IPython
e = IPython.embed

class RealEnv:
    """
    Environment for real robot bi-manual manipulation
    Action space:      [left_arm_qpos (6),             # absolute joint position
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_qpos (6),            # absolute joint position
                        right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ left_arm_qpos (6),          # absolute joint position
                                        left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                        right_arm_qpos (6),         # absolute joint position
                                        right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                        left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                        right_arm_qvel (6),         # absolute joint velocity (rad)
                                        right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                        "images": {"cam_high": (480x640x3),        # h, w, c, dtype='uint8'
                                   "cam_low": (480x640x3),         # h, w, c, dtype='uint8'
                                   "cam_left_wrist": (480x640x3),  # h, w, c, dtype='uint8'
                                   "cam_right_wrist": (480x640x3)} # h, w, c, dtype='uint8'
    """

    def __init__(self, init_node, setup_robots=True, setup_base=False):
        self.recorder_arms = Recorder('left', init_node=False)
        self.image_recorder = ImageRecorder(init_node=False)
        self.left_hand_pub = rospy.Publisher("/cb_left_hand_control_cmd", JointState, queue_size=10)
        self.right_hand_pub = rospy.Publisher("/cb_right_hand_control_cmd", JointState, queue_size=10)
        self.arm_pub = rospy.Publisher("/cb_arm_control_cmd", JointState, queue_size=10)

    def joint_msg(self,action, name):
        joint_state = JointState()
        joint_state.header = Header()
        # 自动生成时间戳和序列号
        joint_state.header.stamp = rospy.Time.now()  # 获取当前时间
        joint_state.header.seq = rospy.get_time()    # 可以使用 get_time() 生成序列号，但通常 seq 是消息流中递增的
        joint_state.header.frame_id = "world"
        joint_state.header.stamp = rospy.Time.now()
        joint_state.name = name
        joint_state.position = action
        if len(name) > 9:
            joint_state.velocity = [2.0, 2.0, 2.0, 2.0, 2.0,
                                    7.168999671936035, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
        joint_state.effort = []
        print("names")
        print(name)
        print(joint_state)
        return joint_state


        
    '''
    def setup_t265(self):
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        # if only pose stream is enabled, fps is higher (202 vs 30)
        cfg.enable_stream(rs.stream.pose)
        self.pipeline.start(cfg)
    
    def setup_dxl(self):
        self.dxl_client = DynamixelClient([1, 2], port='/dev/ttyDXL_wheels', lazy_connect=True)
        self.wheel_r = 0.101 / 2  # 101 mm is the diameter
        self.base_r = 0.622  # 622 mm is the distance between the two wheels
    
    def setup_base(self):
        self.tracer = pyagxrobots.pysdkugv.TracerBase()
        self.tracer.EnableCAN()

    def setup_robots(self):
        setup_puppet_bot(self.puppet_bot_left)
        setup_puppet_bot(self.puppet_bot_right)
    '''
    def get_qpos(self):
        arm = self.recorder_arms.qpos
        l_qpos = arm[:6]
        r_qpos = arm[7:13]
        return np.concatenate([l_qpos, r_qpos])

    def get_aloha_qpos(self):
        arm = self.recorder_arms.qpos
        l_qpos = arm[:6]
        r_qpos = arm[6:]
        return np.concatenate([l_qpos, r_qpos, self.recorder_arms.left_hand_qpos,  self.recorder_arms.right_hand_qpos])
    def get_qvel(self):
        arm_qvel = [0]*LEFT_ARM_JOINT + [0]*RIGHT_ARM_JOINT
        return np.array(arm_qvel)
    def get_aloha_qvel(self):
        arm_qvel = [0]*LEFT_ARM_JOINT + [0]*RIGHT_ARM_JOINT
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
        obs['qpos'] = self.get_aloha_qpos()
        obs['qvel'] = self.get_aloha_qvel()
        obs['effort'] = self.get_effort()
        obs['images'] = self.get_images()
        # obs['base_vel_t265'] = self.get_base_vel_t265()
        obs['base_vel'] = self.get_base_vel()
        return obs

    def get_reward(self):
        return 0

    def reset(self, fake=False):
        # if not fake:
        #     # Reboot puppet robot gripper motors
        #     self.puppet_bot_left.dxl.robot_reboot_motors("single", "gripper", True)
        #     self.puppet_bot_right.dxl.robot_reboot_motors("single", "gripper", True)
        #     self._reset_joints()
        #     self._reset_gripper()
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=0,
            discount=None,
            observation=self.get_observation())

    def step(self, action, base_action=None, get_tracer_vel=False, get_obs=True):
        
        # state_len = int(len(action) / 2) # 
        # arms_action = action[:state_len] # 12
        # arms_len = int(len(arms_action) / 2)
        # left_arm_position = arms_action[:arms_len] # 左臂数据
        # right_arm_position = arms_action[arms_len:] # 右臂数据
        # hands_action = action[state_len:] # 12
        # hands_len = int(len(hands_action) / 2)
        # left_hand_position = hands_action[:hands_len] # 左手数据
        # print("*"*20)
        # print(len(left_hand_position))
        # right_hand_position = hands_action[hands_len:] # 右手数据
        # 截取双臂数据
        arms_action = action[:12]
        # 截取双手数据
        hands_action = action[12:]
        # 左手数据
        left_hand_position = hands_action[:LEFT_HAND_JOINT]
        right_hand_position = hands_action[RIGHT_HAND_JOINT:]

        # 遍历列表并修改小于 0.0 的元素
        for i, value in enumerate(left_hand_position):
            if value < 0.0:
                left_hand_position[i] = 0.0
        for j, v in enumerate(right_hand_position):
            if v < 0.0:
                right_hand_position[j] = 0.0
        # 左手数据
        left_j=self.joint_msg(action=left_hand_position,name=self.recorder_arms.left_hand_name)
        #print(f"左手数据：{left_j}")
        self.left_hand_pub.publish(left_j)
        # 右手数据
        right_j = self.joint_msg(action=right_hand_position,name=self.recorder_arms.right_hand_name)
        #print(f"右手数据：{right_j}")
        self.right_hand_pub.publish(right_j)
        # 双臂数据
        arms_j = self.joint_msg(action=arms_action, name=self.recorder_arms.name)
        self.arm_pub.publish(arms_j)
        
        if base_action is not None:
            # linear_vel_limit = 1.5
            # angular_vel_limit = 1.5
            # base_action_linear = np.clip(base_action[0], -linear_vel_limit, linear_vel_limit)
            # base_action_angular = np.clip(base_action[1], -angular_vel_limit, angular_vel_limit)
            base_action_linear, base_action_angular = base_action
            #self.tracer.SetMotionCommand(linear_vel=base_action_linear, angular_vel=base_action_angular)
        time.sleep(DT)
        if get_obs:
            obs = self.get_observation(get_tracer_vel)
        else:
            obs = None
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=obs)

    def get_action(self):
        arms_action = self.recorder_arms.arm_action
        left_hand_action = self.recorder_arms.left_hand_action
        right_hand_action = self.recorder_arms.right_hand_action
        action = arms_action+left_hand_action+right_hand_action
        return action
    def clean_data(self):
        self.recorder_arms.clean_recorder()



def make_real_env(init_node, setup_robots=True, setup_base=False):
    env = RealEnv(init_node, setup_robots, setup_base)
    return env


def test_real_teleop():
    """
    Test bimanual teleoperation and show image observations onscreen.
    It first reads joint poses from both master arms.
    Then use it as actions to step the environment.
    The environment returns full observations including images.

    An alternative approach is to have separate scripts for teleoperation and observation recording.
    This script will result in higher fidelity (obs, action) pairs
    """

    onscreen_render = True
    render_cam = 'cam_left_wrist'

    # source of data
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_left', init_node=True)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                               robot_name=f'master_right', init_node=False)
    setup_master_bot(master_bot_left)
    setup_master_bot(master_bot_right)

    # setup the environment
    env = make_real_env(init_node=False)
    ts = env.reset(fake=True)
    episode = [ts]
    # setup visualization
    if onscreen_render:
        ax = plt.subplot()
        plt_img = ax.imshow(ts.observation['images'][render_cam])
        plt.ion()

    for t in range(1000):
        action = get_action(master_bot_left, master_bot_right)
        ts = env.step(action)
        episode.append(ts)

        if onscreen_render:
            plt_img.set_data(ts.observation['images'][render_cam])
            plt.pause(DT)
        else:
            time.sleep(DT)


if __name__ == '__main__':
    test_real_teleop()

