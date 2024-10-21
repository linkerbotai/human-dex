'''
Date: 2024-09-26 14:02:17
LastEditors: HJX
LastEditTime: 2024-09-26 18:41:58
FilePath: /humanplus_ros/src/humanplus_ros/scripts/utils/HIT/constants.py
'''
import os

### Task parameters
DATA_DIR = f'{os.path.dirname(os.path.realpath(__file__))}/data' 
LEFT_HAND_JOINT = 9 # 左手关节数
RIGHT_HAND_JOINT = 9 # 右手关节数
LEFT_ARM_JOINT = 6 # 左臂关节数
RIGHT_ARM_JOINT = 6 # 右臂关节数
TASK_CONFIGS = {
    'data_cb_grasp':{
        'dataset_dir': '/home/nx/ROS/robot_internal_srv/collection_data/hdf5',
        'camera_names': ['cam_left', 'cam_right'],
        'observation_name': ['qpos','hand_action'],
        'num_episodes': 100,
        'episode_len': 1000,
        'joint_total': LEFT_ARM_JOINT+RIGHT_ARM_JOINT+LEFT_HAND_JOINT+RIGHT_HAND_JOINT,
        'state_dim':[1]*LEFT_ARM_JOINT+[1]*RIGHT_ARM_JOINT+[1]*LEFT_HAND_JOINT+[1]*RIGHT_HAND_JOINT,
        'action_dim':[1]*LEFT_ARM_JOINT+[1]*RIGHT_ARM_JOINT+[1]*LEFT_HAND_JOINT+[1]*RIGHT_HAND_JOINT,
        'state_mask': [1]*LEFT_ARM_JOINT+[1]*RIGHT_ARM_JOINT+[1]*LEFT_HAND_JOINT+[1]*RIGHT_HAND_JOINT,
        'action_mask': [1]*LEFT_ARM_JOINT+[1]*RIGHT_ARM_JOINT+[1]*LEFT_HAND_JOINT+[1]*RIGHT_HAND_JOINT,
    },
    # 'data_cb_grasp':{
    #     'dataset_dir': '/home/nx/ROS/robot_internal_srv/collection_data/hdf5',
    #     'camera_names': ['cam_left', 'cam_right'],
    #     'observation_name': ['qpos','hand_action'],
    #     'num_episodes': 100,
    #     'episode_len': 400,
    #     'state_dim':30,
    #     'action_dim':30,
    #     'state_mask': [1]*30,
    #     'action_mask': [1]*30 #arms(11-16right,21-26left)+left_hand+right_hand 
    # },
    'data_fold_clothes':{
        'dataset_dir': DATA_DIR + '/data_fold_clothes',
        'camera_names': ['cam_left', 'cam_right'],
        'observation_name': ['qpos','hand_action','wrist'],
        'state_dim':35,
        'action_dim':40,
        'state_mask': [0]*11 + [1]*24,
        'action_mask': [0]*11 + [1]*8 + [0]*5 + [1]*16 #11 for leg, 8 for arm, 5 for imu, 16 for gripper 
    },
    
    'data_rearrange_objects':{
        'dataset_dir': DATA_DIR + '/data_rearrange_objects',
        'camera_names': ['cam_left', 'cam_right'],
        'observation_name': ['qpos','hand_action'], # imu_orn -> only 0,1
        'state_dim':33,
        'action_dim':40,
        'state_mask': [0]*11 + [1]*22,
        'action_mask': [0]*10 + [0] + [1]*8 + [0]*5 + [1]*16, #10 for leg, 1 for waist, 8 for arm, 5 for imu, 16 for gripper 
    },
    
    'data_two_robot_greeting':{
        'dataset_dir': DATA_DIR + '/data_two_robot_greeting',
        'num_episodes': 100,
        'episode_len': 1000,
        'camera_names': ['cam_left', 'cam_right'],
        'observation_name': ['qpos','hand_action'], # imu_orn -> only 0,1
        'state_dim':33,
        'action_dim':40,
        'state_mask': [0]*10 + [1]+ [1]*22,
        'action_mask': [0]*10 + [1] + [1]*8 + [0]*5 + [1]*16, #10 for leg, 1 for waist, 8 for arm, 5 for imu, 16 for gripper 
    },
    
    'data_warehouse':{
        'dataset_dir': DATA_DIR + '/data_warehouse',
        'camera_names': ['cam_left', 'cam_right'],
        'observation_name': ['qpos','hand_action'], # imu_orn -> only 0,1
        'state_dim':33,
        'action_dim':40,
        'state_mask': [1]*11 + [1]*22,
        'action_mask': [1]*10 + [1] + [1]*8 + [0]*5 + [1]*16, #10 for leg, 1 for waist, 8 for arm, 5 for imu, 16 for gripper 
    },
}

### Simulation envs fixed constants
DT = 0.04
FPS = 25