#! /usr/bin/env python3
import sys
print(sys.executable)
import rospy, tf, h5py, ast, rospkg, cv2,sys,os,json,time, re,json
from PIL import Image as PILImage  # 为 Pillow 的 Image 模块重命名
import numpy as np
from enum import Enum
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header, String, Float32MultiArray, Float32
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import JointState, Image, CameraInfo
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.color_msg import ColorMsg
from utils.real_env import RealEnv
from utils.constants import DT, TASK_CONFIGS, FPS,LEFT_HAND_JOINT,RIGHT_HAND_JOINT,LEFT_ARM_JOINT,RIGHT_ARM_JOINT
# rostopic pub /record_hdf5 std_msgs/String "data: '{\"method\":\"start\",\"type\":\"humanplus\"}'"
# rostopic pub /record_hdf5 std_msgs/String "data: '{\"method\":\"start\",\"type\":\"aloha\"}'"

class RecordHdf5:
    def __init__(self):
        self.task_pub = rospy.Publisher("/app_control_cmd",String,queue_size=10)
        self.head_pub = rospy.Publisher("/head_hand_cmd", String, queue_size=1)
        self.humanplus_data_dic = {
            '/observations/hand_action': [],
            '/observations/imu_orn': [],
            '/observations/imu_vel': [],
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/observations/wrist': [],
            '/action': []
        }
        self.aloha_data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/observations/effort': [],
            '/action': [],
            '/base_action': [],
            # '/base_action_t265': [],
        }
        # 创建CvBridge对象
        self.bridge = CvBridge()


        # self.task_config = TASK_CONFIGS["data_columbus_test"]
        # # self.task_config = TASK_CONFIGS[task_type]
        # self.joint_total = self.task_config["joint_total"]
        # self.dataset_dir = self.task_config['dataset_dir']
        # self.max_timesteps = self.task_config['episode_len']
        # self.camera_names = self.task_config['camera_names']
        # self.episode_idx = self.get_max_episode_number() + 1
        ColorMsg(msg="hdf5采集功能准备就绪", color="green")
        sub = rospy.Subscriber("/record_hdf5", String, self.record_hdf5, queue_size=1)
    def record_hdf5(self,data):
        result = json.loads(data.data)
        task_type = result["task"]
        self.task_config = TASK_CONFIGS[task_type]
        self.joint_total = self.task_config["joint_total"]
        self.dataset_dir = self.task_config['dataset_dir']
        self.max_timesteps = self.task_config['episode_len']
        self.camera_names = self.task_config['camera_names']
        self.episode_idx = self.get_max_episode_number() + 1
        self.env = RealEnv(self.camera_names)
        # print

        if result["method"] == "stop":
            self.clean_data()
        if result["method"] == "start":
            self.clean_data()
            s = String()
            head_msg = {
                "method":"head_motion",
                "id":121212,
                "params":{
                    "action": "down"
                },
            }
            s.data = json.dumps(head_msg)
            self.head_pub.publish(s)
            time.sleep(0.01)
            
            if result["type"] == "humanplus":
                self.dataset_dir = self.task_config['dataset_dir']+"/humanplus_hdf5/"
                self.episode_idx = self.get_max_episode_number() + 1
                self.env = RealEnv(self.camera_names)
                self.record(d="humanplus")
            elif result["type"] == "aloha":
                self.dataset_dir = self.task_config['dataset_dir']+"/aloha_hdf5/"
                self.episode_idx = self.get_max_episode_number() + 1
                self.env = RealEnv(self.camera_names)
                self.record(d="aloha")
    def clean_data(self):
        self.humanplus_data_dic = {
            '/observations/hand_action': [],
            '/observations/imu_orn': [],
            '/observations/imu_vel': [],
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/observations/wrist': [],
            '/action': []
        }
        self.aloha_data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/observations/effort': [],
            '/action': [],
            '/base_action': [],
            # '/base_action_t265': [],
        }
        self.env.clean_data()
    def get_max_episode_number(self):
        directory = self.dataset_dir
        # 定义匹配 episode 文件的正则表达式，匹配形如 'episode_123.hdf5' 的文件
        pattern = re.compile(r'episode_(\d+)\.hdf5')
        max_number = -1  # 初始化最大值为 -1，表示没有找到任何符合条件的文件
        # 遍历目录中的所有文件
        if not os.path.exists(directory):
            os.makedirs(directory)
        for filename in os.listdir(directory):
            match = pattern.match(filename)
            if match:
                # 提取并转换匹配到的数字部分为整数
                number = int(match.group(1))
                # 更新最大值
                max_number = max(max_number, number)
        return max_number if max_number != -1 else 0  # 如果没有找到匹配的文件，则返回 None

    def cam_left_cb(self, data):
        self.cam_left = data
    def cam_right_cb(self, data):
        self.cam_right = data
    def joint_states_cb(self, data):
        self.joint_states = data
    def arms_cb(self, data):
        self.arms = data
    def left_hand_cb(self, data):
        self.left_hand = data
    def right_hand_cb(self, data):
        self.right_hand = data

    def record(self, d="aloha"):
        if d == "aloha":
            r_s = String()
            r_s.data = '{\"method\":\"start_speak\", \"id\":121212, \"params\":\"开始采集aloha\"}'
            self.task_pub.publish(r_s)
            ColorMsg(msg="开始采集aloha模式hdf5文件", color="green")
            ts = self.env.reset(fake=True,t="aloha")
        else:
            r_s = String()
            r_s.data = '{\"method\":\"start_speak\", \"id\":121212, \"params\":\"开始采集human plus\"}'
            self.task_pub.publish(r_s)
            ColorMsg(msg="开始采集human_plus模式hdf5文件", color="green")
            ts = self.env.reset(fake=True,t="humanplus")
        
        timesteps = [ts]
        actions = []
        actual_dt_history = []
        time0 = time.time()
        DT = 1 / FPS
        for t in tqdm(range(self.max_timesteps)):
            t0 = time.time() #
            action = self.env.get_action()
            t1 = time.time() #
            if d == "aloha":
                ts = self.env.step(action,t="aloha")
            else:
                ts = self.env.step(action,t="humanplus")
           
            t2 = time.time() #
            timesteps.append(ts)
            actions.append(action)
            actual_dt_history.append([t0, t1, t2])
            time.sleep(max(0, DT - (time.time() - t0)))
        print(f'Avg fps: {10 / (time.time() - time0)}')
        if actions[0][0] != None:
            #self.to_aloha_hdf5(ts, timesteps, actions)
            if d=="aloha":
                r,f_name = self.to_aloha_hdf5(ts, timesteps=timesteps, actions=actions)
            else:
                r,f_name = self.to_humanplus_hdf5(ts, timesteps=timesteps, actions=actions)
            if r == True:
                s = String()
                s.data = '{\"method\":\"start_speak\", \"id\":121212, \"params\":\"采集完毕,'+f_name+'\"}'
                self.task_pub.publish(s)

    def to_humanplus_hdf5(self,ts, timesteps, actions):
        dataset_path = self.dataset_dir+f"episode_{self.episode_idx}"
        # if not os.path.exists(dataset_path):
        #     os.makedirs(dataset_path)
        
        for cam_name in self.camera_names:
            self.humanplus_data_dic[f'/observations/images/{cam_name}'] = []

        # len(action): max_timesteps, len(time_steps): max_timesteps + 1
        while actions:
            
            action = actions.pop(0)
            ts = timesteps.pop(0)
            arm_j = int(LEFT_ARM_JOINT+RIGHT_ARM_JOINT)
            self.humanplus_data_dic["/action"].append(action)
            self.humanplus_data_dic["/observations/hand_action"].append(action[6:]) # 总action切除手臂元素=双手action
            self.humanplus_data_dic["/observations/imu_orn"].append(np.array([0]*self.joint_total, dtype=np.float32))
            self.humanplus_data_dic["/observations/imu_vel"].append(np.array([0]*self.joint_total, dtype=np.float32))
            self.humanplus_data_dic["/observations/qpos"].append(ts.observation['qpos'])
            self.humanplus_data_dic["/observations/qvel"].append(ts.observation['qvel'])
            self.humanplus_data_dic["/observations/wrist"].append(np.array([0.0]*2, dtype=np.float32))

            # data_dict['/base_action_t265'].append(ts.observation['base_vel_t265'])
            # print(ts.observation['images'])
            for cam_name in self.camera_names:
                self.humanplus_data_dic[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])
        COMPRESS = True
        if COMPRESS:
            # JPEG compression
            t0 = time.time()
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50] # tried as low as 20, seems fine
            compressed_len = []
            for cam_name in self.camera_names:
                image_list = self.humanplus_data_dic[f'/observations/images/{cam_name}']
                compressed_list = []
                compressed_len.append([])
                for image in image_list:
                    result, encoded_image = cv2.imencode('.jpg', image, encode_param) # 0.02 sec # cv2.imdecode(encoded_image, 1)
                    compressed_list.append(encoded_image)
                    compressed_len[-1].append(len(encoded_image))
                self.humanplus_data_dic[f'/observations/images/{cam_name}'] = compressed_list
            print(f'compression: {time.time() - t0:.2f}s')

            # pad so it has same length
            t0 = time.time()
            compressed_len = np.array(compressed_len)
            padded_size = compressed_len.max()
            for cam_name in self.camera_names:
                compressed_image_list = self.humanplus_data_dic[f'/observations/images/{cam_name}']
                padded_compressed_image_list = []
                for compressed_image in compressed_image_list:
                    padded_compressed_image = np.zeros(padded_size, dtype='uint8')
                    image_len = len(compressed_image)
                    padded_compressed_image[:image_len] = compressed_image[0]
                    padded_compressed_image_list.append(padded_compressed_image)
                self.humanplus_data_dic[f'/observations/images/{cam_name}'] = padded_compressed_image_list
            print(f'padding: {time.time() - t0:.2f}s')
        # HDF5
        t0 = time.time()
        
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024**2*2) as root:
            root.attrs['sim'] = False
            root.attrs['compress'] = COMPRESS
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in self.camera_names:
                if COMPRESS:
                    _ = image.create_dataset(cam_name, (self.max_timesteps, padded_size), dtype='uint8',
                                            chunks=(1, padded_size), )
                else:
                    _ = image.create_dataset(cam_name, (self.max_timesteps, 480, 640, 3), dtype='uint8',
                                            chunks=(1, 480, 640, 3), )
            _ = root.create_dataset('action', (self.max_timesteps, self.joint_total))
            _ = obs.create_dataset('hand_action', (self.max_timesteps, RIGHT_HAND_JOINT))
            _ = obs.create_dataset('imu_orn', (self.max_timesteps, self.joint_total))
            _ = obs.create_dataset('imu_vel', (self.max_timesteps, self.joint_total))
            _ = obs.create_dataset('qpos', (self.max_timesteps, 12))
            _ = obs.create_dataset('qvel', (self.max_timesteps, 12))
            _ = obs.create_dataset('wrist', (self.max_timesteps, 2))
            # _ = root.create_dataset('base_action_t265', (max_timesteps, 2))

            for name, array in self.humanplus_data_dic.items():
                print(name)
                # print(array.shape)
                root[name][...] = array
            if COMPRESS:
                _ = root.create_dataset('compress_len', (len(self.camera_names), self.max_timesteps))
                root['/compress_len'][...] = compressed_len

        print(f'Saving: {time.time() - t0:.1f} secs')
        ColorMsg(msg=f"{dataset_path}.hdf5文件保存成功", color="green")
        print("\n")
        return True, f"episode {self.episode_idx}"

    def to_aloha_hdf5(self, ts, timesteps, actions):
        dataset_path = self.dataset_dir+f"episode_{self.episode_idx}"
        # if not os.path.exists(directory):
        #     os.makedirs(directory)
        
        for cam_name in self.camera_names:
            self.aloha_data_dict[f'/observations/images/{cam_name}'] = []

        # len(action): max_timesteps, len(time_steps): max_timesteps + 1
        while actions:
            action = actions.pop(0)
            ts = timesteps.pop(0)
            self.aloha_data_dict['/observations/qpos'].append(ts.observation['qpos'])
            self.aloha_data_dict['/observations/qvel'].append(ts.observation['qvel'])
            self.aloha_data_dict['/observations/effort'].append(ts.observation['effort'])
            self.aloha_data_dict['/action'].append(action)
            self.aloha_data_dict['/base_action'].append(ts.observation['base_vel'])
            # data_dict['/base_action_t265'].append(ts.observation['base_vel_t265'])
            for cam_name in self.camera_names:
                self.aloha_data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])
        COMPRESS = True

        if COMPRESS:
            # JPEG compression
            t0 = time.time()
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50] # tried as low as 20, seems fine
            compressed_len = []
            for cam_name in self.camera_names:
                image_list = self.aloha_data_dict[f'/observations/images/{cam_name}']
                compressed_list = []
                compressed_len.append([])
                for image in image_list:
                    result, encoded_image = cv2.imencode('.jpg', image, encode_param) # 0.02 sec # cv2.imdecode(encoded_image, 1)
                    compressed_list.append(encoded_image)
                    compressed_len[-1].append(len(encoded_image))
                self.aloha_data_dict[f'/observations/images/{cam_name}'] = compressed_list
            print(f'compression: {time.time() - t0:.2f}s')

            # pad so it has same length
            t0 = time.time()
            compressed_len = np.array(compressed_len)
            padded_size = compressed_len.max()
            for cam_name in self.camera_names:
                compressed_image_list = self.aloha_data_dict[f'/observations/images/{cam_name}']
                padded_compressed_image_list = []
                for compressed_image in compressed_image_list:
                    padded_compressed_image = np.zeros(padded_size, dtype='uint8')
                    image_len = len(compressed_image)
                    padded_compressed_image[:image_len] = compressed_image
                    padded_compressed_image_list.append(padded_compressed_image)
                self.aloha_data_dict[f'/observations/images/{cam_name}'] = padded_compressed_image_list
            print(f'padding: {time.time() - t0:.2f}s')

        # HDF5
        t0 = time.time()
        
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024**2*2) as root:
            root.attrs['sim'] = False
            root.attrs['compress'] = COMPRESS
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in self.camera_names:
                if COMPRESS:
                    _ = image.create_dataset(cam_name, (self.max_timesteps, padded_size), dtype='uint8',
                                            chunks=(1, padded_size), )
                else:
                    _ = image.create_dataset(cam_name, (self.max_timesteps, 480, 640, 3), dtype='uint8',
                                            chunks=(1, 480, 640, 3), )
            _ = obs.create_dataset('qpos', (self.max_timesteps, self.joint_total))
            _ = obs.create_dataset('qvel', (self.max_timesteps, self.joint_total))
            _ = obs.create_dataset('effort', (self.max_timesteps, self.joint_total))
            _ = root.create_dataset('action', (self.max_timesteps, self.joint_total))
            _ = root.create_dataset('base_action', (self.max_timesteps, self.joint_total))
            # _ = root.create_dataset('base_action_t265', (max_timesteps, 2))

            for name, array in self.aloha_data_dict.items():
                root[name][...] = array

            if COMPRESS:
                _ = root.create_dataset('compress_len', (len(self.camera_names), self.max_timesteps))
                root['/compress_len'][...] = compressed_len

        print(f'Saving: {time.time() - t0:.1f} secs')
        ColorMsg(msg=f"{dataset_path}.hdf5文件保存成功", color="green")
        print("\n")
        return True, f"episode {self.episode_idx}"

    # 将图像数据压缩为数据流
    def ros_image_to_compressed_byte_stream(self,ros_image_msg, format='.jpg', quality=90):
        # 将ROS的Image消息转换为OpenCV的图像（numpy数组）
        cv_image = self.bridge.imgmsg_to_cv2(ros_image_msg, desired_encoding='bgr8')
        # 设置压缩参数，quality参数控制JPEG压缩质量（取值范围为0-100）
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality] if format == '.jpg' else []
        # 将图像压缩为字节流
        success, compressed_image = cv2.imencode(format, cv_image, encode_param)
        if success:
            # 将压缩后的图像字节流返回
            return np.array(compressed_image)  # 返回字节流
        else:
            # raise RuntimeError("图像压缩失败")
            return False

if __name__ == "__main__":
    rospy.init_node("record_hdf5",anonymous=True)
    h = RecordHdf5()
    
    rospy.spin()