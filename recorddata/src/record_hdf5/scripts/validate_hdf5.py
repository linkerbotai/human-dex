import os,sys,time, rospy
import numpy as np
import cv2
import h5py
import argparse
from std_msgs.msg import Header, String
import matplotlib.pyplot as plt


import IPython
e = IPython.embed

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.color_msg import ColorMsg
from utils.constants import DT

class ValidateHdf5():
    def __init__(self) -> None:
        self.left_image = []
        rospy.Subscriber("/recurrent_ckpt", String, self.main)
    def load_hdf5(self,dataset_dir="/home/nx/ROS/robot_internal_srv/collection_data/hdf5/", dataset_name="episode_1"):
        dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
        if not os.path.isfile(dataset_path):
            print(f'Dataset does not exist at \n{dataset_path}\n')
            exit()

        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            compressed = root.attrs.get('compress', False)
            qpos = root['/observations/qpos'][()]
            qvel = root['/observations/qvel'][()]
            if 'effort' in root.keys():
                effort = root['/observations/effort'][()]
            else:
                effort = None
            action = root['/action'][()]
            #base_action = root['/base_action'][()]
            base_action = ""
            image_dict = dict()
            for cam_name in root[f'/observations/images/'].keys():
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]
            if compressed:
                compress_len = root['/compress_len'][()]

        if compressed:
            for cam_id, cam_name in enumerate(image_dict.keys()):
                # un-pad and uncompress
                padded_compressed_image_list = image_dict[cam_name]
                image_list = []
                for frame_id, padded_compressed_image in enumerate(padded_compressed_image_list): # [:1000] to save memory
                    image_len = int(compress_len[cam_id, frame_id])
                    compressed_image = padded_compressed_image
                    image = cv2.imdecode(compressed_image, 1)
                    image_list.append(image)
                image_dict[cam_name] = image_list


        return qpos, qvel, effort, action, base_action, image_dict

    def main(self, data):
        if data.data !=None or data.data !="":
            episode_idx = int(data.data)
        else:
            episode_idx = 0
        dataset_dir = "/home/nx/ROS/robot_internal_srv/collection_data/hdf5/humanplus_hdf5"
        #episode_idx = "1"
        ismirror = False
        if ismirror:
            dataset_name = f'mirror_episode_{episode_idx}'
        else:
            dataset_name = f'episode_{episode_idx}'

        qpos, qvel, effort, action, base_action, image_dict = self.load_hdf5(dataset_dir, dataset_name)
        print('hdf5 loaded!!')
        self.save_videos(image_dict, DT, video_path=os.path.join(dataset_dir, dataset_name + '_video.mp4'))
        #self.visualize_joints(qpos, action, plot_path=os.path.join(dataset_dir, dataset_name + '_qpos.png'))
        # visualize_single(effort, 'effort', plot_path=os.path.join(dataset_dir, dataset_name + '_effort.png'))
        # visualize_single(action - qpos, 'tracking_error', plot_path=os.path.join(dataset_dir, dataset_name + '_error.png'))
        #self.visualize_base(base_action, plot_path=os.path.join(dataset_dir, dataset_name + '_base_action.png'))
        # visualize_timestamp(t_list, dataset_path) # TODO addn timestamp back


    def save_videos(self,video, dt, video_path=None):
        if isinstance(video, list):
            cam_names = list(video[0].keys())
            h, w, _ = video[0][cam_names[0]].shape
            w = w * len(cam_names)
            fps = int(1/dt)
            out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            for ts, image_dict in enumerate(video):
                images = []
                for cam_name in cam_names:
                    image = image_dict[cam_name]
                    image = image[:, :, [2, 1, 0]] # swap B and R channel
                    images.append(image)
                images = np.concatenate(images, axis=1)
                out.write(images)
            out.release()
            print(f'Saved video to: {video_path}')
        elif isinstance(video, dict):
            cam_names = list(video.keys())
            all_cam_videos = []
            for cam_name in cam_names:
                all_cam_videos.append(video[cam_name])
            all_cam_videos = np.concatenate(all_cam_videos, axis=2) # width dimension

            n_frames, h, w, _ = all_cam_videos.shape
            fps = int(1 / dt)
            out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            for t in range(n_frames):
                image = all_cam_videos[t]
                image = image[:, :, [2, 1, 0]]  # swap B and R channel
                out.write(image)
            out.release()
            print(f'Saved video to: {video_path}')


    def visualize_joints(self,qpos_list, command_list, plot_path=None, ylim=None, label_overwrite=None):
        if label_overwrite:
            label1, label2 = label_overwrite
        else:
            label1, label2 = 'State', 'Command'

        qpos = np.array(qpos_list) # ts, dim
        command = np.array(command_list)
        num_ts, num_dim = qpos.shape
        h, w = 2, num_dim
        num_figs = num_dim
        fig, axs = plt.subplots(num_figs, 1, figsize=(8, 2 * num_dim))

        # plot joint state
        all_names = [name + '_left' for name in STATE_NAMES] + [name + '_right' for name in STATE_NAMES]
        for dim_idx in range(num_dim):
            ax = axs[dim_idx]
            ax.plot(qpos[:, dim_idx], label=label1)
            ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')
            ax.legend()

        # plot arm command
        for dim_idx in range(num_dim):
            ax = axs[dim_idx]
            ax.plot(command[:, dim_idx], label=label2)
            ax.legend()

        if ylim:
            for dim_idx in range(num_dim):
                ax = axs[dim_idx]
                ax.set_ylim(ylim)

        plt.tight_layout()
        plt.savefig(plot_path)
        print(f'Saved qpos plot to: {plot_path}')
        plt.close()

    def visualize_single(self,efforts_list, label, plot_path=None, ylim=None, label_overwrite=None):
        efforts = np.array(efforts_list) # ts, dim
        num_ts, num_dim = efforts.shape
        h, w = 2, num_dim
        num_figs = num_dim
        fig, axs = plt.subplots(num_figs, 1, figsize=(w, h * num_figs))

        # plot joint state
        all_names = [name + '_left' for name in STATE_NAMES] + [name + '_right' for name in STATE_NAMES]
        for dim_idx in range(num_dim):
            ax = axs[dim_idx]
            ax.plot(efforts[:, dim_idx], label=label)
            ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')
            ax.legend()

        if ylim:
            for dim_idx in range(num_dim):
                ax = axs[dim_idx]
                ax.set_ylim(ylim)

        plt.tight_layout()
        plt.savefig(plot_path)
        print(f'Saved effort plot to: {plot_path}')
        plt.close()

    def visualize_base(self,readings, plot_path=None):
        readings = np.array(readings) # ts, dim
        num_ts, num_dim = readings.shape
        num_figs = num_dim
        fig, axs = plt.subplots(num_figs, 1, figsize=(8, 2 * num_dim))

        # plot joint state
        all_names = BASE_STATE_NAMES
        for dim_idx in range(num_dim):
            ax = axs[dim_idx]
            ax.plot(readings[:, dim_idx], label='raw')
            ax.plot(np.convolve(readings[:, dim_idx], np.ones(20)/20, mode='same'), label='smoothed_20')
            ax.plot(np.convolve(readings[:, dim_idx], np.ones(10)/10, mode='same'), label='smoothed_10')
            ax.plot(np.convolve(readings[:, dim_idx], np.ones(5)/5, mode='same'), label='smoothed_5')
            ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')
            ax.legend()

        # if ylim:
        #     for dim_idx in range(num_dim):
        #         ax = axs[dim_idx]
        #         ax.set_ylim(ylim)

        plt.tight_layout()
        plt.savefig(plot_path)
        print(f'Saved effort plot to: {plot_path}')
        plt.close()


    def visualize_timestamp(self,t_list, dataset_path):
        plot_path = dataset_path.replace('.pkl', '_timestamp.png')
        h, w = 4, 10
        fig, axs = plt.subplots(2, 1, figsize=(w, h*2))
        # process t_list
        t_float = []
        for secs, nsecs in t_list:
            t_float.append(secs + nsecs * 10E-10)
        t_float = np.array(t_float)

        ax = axs[0]
        ax.plot(np.arange(len(t_float)), t_float)
        ax.set_title(f'Camera frame timestamps')
        ax.set_xlabel('timestep')
        ax.set_ylabel('time (sec)')

        ax = axs[1]
        ax.plot(np.arange(len(t_float)-1), t_float[:-1] - t_float[1:])
        ax.set_title(f'dt')
        ax.set_xlabel('timestep')
        ax.set_ylabel('time (sec)')

        plt.tight_layout()
        plt.savefig(plot_path)
        print(f'Saved timestamp plot to: {plot_path}')
        plt.close()

if __name__ == "__main__":
    #2.初始化 ROS 节点:命名(唯一)
    rospy.init_node("recurrent_ckpt", anonymous=True)
    r = ValidateHdf5()
    #r.main()
    #5.设置循环调用回调函数
    rospy.spin()