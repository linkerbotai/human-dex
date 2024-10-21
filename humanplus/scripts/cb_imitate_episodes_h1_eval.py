import torch, time, rospy
import numpy as np
import os,sys
import pickle
import argparse
from copy import deepcopy
from itertools import repeat
from tqdm import tqdm
from cv_bridge import CvBridge
from einops import rearrange
import json
import wandb
from sensor_msgs.msg import Image,JointState
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.HIT.utils import compute_dict_mean, set_seed, load_data # data functions
from utils.color_msg import ColorMsg
from utils.real_env import make_real_env
from utils.HIT.constants import TASK_CONFIGS
from utils.HIT.constants import DT,FPS
from utils.HIT.model_util import make_policy, make_optimizer

def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    return policy(qpos_data, image_data, action_data, is_pad) # TODO remove None

def train_bc(train_dataloader, val_dataloader, config):
    num_steps = config['num_steps']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    validate_every = config['validate_every']
    save_every = config['save_every']

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    if config['load_pretrain']:
        loading_status = policy.deserialize(torch.load(f'{config["pretrained_path"]}/policy_last.ckpt', map_location='cuda'))
        print(f'loaded! {loading_status}')
    if config['resume_ckpt_path'] is not None:
        loading_status = policy.deserialize(torch.load(config['resume_ckpt_path']))
        print(f'Resume policy from: {config["resume_ckpt_path"]}, Status: {loading_status}')
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)
    if config['load_pretrain']:
        optimizer.load_state_dict(torch.load(f'{config["pretrained_path"]}/optimizer_last.ckpt', map_location='cuda'))
        

    min_val_loss = np.inf
    best_ckpt_info = None
    
    train_dataloader = repeater(train_dataloader)
    for step in tqdm(range(num_steps+1)):
        if step % validate_every == 0:
            print('validating')

            with torch.inference_mode():
                policy.eval()
                validation_dicts = []
                for batch_idx, data in enumerate(val_dataloader):
                    forward_dict = forward_pass(data, policy)
                    validation_dicts.append(forward_dict)
                    if batch_idx > 50:
                        break

                validation_summary = compute_dict_mean(validation_dicts)

                epoch_val_loss = validation_summary['loss']
                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    best_ckpt_info = (step, min_val_loss, deepcopy(policy.serialize()))
            for k in list(validation_summary.keys()):
                validation_summary[f'val_{k}'] = validation_summary.pop(k)     
            if config['wandb']:       
                wandb.log(validation_summary, step=step)
            print(f'Val loss:   {epoch_val_loss:.5f}')
            summary_string = ''
            for k, v in validation_summary.items():
                summary_string += f'{k}: {v.item():.3f} '
            print(summary_string)
                
        # training
        policy.train()
        optimizer.zero_grad()
        data = next(train_dataloader)
        forward_dict = forward_pass(data, policy)
        # backward
        loss = forward_dict['loss']
        loss.backward()
        optimizer.step()
        if config['wandb']:
            wandb.log(forward_dict, step=step) # not great, make training 1-2% slower

        if step % save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_step_{step}_seed_{seed}.ckpt')
            torch.save(policy.serialize(), ckpt_path)
            #save optimizer state
            optimizer_ckpt_path = os.path.join(ckpt_dir, f'optimizer_step_{step}_seed_{seed}.ckpt')
            torch.save(optimizer.state_dict(), optimizer_ckpt_path)
        if step % 2000 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
            torch.save(policy.serialize(), ckpt_path)
            optimizer_ckpt_path = os.path.join(ckpt_dir, f'optimizer_last.ckpt')
            torch.save(optimizer.state_dict(), optimizer_ckpt_path)
            
    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.serialize(), ckpt_path)
    optimizer_ckpt_path = os.path.join(ckpt_dir, f'optimizer_last.ckpt')
    torch.save(optimizer.state_dict(), optimizer_ckpt_path)

    return best_ckpt_info

def repeater(data_loader):
    epoch = 0
    for loader in repeat(data_loader):
        for data in loader:
            yield data
        print(f'Epoch {epoch} done')
        epoch += 1

def main_train(args):
    set_seed(1)
    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_steps = args['num_steps']
    eval_every = args['eval_every']
    validate_every = args['validate_every']
    save_every = args['save_every']
    resume_ckpt_path = args['resume_ckpt_path']
    backbone = args['backbone']
    same_backbones = args['same_backbones']
    
    args['ckpt_dir'] = ckpt_dir 
    # get task parameters
    is_sim = task_name[:4] == 'sim_'

    task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config['dataset_dir']
    camera_names = task_config['camera_names']
    stats_dir = task_config.get('stats_dir', None)
    sample_weights = task_config.get('sample_weights', None)
    train_ratio = task_config.get('train_ratio', 0.99)
    name_filter = task_config.get('name_filter', lambda n: True)
    randomize_index = task_config.get('randomize_index', False)

    print(f'===========================START===========================:')
    print(f"{task_name}")
    print(f'===========================Config===========================:')
    print(f'ckpt_dir: {ckpt_dir}')
    print(f'policy_class: {policy_class}')
    # fixed parameters
    state_dim = task_config.get('state_dim', 24)
    action_dim = task_config.get('action_dim', 24)
    state_mask = task_config.get('state_mask', np.ones(state_dim))
    action_mask = task_config.get('action_mask', np.ones(action_dim))
    if args['use_mask']:
        state_dim = sum(state_mask)
        action_dim = sum(action_mask)
        state_idx = np.where(state_mask)[0].tolist()
        action_idx = np.where(action_mask)[0].tolist()
    else:
        state_idx = np.arange(state_dim).tolist()
        action_idx = np.arange(action_dim).tolist()
    lr_backbone = 1e-5
    backbone = args['backbone']
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'same_backbones': args['same_backbones'],
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         'vq': False,
                         'action_dim': action_dim,
                         'state_dim': state_dim,
                         'no_encoder': args['no_encoder'],
                         'state_idx': state_idx,
                         'action_idx': action_idx,
                         'state_mask': state_mask,
                         'action_mask': action_mask,
                         }
    elif policy_class == 'HIT':
        policy_config = {'lr': args['lr'],
                         'hidden_dim': args['hidden_dim'],
                         'dec_layers': args['dec_layers'],
                         'nheads': args['nheads'],
                         'num_queries': args['chunk_size'],
                         'camera_names': camera_names,
                         'action_dim': action_dim,
                         'state_dim': state_dim,
                         'backbone': backbone,
                         'same_backbones': args['same_backbones'],
                         'lr_backbone': lr_backbone,
                         'context_len': 183, #for 224,400
                         'num_queries': args['chunk_size'], 
                         'use_pos_embd_image': args['use_pos_embd_image'],
                         'use_pos_embd_action': args['use_pos_embd_action'],
                         'feature_loss': args['feature_loss_weight']>0,
                         'feature_loss_weight': args['feature_loss_weight'],
                         'self_attention': args['self_attention']==1,
                         'state_idx': state_idx,
                         'action_idx': action_idx,
                         'state_mask': state_mask,
                         'action_mask': action_mask,
                         }
    else:
        raise NotImplementedError
    print(f'====================FINISH INIT POLICY========================:')
    config = {
        'num_steps': num_steps,
        'eval_every': eval_every,
        'validate_every': validate_every,
        'save_every': save_every,
        'ckpt_dir': ckpt_dir,
        'resume_ckpt_path': resume_ckpt_path,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': False,
        'camera_names': camera_names,
        'real_robot': not is_sim,
        'load_pretrain': args['load_pretrain'],
        'height':args['height'],
        'width':args['width'],
        'normalize_resnet': args['normalize_resnet'],
        'wandb': args['wandb'],
        'pretrained_path': args['pretrained_path'],
        'randomize_data_degree': args['randomize_data_degree'],
        'randomize_data': args['randomize_data'],
    }
    
    ckpt_names = [f'policy_last.ckpt']
    results = []
    for ckpt_name in ckpt_names:
        success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True, num_rollouts=10)
        # wandb.log({'success_rate': success_rate, 'avg_return': avg_return})
        results.append([ckpt_name, success_rate, avg_return])

def get_image(camera_names):
    
    curr_images = []
    for cam_name in camera_names:
        if cam_name == "cam_left":
            img_msg = rospy.wait_for_message("/camera/rgb/image_raw/cam_left",Image, timeout=0.1)
        elif cam_name == "cam_right":
            img_msg = rospy.wait_for_message("/camera/rgb/image_raw/cam_right",Image, timeout=0.1)
        #curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        # 使用cv_bridge将ROS图像消息转换为OpenCV格式
        bridge = CvBridge()
        img = bridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')
        curr_image = rearrange(img, 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image

def eval_bc(config, ckpt_name, save_episode=True, num_rollouts=50):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    ckpt_dir = "/home/hjx/ROS/humanplus_ros/src/humanplus_ros/scripts/utils/hardware-script/ckpt_bak/"
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = 400 #config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'angle'
    vq = False #config['policy_config']['vq']
    # actuator_config = config['actuator_config']
    # use_actuator_net = actuator_config['actuator_network_dir'] is not None

    # load policy and stats
    # /home/hjx/humanplus/hardware-script/ckpt/policy_last.ckpt
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.deserialize(torch.load(ckpt_path))
    policy.cuda()
    policy.eval()
    if vq:
        vq_dim = config['policy_config']['vq_dim']
        vq_class = config['policy_config']['vq_class']
        latent_model = Latent_Model_Transformer(vq_dim, vq_dim, vq_class)
        latent_model_ckpt_path = os.path.join(ckpt_dir, 'latent_model_last.ckpt')
        latent_model.deserialize(torch.load(latent_model_ckpt_path))
        latent_model.eval()
        latent_model.cuda()
        print(f'Loaded policy from: {ckpt_path}, latent model from: {latent_model_ckpt_path}')
    else:
        print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
   
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    if policy_class == 'Diffusion':
        post_process = lambda a: ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']
    else:
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # load environment
    # if real_robot:
    #     env = make_real_env(init_node=True, setup_robots=True, setup_base=True)
    #     env_max_reward = 0
    env = make_real_env(init_node=True, setup_robots=True, setup_base=True)
    query_frequency = policy_config['num_queries']
    # if temporal_agg:
    #     query_frequency = 1
    #     num_queries = policy_config['num_queries']
    # if real_robot:
    #     BASE_DELAY = 13
    #     query_frequency -= BASE_DELAY

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):
        # if real_robot:
        #     e()
        # rollout_id += 0
        # ### set task
        # if 'sim_transfer_cube' in task_name:
        #     BOX_POSE[0] = sample_box_pose() # used in sim reset
        # elif 'sim_insertion' in task_name:
        #     BOX_POSE[0] = np.concatenate(sample_insertion_pose()) # used in sim reset

        # ts = env.reset()

        # ### onscreen render
        # if onscreen_render:
        #     ax = plt.subplot()
        #     plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
        #     plt.ion()

        # ### evaluation loop
        # if temporal_agg:
        #     all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, 16]).cuda()

        # qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        qpos_history_raw = np.zeros((max_timesteps, state_dim))
        image_list = [] # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        culmulated_delay=0
        time0 = time.time()
        DT = 1 / FPS
        # if use_actuator_net:
        #     norm_episode_all_base_actions = [actuator_norm(np.zeros(history_len, 2)).tolist()]
        with torch.inference_mode():
            
            
            # culmulated_delay = 0 
            for t in range(max_timesteps):
                t0 = time.time()
                ### update onscreen render and wait for DT
                ### process previous timestep to get qpos and image_list
                # time2 = time.time()
                # obs = ts.observation
                # if 'images' in obs:
                #     image_list.append(obs['images'])
                # else:
                #     image_list.append({'main': obs['image']})
                arm_sub = rospy.wait_for_message("/joint_states",JointState, timeout=0.1)
                left_hand_sub = rospy.wait_for_message("/cb_left_hand_state",JointState, timeout=0.1)
                right_hand_sub = rospy.wait_for_message("/cb_right_hand_state",JointState, timeout=0.1)
                arm_position = list(arm_sub.position)
                arm_position.pop(6)
                arm_position.pop(12)
                arm_position.pop(12)
                arm_position.pop(12)
                arm_position.pop(12)
                left_hand_list = list(left_hand_sub.position)
                left_position = left_hand_list[:6]
                
                right_hand_list = list(right_hand_sub.position)
                right_position = right_hand_list[:6]
                qpos_numpy = np.array(arm_position+left_position+right_position)
                #qpos_history_raw[t] = qpos_numpy

                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                
                # qpos_history[:, t] = qpos
                if t % query_frequency == 0:
                    #curr_image = 0 
                    curr_image = get_image(camera_names=camera_names)
                # print('get image: ', time.time() - time2)
                #qpos = [0.0]*7+arm_list+[0.0]*5+left+right+[0.0]*4
                if t == 0:
                    # warm up
                    for _ in range(10):
                        policy(qpos, curr_image)
                    print('network warm up done,put current pos and image 10 times--Ewen')
                    # time1 = time.time()

                ### query policy
                time3 = time.time()
                if config['policy_class'] == "HIT":
                    if t % query_frequency == 0:
                        if vq:
                            if rollout_id == 0:
                                for _ in range(10):
                                    vq_sample = latent_model.generate(1, temperature=1, x=None)
                            vq_sample = latent_model.generate(1, temperature=1, x=None)
                            all_actions = policy(qpos, curr_image, vq_sample=vq_sample)
                        else:
                            # 进入到这里
                            all_actions = policy(qpos, curr_image)
                    
                    raw_action = all_actions[:, t % query_frequency]
                ### post-process actions
                #raw_action = raw_action.squeeze(0).cpu().numpy()
                raw_action = all_actions.squeeze(0).cpu().numpy()
                
                action = post_process(raw_action)
                target_qpos = action[:-2]
                # if use_actuator_net:
                #     assert(not temporal_agg)
                #     if t % prediction_len == 0:
                #         offset_start_ts = t + history_len
                #         actuator_net_in = np.array(norm_episode_all_base_actions[offset_start_ts - history_len: offset_start_ts + future_len])
                #         actuator_net_in = torch.from_numpy(actuator_net_in).float().unsqueeze(dim=0).cuda()
                #         pred = actuator_network(actuator_net_in)
                #         base_action_chunk = actuator_unnorm(pred.detach().cpu().numpy()[0])
                #     base_action = base_action_chunk[t % prediction_len]
                # else:
                base_action = action[-2:]
                # base_action = calibrate_linear_vel(base_action, c=0.19)
                # base_action = postprocess_base_action(base_action)
                # print('post process: ', time.time() - time4)

                ### step the environment 
                # time5 = time.time()
                # if real_robot:
                #     ts = env.step(target_qpos, base_action) # 这里不需要设置电机
                # else:
                #     ts = env.step(target_qpos)
                
                ts = env.step(action[33])
                # print('step env: ', time.time() - time5)

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)
                duration = time.time() - t0
                sleep_time = max(0, DT - duration)
                # print(sleep_time)
                
                # time.sleep(max(0, DT - duration - culmulated_delay))
                if duration >= DT:
                    culmulated_delay += (duration - DT)
                    print(f'Warning: step duration: {duration:.3f} s at step {t} longer than DT: {DT} s, culmulated delay: {culmulated_delay:.3f} s')
                # else:
                #     culmulated_delay = max(0, culmulated_delay - (DT - duration))
                print("_-"*30)
                print(sleep_time)
                ColorMsg(msg=f"第{t}次循环完毕", color="yellow")
                time.sleep(sleep_time)
            print(f'Avg fps {max_timesteps / (time.time() - t0)}')
            plt.close()
        if real_robot:
            move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open
            # save qpos_history_raw
            log_id = get_auto_index(ckpt_dir)
            np.save(os.path.join(ckpt_dir, f'qpos_{log_id}.npy'), qpos_history_raw)
            plt.figure(figsize=(10, 20))
            # plot qpos_history_raw for each qpos dim using subplots
            for i in range(state_dim):
                plt.subplot(state_dim, 1, i+1)
                plt.plot(qpos_history_raw[:, i])
                # remove x axis
                if i != state_dim - 1:
                    plt.xticks([])
            plt.tight_layout()
            plt.savefig(os.path.join(ckpt_dir, f'qpos_{log_id}.png'))
            plt.close()
            '''
            print("0"*30)
            print(actions)
            time0 = time.time()
            DT = 1 / FPS
            for action in actions:
                t0 = time.time()
                # base_action = calibrate_linear_vel(base_action, c=0.19)
                # base_action = postprocess_base_action(base_action)
                ts = env.step(action, get_tracer_vel=True)
                time.sleep(max(0, DT - (time.time() - t0)))
                print(f'padding: {time.time() - t0:.2f}s')
            print(f'Avg fps: {len(actions) / (time.time() - time0)}')
            '''
        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')

        # if save_episode:
        #     save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    # save success rate to txt
    result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))

    return success_rate, avg_return

if __name__ == '__main__':
    rospy.init_node("humanplus", anonymous=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', action='store', type=str, help='config file', required=False)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', default="rearrange_objects/_data_rearrange_objects_HIT_resnet18_True", action='store', type=str, help='ckpt_dir')
    parser.add_argument('--policy_class', default="HIT", action='store', type=str, help='policy_class, capitalize')
    parser.add_argument('--task_name', default="data_rearrange_objects", action='store', type=str, help='task_name')
    
    parser.add_argument('--batch_size', default=32, action='store', type=int, help='batch_size')
    parser.add_argument('--seed', default=0, action='store', type=int, help='seed')
    parser.add_argument('--num_steps', default=10000, action='store', type=int, help='num_steps')
    parser.add_argument('--lr', default=1e-5, action='store', type=float, help='lr')
    parser.add_argument('--load_pretrain', action='store_true', default=False)
    parser.add_argument('--pretrained_path', action='store', type=str, help='pretrained_path', required=False)
    
    parser.add_argument('--eval_every', action='store', type=int, default=100000, help='eval_every', required=False)
    parser.add_argument('--validate_every', action='store', type=int, default=1000, help='validate_every', required=False)
    parser.add_argument('--save_every', action='store', type=int, default=10000, help='save_every', required=False)
    parser.add_argument('--resume_ckpt_path', action='store', type=str, help='resume_ckpt_path', required=False)
    parser.add_argument('--skip_mirrored_data', action='store_true')
    parser.add_argument('--actuator_network_dir', action='store', type=str, help='actuator_network_dir', required=False)
    parser.add_argument('--history_len', action='store', type=int)
    parser.add_argument('--future_len', action='store', type=int)
    parser.add_argument('--prediction_len', action='store', type=int)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--no_encoder', action='store_true')
    #dec_layers
    parser.add_argument('--dec_layers', action='store', type=int, default=7, required=False)
    parser.add_argument('--nheads', action='store', type=int, default=8, required=False)
    parser.add_argument('--use_pos_embd_image', action='store', type=int, default=0, required=False)
    parser.add_argument('--use_pos_embd_action', action='store', type=int, default=0, required=False)
    
    #feature_loss_weight
    parser.add_argument('--feature_loss_weight', action='store', type=float, default=0.0)
    #self_attention
    parser.add_argument('--self_attention', action="store", type=int, default=1)
    
    #for backbone 
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--same_backbones', action='store_true')
    #use mask
    parser.add_argument('--use_mask', action='store_true')
    
    # for image 
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=360)
    parser.add_argument('--data_aug', action='store_true')
    parser.add_argument('--normalize_resnet', action='store_true') ### not used - always normalize - in the model.forward
    parser.add_argument('--grayscale', action='store_true')
    parser.add_argument('--randomize_color', action='store_true')
    parser.add_argument('--randomize_data', action='store_true')
    parser.add_argument('--randomize_data_degree', action='store', type=int, default=3)
    
    parser.add_argument('--wandb', action='store_true')
    
    parser.add_argument('--model_type', type=str, default="HIT")
    parser.add_argument('--gpu_id', type=int, default=0)
    
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu_id)
    PROJECT_NAME = 'H1'
    WANDB_USERNAME = "WANDB_USERNAME"
    main_train(vars(args))
    rospy.spin()
    
 