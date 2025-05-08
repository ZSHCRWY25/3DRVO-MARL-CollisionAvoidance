
import os
import sys
import gym
import matplotlib.pyplot as plt
import pickle
import shutil
import argparse
from torch import nn
from pathlib import Path
from policy.multi_ppo import multi_ppo
from policy.policy_rnn_ac import rnn_ac
import matplotlib
import numpy as np
import csv


cur_path = Path(__file__).parent
world_abs_path = str(cur_path/'drone_paths.yaml')
counter = 0
parser = argparse.ArgumentParser(description='drl rvo parameters')
par_env = parser.add_argument_group('par env', 'environment parameters') 
par_env.add_argument('--env_name', default='mdin-v1')
par_env.add_argument('--world_path', default='drone_paths.yaml')
par_env.add_argument('--drone_number', type=int, default=3)
par_env.add_argument('--mpi', default=False)
par_env.add_argument('--neighbors_region', default=4)
par_env.add_argument('--neighbors_num', type=int, default=5)
par_env.add_argument('--reward_parameter', type=float, default=(0.5, 2.0, 0.8, -50, 10, 15, 0.4))##
par_env.add_argument('--env_train', default=True)
par_env.add_argument('--random_bear', default=True)
par_env.add_argument('--random_radius', default=False)
par_env.add_argument('--full', default=False)

par_policy = parser.add_argument_group('par policy', 'policy parameters') 
par_policy.add_argument('--state_dim', default=12)
par_policy.add_argument('--rnn_input_dim', default=9)
par_policy.add_argument('--rnn_hidden_dim', default=256)
par_policy.add_argument('--trans_input_dim', default=21)
par_policy.add_argument('--trans_max_num', default=10)
par_policy.add_argument('--trans_nhead', default=1)
par_policy.add_argument('--trans_mode', default='attn')
par_policy.add_argument('--hidden_sizes_ac', default=(256, 256))
par_policy.add_argument('--drop_p', type=float, default=0)
par_policy.add_argument('--hidden_sizes_v', type=tuple, default=(256, 256))  # 16 16
par_policy.add_argument('--activation', default=nn.ReLU)
par_policy.add_argument('--output_activation', default=nn.Tanh)
par_policy.add_argument('--output_activation_v', default=nn.Identity)
par_policy.add_argument('--use_gpu', action='store_true', default=True)   #action='store_true'
par_policy.add_argument('--rnn_mode', default='biGRU')   # LSTM

par_train = parser.add_argument_group('par train', 'train parameters') 
par_train.add_argument('--pi_lr', type=float, default=4e-6)#
par_train.add_argument('--vf_lr', type=float, default=5e-5)
par_train.add_argument('--train_epoch', type=int, default=600)
par_train.add_argument('--steps_per_epoch', type=int, default=300)
par_train.add_argument('--max_ep_len', default=500)
par_train.add_argument('--gamma', default=0.99)
par_train.add_argument('--lam', default=0.97)
par_train.add_argument('--clip_ratio', default=0.2)
par_train.add_argument('--train_pi_iters', default=50)
par_train.add_argument('--train_v_iters', default=50)
par_train.add_argument('--target_kl',type=float, default=0.05)
par_train.add_argument('--render', default=True)
par_train.add_argument('--render_freq', default=5)
par_train.add_argument('--no-con_train', action='store_false', dest='con_train', default=True)
par_train.add_argument('--seed', default=7)
par_train.add_argument('--save_freq', default=50)
par_train.add_argument('--save_figure', default=False)
par_train.add_argument('--figure_save_path', default=str(cur_path / 'fig_save'/'3UAV_1') + '/')
par_train.add_argument('--save_path', default=str(cur_path / 'model_save') + '/')
par_train.add_argument('--save_name', default= 'r')
par_train.add_argument('--load_path', default=str(cur_path / 'model_save')+ '/')
par_train.add_argument('--load_name', default='r8_0/r8_0_check_point_1200.pt')
par_train.add_argument('--save_result', type=bool, default=True)
par_train.add_argument('--lr_decay_epoch', type=int, default=1400)
par_train.add_argument('--max_update_num', type=int, default=10)

args = parser.parse_args()

model_path_check = args.save_path + args.save_name + str(args.drone_number) + '_{}'
model_name_check = args.save_name + str(args.drone_number) +  '_{}'
while os.path.isdir(model_path_check.format(counter)):
    counter+=1

model_abs_path = model_path_check.format(counter) + '/'
model_name = model_name_check.format(counter)


load_fname = args.load_path + args.load_name

env = gym.make(args.env_name, world_name=args.world_path)

test_env = gym.make(args.env_name, world_name=args.world_path)

policy = rnn_ac(env.observation_space, env.action_space, args.state_dim, args.rnn_input_dim, args.rnn_hidden_dim, 
                     args.hidden_sizes_ac, args.hidden_sizes_v, args.activation, args.output_activation, 
                    args.output_activation_v, args.use_gpu, args.rnn_mode, args.drop_p)

ppo = multi_ppo(env, policy, args.pi_lr, args.vf_lr, args.train_epoch, args.steps_per_epoch, args.max_ep_len, args.gamma, 
                args.lam, args.clip_ratio, args.train_pi_iters, args.train_v_iters, args.target_kl, args.render, args.render_freq,
                  args.con_train,  args.seed, args.save_freq, args.save_figure, model_abs_path, model_name, load_fname, args.use_gpu, 
                   args.save_result, counter, test_env, args.lr_decay_epoch, args.max_update_num, args.mpi, args.figure_save_path)

# save hyparameters
if not os.path.exists(model_abs_path):
    os.makedirs(model_abs_path)

f = open(model_abs_path + model_name, 'wb')
pickle.dump(args, f)
f.close()

with open(model_abs_path+model_name+'.txt', 'w') as p:
    print(vars(args), file=p)
p.close()


ppo.training_loop()

# 获取每个智能体每回合的回报
all_epochs_mean = ppo.all_epochs_mean
all_epochs_mean_transposed = list(map(list, zip(*all_epochs_mean)))

all_epochs_max_diviation = ppo.min_diviation
all_epochs_max_diviation_transposed = list(map(list, zip(*all_epochs_max_diviation)))

all_epochs_max_extra_len =ppo.extra_len
all_epochs_max_extra_len_transposed = list(map(list, zip(*all_epochs_max_extra_len )))


# 保存 all_epochs_mean_transposed 到 CSV 文件
with open('all_epochs_mean_transposed.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    # 写入标题行（可选）
    csvwriter.writerow(['Drone 1', 'Drone 2', 'Drone 3'])  # 根据实际无人机数量调整
    # 写入数据
    for epoch in all_epochs_mean_transposed:
        csvwriter.writerow(epoch)

# # 保存 all_epochs_max_diviation_transposed 到 CSV 文件
# with open('all_epochs_max_diviation_transposed.csv', 'w', newline='') as csvfile:
#     csvwriter = csv.writer(csvfile)
#     # 写入标题行（可选）
#     csvwriter.writerow(['Drone 1', 'Drone 2', 'Drone 3'])  # 根据实际无人机数量调整
#     # 写入数据
#     for epoch in all_epochs_max_diviation_transposed:
#         csvwriter.writerow(epoch)

# # 保存 all_epochs_max_extra_len_transposed 到 CSV 文件
# with open('all_epochs_max_extra_len_transposed.csv', 'w', newline='') as csvfile:
#     csvwriter = csv.writer(csvfile)
#     # 写入标题行（可选）
#     csvwriter.writerow(['Drone 1', 'Drone 2', 'Drone 3'])  # 根据实际无人机数量调整
#     # 写入数据
#     for epoch in all_epochs_max_extra_len_transposed:
#         csvwriter.writerow(epoch)


        
# 绘制图表
plt.figure(figsize=(10, 5))
for i in range(len(all_epochs_mean_transposed[0])):
    rewards = [epoch[i] for epoch in all_epochs_mean_transposed]
    plt.plot(rewards, label=f'Drone {i+1}')

plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward for Each Drone')
plt.legend()
plt.savefig('3UAV.png')
plt.show()

# plt.figure(figsize=(10, 5))
# for i in range(len(all_epochs_max_diviation_transposed[0])):
#     max_diviations = [epoch[i] for epoch in all_epochs_max_diviation_transposed]
#     plt.plot(max_diviations, label=f'Drone {i+1}')

# plt.xlabel('Episod')
# plt.ylabel('Diviation')
# plt.title('Diviation for Each Drone')
# plt.legend()
# plt.savefig('3UAV.png')
# plt.show()



# plt.figure(figsize=(10, 5))
# for i in range(len(all_epochs_max_extra_len_transposed[0])):
#     extar_lens = [epoch[i] for epoch in all_epochs_max_extra_len_transposed]
#     plt.plot(extar_lens, label=f'Drone {i+1}')

# plt.xlabel('Episode')
# plt.ylabel('Extar_len')
# plt.title('Extar_len for Each Drone')
# plt.legend()
# plt.savefig('3UAV.png')
# plt.show()



