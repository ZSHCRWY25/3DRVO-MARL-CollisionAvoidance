
import numpy as np
import torch
from torch.optim import Adam
from pathlib import Path
import scipy
import scipy.signal
import time
import os
from policy.post_train import post_train
import threading



def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class multi_PPObuf:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):#

        # gamma: discount factor
        # Lambda for GAE-Lambda. (Always between 0 and 1, close to 1.)

        self.obs_buf = [0] * size
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
    
    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size
        #观测、动作、奖励、价值和日志概率。
        self.obs_buf[self.ptr] = obs.copy()
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1#指针

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self):     
        assert self.ptr == self.max_size    # buffer has to be full before you can get

        self.ptr, self.path_start_idx = 0, 0

        act_ten = torch.as_tensor(self.act_buf, dtype=torch.float32)
        ret_ten = torch.as_tensor(self.ret_buf, dtype=torch.float32)
        adv_ten = torch.as_tensor(self.adv_buf, dtype=torch.float32)
        logp_ten = torch.as_tensor(self.logp_buf, dtype=torch.float32)
        obs_tensor_list = list(map(lambda o: torch.as_tensor(o, dtype=torch.float32), self.obs_buf))
        data = dict(obs=obs_tensor_list, act=act_ten, ret=ret_ten,
                    adv=adv_ten, logp=logp_ten)
        return data

    def complete(self):
        self.ptr, self.path_start_idx = 0, 0

class multi_ppo:
    def __init__(self, env, ac_policy, pi_lr=3e-4, vf_lr=1e-3, train_epoch=50, steps_per_epoch = 600, 
                 max_ep_len=300, gamma=0.99, lam=0.97, clip_ratio=0.2, train_pi_iters=100, train_v_iters=100, 
                 target_kl=0.01, render=False, render_freq=20, con_train=True, seed=7, save_freq=50, save_figure=False, 
                 save_path='test/', save_name='test', load_fname=None, use_gpu = False, save_result=False, 
                 counter=0, test_env=None, lr_decay_epoch=1000, max_update_num=10, mpi=False, figure_save_path=None, **kwargs):

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed) 
        np.random.seed(seed)

        self.env = env
        self.ac = ac_policy
        self.con_train=con_train
        self.robot_num = env.ir_gym.drone_num
        self.obs_dim = env.observation_space.shape
        self.act_dim = env.action_space.shape

        # Set up optimizers for policy and value function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.vf_optimizer = Adam(self.ac.v.parameters(), lr=vf_lr)

        if con_train:
            check_point = torch.load(load_fname)
            self.ac.load_state_dict(check_point['model_state'], strict=True)
            self.ac.train()

        # parameter
        self.epoch = train_epoch
        self.max_ep_len = max_ep_len
        self.steps_per_epoch = steps_per_epoch
        self.all_epochs_mean = [[] for _ in range(self.robot_num)]
        
        self.buf_list = [multi_PPObuf(self.obs_dim, self.act_dim, steps_per_epoch, gamma, lam) for i in range(self.robot_num)]
        #obs_dim, act_dim, size, gamma=0.99, lam=0.95)

        # update parameters
        self.clip_ratio = clip_ratio
        self.train_pi_iters = train_pi_iters


        self.train_v_iters=train_v_iters
        self.target_kl=target_kl    

        self.render = render
        self.render_freq = render_freq

        self.save_freq = save_freq
        self.save_path = save_path
        self.figure_save_path = figure_save_path
        self.save_name = save_name
        self.save_figure = save_figure  
        self.use_gpu = use_gpu 

        self.save_result = save_result
        self.counter = counter
        self.pt = post_train(test_env, inf_print=False, render=False)
        torch.cuda.synchronize()

        self.lr_decay_epoch = lr_decay_epoch
        self.max_update_num = max_update_num
        self.ep_ret_list_all_epochs = [[] for _ in range(self.robot_num)]
        self.mpi = mpi
        self.min_diviation = [[] for _ in range(self.robot_num)]
        self.extra_len = [[] for _ in range(self.robot_num)]

        # if self.mpi:
        #     self.comm = MPI.COMM_WORLD
        #     self.rank = self.comm.Get_rank()

    def training_loop(self):

        obs_list, ep_ret_list, ep_len_list = self.env.drone_reset(self.render), [0] * self.robot_num, [0] * self.robot_num
        ep_ret_list_mean = [[] for i in range(self.robot_num)]
        deviation_list = [[] for i in range(self.robot_num)]#
        extra_len_list = [[] for i in range(self.robot_num)]



        for epoch in range(self.epoch + 1):
            start_time = time.time()
            print('current epoch', epoch)

            if self.mpi:
                state_dict = self.comm.bcast(self.ac.state_dict(), root=0)
                self.ac.load_state_dict(state_dict)

            for t in range(self.steps_per_epoch):#

                # if self.render and (epoch % self.render_freq == 0 or epoch == self.epoch):
                #     self.env.drone_render(save=self.save_figure, path=self.figure_save_path, i = t )

                # if self.save_figure and epoch == 1:
                #     self.env.render(save=True, path=self.save_path+'figure/', i=t)

                a_list, v_list, logp_list, abs_action_list = [], [], [], []
            
                for i in range(self.robot_num):
                    obs = obs_list[i]

                    a_inc, v, logp = self.ac.step(torch.as_tensor(obs, dtype=torch.float32))
                    a_inc = np.round(a_inc, 2)
                    a_list.append(a_inc)
                    v_list.append(v)
                    logp_list.append(logp)

                    cur_vel = np.squeeze(self.env.ir_gym.drone_list[i].vel)
                    abs_action = self.env.ir_gym.acceler * np.round(a_inc, 2)  + cur_vel
                    # abs_action = 1.5*a_inc
                    abs_action = np.round(abs_action, 2)
                    #print(abs_action)
                    self.env.ir_gym.drone_list[i].rvo_vel = abs_action
                    abs_action_list.append(abs_action)

                next_obs_list, reward_list, done_list, info_list, finish_list = self.env.drone_step(abs_action_list)
                #获取指标
                deviation = self.env.ir_gym.indicators_deviation()
                if self.render and (epoch % self.render_freq == 0 or epoch == self.epoch):
                    self.env.drone_render(save=self.save_figure, path=self.figure_save_path, i = t )

                # save to buffer
                for i in range(self.robot_num):
                    
                    self.buf_list[i].store(obs_list[i], a_list[i], reward_list[i], v_list[i], logp_list[i])
                    ep_ret_list[i] += reward_list[i]
                    ep_len_list[i] += 1

                # Update obs 
                obs_list = next_obs_list[:]

                epoch_ended = t == self.steps_per_epoch-1
                arrive_all = min(finish_list) == True
                collision = max(done_list)
                terminal = max(finish_list) == True or max(ep_len_list) > self.max_ep_len
                if collision:
                    for i in range(self.robot_num):
                        if done_list[i]:
                            deviation_list[i].append(deviation[i])
                            extra_len_list[i].append(0)
                            self.env.drone_reset_one(self.render, i)
                            ep_ret_list_mean[i].append(ep_ret_list[i])
     
                            ep_ret_list[i] = 0
                            ep_len_list[i]= 0


                    obs_list = self.env.ir_gym.env_observation()

                if epoch_ended or arrive_all:#如果回合结束，或无人机全部到达：重置环境、清空回合回报和长度、完成路径

                    extra_len = self.env.ir_gym.indicators_extra_len()

                    if epoch + 1 % 300 == 0:
                        obs_list = self.env.drone_reset(self.render)
                    else:
                        obs_list = self.env.drone_reset(self.render)
                    
                    for i in range(self.robot_num):
                        
                        if arrive_all:
                            ep_ret_list_mean[i].append(ep_ret_list[i])
                            arrive_all= False
                            deviation_list[i].append(deviation[i])
                            extra_len_list[i].append(extra_len[i])

                        ep_ret_list[i] = 0
                        ep_len_list[i] = 0

                        self.buf_list[i].finish_path(0)#结束路径并切片计算回报

                elif terminal:#如果某个机器人到达结束条件：重置单个机器人、保存回合回报、清空回合回报和长度、完成路径

                    for i in range(self.robot_num):
                        if finish_list[i] or ep_len_list[i] > self.max_ep_len:
                            extra_len = self.env.ir_gym.indicators_extra_len()
                            deviation_list[i].append(deviation[i])
                            extra_len_list[i].append(extra_len[i])
                            self.env.drone_reset_one(self.render,i)
                            ep_ret_list_mean[i].append(ep_ret_list[i])
                            self.ep_ret_list_all_epochs[i].append(ep_ret_list[i]) 
                            ep_ret_list[i] = 0
                            ep_len_list[i]= 0

                        self.buf_list[i].finish_path(0)
                    
                    obs_list = self.env.ir_gym.env_observation()

            if (epoch % self.save_freq == 0) or (epoch == self.epoch):
                self.save_model(epoch)
                if self.save_result and epoch != 0:
                # if self.save_result:
                    policy_model = self.save_path + self.save_name+'_'+str(epoch)+'.pt'#### ... 保存模型文件名设置 ...  
                    # policy_model = self.save_path + self.save_name+'_'+'check_point_'+ str(epoch)+'.pt'
                    result_path = self.save_path
                    policy_name = self.save_name+'_'+str(epoch)
                    thread = threading.Thread(target=self.pt.policy_test, args=('drl', policy_model, policy_name, result_path, '/results.txt'))
                    thread.start()

            mean = [round(np.mean(r), 2) if r else 0 for r in ep_ret_list_mean]               
            max_ret = [round(np.max(r), 2) if r else 0 for r in ep_ret_list_mean]  
            min_ret = [round(np.min(r), 2) if r else 0 for r in ep_ret_list_mean]   
            print('The reward in this epoch: ', 'min', min_ret, 'mean', mean, 'max', max_ret)
            ep_ret_list_mean = [[] for i in range(self.robot_num)]

            # min_deviation = [round(np.min(r), 2) if r else 0 for r in deviation_list] 
            # max_extra_len = [round(np.mean(r), 2) if r else 0 for r in extra_len_list]

            # deviation_list = [[] for i in range(self.robot_num)]#
            # extra_len_list = [[] for i in range(self.robot_num)]

            # for m in range(len(mean)):
            #     self.all_epochs_mean[m].append(mean[m])

            # for m in range(len(min_deviation)):
            #     self.min_diviation[m].append(min_deviation[m])

            # for m in range(len(max_extra_len )):
            #     self.extra_len[m].append(max_extra_len[m])

            

            # update
            # self.update()
            data_list = [buf.get() for buf in self.buf_list]
            if self.mpi:#如果使用mpi
                rank_data_list = self.comm.gather(data_list, root=0)#

                if self.rank == 0:
                    for data_list in rank_data_list:
                        self.update(data_list)
            else:
                self.update(data_list)#否则直接在当前进程中更新模型
    
            # animate
            # if epoch == 1:
            #     self.env.create_animate(self.save_path+'figure/')
            #计算了一个epoch的时间成本，并打印出来。同时，它还估算了剩余时间（以小时为单位）。注意，在MPI环境中，只有在根节点（self.rank == 0）上才会打印这些信息
            if self.mpi:
                if self.rank == 0:
                    time_cost = time.time()-start_time 
                    print('time cost in one epoch', time_cost, 'estimated remain time', time_cost*(self.epoch-epoch)/3600, 'hours' )
            else:
                time_cost = time.time()-start_time 
                print('time cost in one epoch', time_cost, 'estimated remain time', time_cost*(self.epoch-epoch)/3600, 'hours' )
            
    def update(self, data_list):
        
        randn = np.arange(self.robot_num)
        np.random.shuffle(randn)#打乱顺序（减少数据顺序偏见
        
        update_num = 0
        for r in randn:  
            
            data = data_list[r]##获取数据
            update_num += 1

            if update_num > self.max_update_num:
                continue

            for i in range(self.train_pi_iters):
                self.pi_optimizer.zero_grad()#清除之前的梯度
                loss_pi, pi_info = self.compute_loss_pi(data)#计算策略网络的损失
                #受一个包含观测值（obs）、动作（act）、优势函数（adv）和旧的对数概率（logp_old）的数据字典
                kl = pi_info['kl']
               
                
                if kl > self.target_kl:
                    print('Early stopping at step %d due to reaching max kl.'%i)
                    break
                
                loss_pi.backward()#反向传播
                torch.nn.utils.clip_grad_norm_(self.ac.parameters(), max_norm=2.0)
                self.pi_optimizer.step()#更新策略网络的参数

            # Value function learning
            for i in range(self.train_v_iters):
                self.vf_optimizer.zero_grad()
                loss_v = self.compute_loss_v(data)#计算价值网络的损失

                loss_v.backward()
                self.vf_optimizer.step()


    def compute_loss_v(self, data):
        obs, ret = data['obs'], data['ret']#观测值，回报
        if self.use_gpu:
            ret = ret.cuda()
        return ((self.ac.v(obs) - ret)**2).mean()#计算价值网络（self.ac.v(obs)）的预测值与真实回报之间的均方误差，并返回该误差的均值

    def compute_loss_pi(self, data):
         # Set up function for computing PPO policy loss
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']#含观测值（obs）、动作（act）、优势函数（adv）和旧的对数概率（logp_old

        if self.use_gpu:
            logp_old = logp_old.cuda()
            adv = adv.cuda()

        # Policy loss
        pi, logp = self.ac.pi(obs, act)#通过self.ac.pi(obs, act)得到新策略下的动作概率（pi）和对数概率（logp）
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()
        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)
        return loss_pi, pi_info

    def save_model(self, index=0):
       
        dir_name = self.save_path
        fname_model = self.save_path + self.save_name+'_{}.pt'
        fname_check_point = self.save_path + self.save_name+'_check_point_{}.pt'
        state_dict = dict(model_state=self.ac.state_dict(), pi_optimizer=self.pi_optimizer.state_dict(), 
        vf_optimizer = self.vf_optimizer.state_dict() )

        if os.path.exists(dir_name):
            torch.save(self.ac, fname_model.format(index))
            torch.save(state_dict, fname_check_point.format(index))
        else:
            os.makedirs(dir_name)
            torch.save(self.ac, fname_model.format(index))
            torch.save(state_dict, fname_check_point.format(index))
                    

                
                
                  

