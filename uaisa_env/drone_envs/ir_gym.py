
from uaisa_env.drone_envs.env_base import env_base
from uaisa_env.vel_obs.rvo_inter import rvo_inter


# from uaisa.uaisa_env.drone_envs.env_base import env_base
# from uaisa.uaisa_env.vel_obs.rvo_inter import rvo_inter

from math import sqrt, pi, acos, degrees

from gym import spaces

import numpy as np
import math 
import torch

class ir_gym(env_base):
    def __init__(self, world_name, neighbors_region=5, neighbors_num=10, vxmax = 2, vymax = 2, vzmax = 2, env_train=True, acceler = 0.5,**kwargs):
        super(ir_gym, self).__init__(**kwargs)#（改完）

        # self.obs_mode = kwargs.get('obs_mode', 0)    # 0 drl_rvo, 1 drl_nrvo
        # self.reward_mode = kwargs.get('reward_mode', 0)

        self.env_train = env_train#环境是否处于训练模式

        self.nr = neighbors_region
        self.nm = neighbors_num#邻居区域和邻居数量

        self.rvo = rvo_inter(neighbors_region, neighbors_num, vxmax, vymax,vzmax, acceler, env_train)

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(21,), dtype=np.float32)#观测空间为21维
        self.action_space = spaces.Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]), dtype=np.float32)#动作空间为3维
        
        self.acceler = acceler
        self.arrive_flag_cur = False#到达标志

        self.rvo_state_dim = 9#RVO（Reciprocal Velocity Obstacles）状态维度
        self.polt_cone = False
        self.init_goal_dist = 100

        
        

    def cal_des_list(self):#计算所有无人机的目标速度列表。(改完)
        des_vel_list = [drone.cal_des_vel() for drone in self.drone_list]
        return des_vel_list



    def rvo_reward_list_cal(self, action_list, **kwargs):#计算机器人在执行一系列动作时的 RVO（Reciprocal Velocity Obstacles）奖励列表   （改完了，在mrnav中调用） 
        drone_state_list = self.components['drones'].total_states() # #所有无人机状态,这个函数在env_drons里面
        rvo_reward_list = []
        vo_flag_list= []
        for i, (drone_state, action) in enumerate(zip(drone_state_list, action_list)):  
        # 排除当前无人机状态，获取其他无人机的状态列表  
            other_drone_states = [s for j, s in enumerate(drone_state_list) if j != i]  
            rvo_reward, vo_flag = self.rvo_reward_cal(drone_state, other_drone_states, action, **kwargs)#所有无人机选择的动作进行打分  
            rvo_reward_list.append(rvo_reward)
            vo_flag_list.append(vo_flag)
        if self.plot:
            self.world_plot.drow_vel()
        return rvo_reward_list,vo_flag_list
    
    def rvo_reward_cal(self, drone_state, other_drone_states, action, 
                 RVO_PARAMS = (
                                -2.5,   # p_base: 基础安全惩罚
                                0.3,     # p_dist: 安全距离系数
                                -8.0,    # p_reverse: 反向运动惩罚
                                -0.5,     # p_angle: 角度惩罚系数
                                -8.0     # p_urgent: 紧急惩罚系数 
                            ), **kwargs):
        """
        改进的RVO避障奖励函数
        Args:
            RVO_PARAMS: (base_safe, dist_coeff, reverse_penalty, angle_coeff, urgency_penalty)

        """
        p_base, p_dist, p_reverse, p_angle, p_urgent = RVO_PARAMS
        
        # 获取避障信息
        vo_flag, min_exp_time, min_dis = self.rvo.config_vo_reward(
            drone_state, other_drone_states, self.building_list, action, **kwargs
        )
        
        # 期望速度方向
        des_vel = np.round(np.squeeze(drone_state[8:11]), 3)
        # des_dir = des_vel / (np.linalg.norm(des_vel) + 1e-6)  # 单位向量
        vel_penalty = 0.2*  np.linalg.norm(action) / np.linalg.norm(des_vel)

        angle_radians = self.calculate_angle_between_vectors(des_vel, action)#####所选速度与期望速度角度的差（-pi---pi
        if -pi/18< angle_radians and angle_radians <pi/18:
            angle_punish = 3
        elif -pi/6< angle_radians and angle_radians <pi/6:
            angle_punish = 1
        elif -pi/3< angle_radians and angle_radians <pi/3:
            angle_punish = 0.5
        elif -pi/2< angle_radians and angle_radians <pi/2:
            angle_punish = 0
        else:
            angle_punish = -4
        
        # ---- 动态惩罚项 ----
        # angle_rad = self.calculate_angle_between_vectors(des_dir, action)
        
        # 1. 连续角度惩罚（指数形式）
        # angle_penalty = p_angle * (angle_rad / np.pi)
        
        # # 2. 反向运动惩罚（夹角>90度时强化）
        # if angle_rad > np.pi/2:
        # # 使用 1 - cos(theta) 强化反向惩罚（值域 [0, 2]）
        #     reverse_penalty = p_reverse * (1 - np.cos(angle_rad))
        # else:
        #     reverse_penalty = 0
        
        # 3. 动态安全奖励
        safety_reward = 0
        if vo_flag:
            # 安全距离奖励（基于反比例函数）
            # dist_reward = p_dist * (2/(1 + np.exp(-min_dis/2)) - 1)  # 修正后公式
            
            # 紧急程度惩罚（指数衰减）
            urgency_penalty = 0
            if min_exp_time < 2:
                urgency_penalty = p_urgent * np.exp(-min_exp_time/0.5)
            
            safety_reward = p_base + urgency_penalty

        # print('angle_punish=',angle_punish )
        # print('vel_penalty=',vel_penalty )
        
        total_rvo_reward = angle_punish + vel_penalty + safety_reward 

        return np.round(total_rvo_reward, 3), False #vo_flag
    

    def obs_move_reward_list(self, action_list, **kwargs):#计算机器人执行一系列动作时的观测和奖励
        drone_state_list = self.components['drones'].total_states() 
        obs_reward_list = []
        for i, (drone, action) in enumerate(zip(self.drone_list , action_list)):  
        # 排除当前无人机状态，获取其他无人机的状态列表  
            other_drone_state_list = [s for j, s in enumerate(drone_state_list ) if j != i]
            obs, reward, done, info, finish = self.observation_reward(drone, other_drone_state_list, action)
            #obs, reward, done, info = 
            obs_reward_list.append((obs, reward, done, info, finish))

        observation_list = [l[0] for l in obs_reward_list]
        reward_list = [l[1] for l in obs_reward_list]
        done_list = [l[2] for l in obs_reward_list]
        info_list = [l[3] for l in obs_reward_list]
        finish_list = [l[4] for l in obs_reward_list]
        #计算观测和奖励。
#       返回观测列表、奖励列表、完成标志列表和其他信息列表。

        return observation_list, reward_list, done_list, info_list, finish_list 

    def observation_reward(self, drone, odro_state_list,action):#算无人机的观测和奖励。

       # 计算无人机的内部观测和外部观测。
       # 返回观测、奖励、完成标志和其他信息。
        drone_state = drone.dronestate()
        des_vel = np.squeeze(drone.cal_des_vel())
        destination_arrive_reward_flag = False
        len_reward_flag = False
        done = False
        waypoint_num = drone.i
        n_points = n_points = drone.n_points - 1
        
        if drone.arrive(drone.state, drone.current_des) and not drone.arrive_flag:##到途经航路点奖励5
            drone.arrive_flag = True
            arrive_reward_flag = True
            waypoint_num = drone.i
        else:
            arrive_reward_flag = False

        if drone.arrive_flag == True:

            if drone.destination_arrive(drone.state) and not drone.destination_arrive_flag:#到最终目的地奖励
                drone.destination_arrive_flag = True
                destination_arrive_reward_flag = True
            else:
                destination_arrive_reward_flag = False

        deviation = drone.Deviation_from_route()

        dist_to_goat = self.dist_to_goat(drone.state, drone.current_des)

        real_route_len = drone.real_route_len
        route_len =drone.route_len 
        exlen = real_route_len - route_len + 4


        if exlen > 0:
            len_reward_flag = True


        #drone_state, drone_state_list, building_list, action
        obs_vo_list, vo_flag, min_exp_time, collision_flag, obs_building_list = self.rvo.config_vo_inf(drone_state, odro_state_list, self.building_list, action)
        #obs_vo_list_nm, vo_flag, min_exp_time, collision_flag, obs_building_list
        
        if drone.drone_out_map(self.map_size):
            collision_flag = True

        cur_vel = np.squeeze(drone.vel)
        radius = drone.radius_collision* np.ones(1,)
        pr = drone.priority* np.ones(1,)
        d =deviation* np.ones(1,)

        propri_obs = np.concatenate([drone.state, cur_vel, radius, pr, des_vel, d]) ##内部观测[3,3,1,1,3,1],13维 
        #状态向量维度[state, cur_vel, radius, pra, des_vel, deviation])
        
        if len(obs_vo_list) == 0:
            exter_obs_vo = np.zeros((self.rvo_state_dim,))
        else:
            exter_obs_vo = np.concatenate(obs_vo_list) # vo list外部观测，#[x, y, z, ve_x, ve_y, ve_z, α, min_dis, input_exp_time] 9维*n(n最大10)

        if len(obs_building_list) == 0:
            exter_obs_building = np.zeros((self.rvo_state_dim,))
        else:
            exter_obs_building = np.concatenate(obs_building_list) # vo list外部观测
        
        self.contains_nan(propri_obs)
        
        self.contains_nan(exter_obs_vo)

        # print(propri_obs)
        # print("exter_obs_vo=", exter_obs_vo)
        # print("---------------------------------")
            
        observation = np.round(np.concatenate([propri_obs, exter_obs_vo]), 2)##链接，21维
        # print(observation)
        # print("+++++++++++++++++++++++++++++++")
        if np.isnan(observation).any() or np.isinf(observation).any():
            print(observation)
            print('*******')
            print("drone.id",drone.id)
            print("drone.vel",cur_vel)
            print("drone.pos",drone.state)
            print("drone.pos",drone_state)
            raise ValueError("原始观测数据包含 NaN/Inf！")


        # dis2goal = sqrt( robot.state[0:2] - robot.goal[0:2
        mov_reward = self.mov_reward(collision_flag, arrive_reward_flag, waypoint_num, n_points, destination_arrive_reward_flag, 
                                     deviation,dist_to_goat, len_reward_flag, exlen, min_exp_time)

        reward = mov_reward

        done = True if collision_flag else False
        info = True if drone.arrive_flag else False
        finish = True if drone.destination_arrive_flag else False
        #done = True if drone.destination_arrive_flag else False
        #info也可以处理更多信息：info = {'arrive_flag': drone.arrive_flag, 'destination_arrive_flag': drone.destination_arrive_flag}  
        
        return observation, reward, done, info, finish##[内外观测（自身+其他无人机速度障碍+冲突建筑物），移动奖励（碰撞+到点+终点），碰撞标准，到点标志 ，到终点标志]

    def mov_reward(self, collision_flag, arrive_reward_flag, waypoint_num, n_points, 
              destination_arrive_flag, deviation, dist_to_goat, len_reward_flag, exlen, min_exp_time,
              MOVE_PARAMS = (
                        5.0,    # p_arrive: 基础到达奖励
                        3.0,    # p_way: 航点奖励系数
                        20.0,   # p_dest: 终点奖励系数
                        -2.5,   # p_dev: 偏离惩罚系数
                        -0.3,   # p_exlen: 航程惩罚系数
                        10.0     # p_progress: 进度奖励系数
                    )):
        """
        改进的导航奖励函数
        Args:
            reward_parameter: (base_arrive, waypoint_coeff, dest_coeff, dev_coeff, exlen_coeff, progress_coeff)
        """

        p_arrive, p_way, p_dest, p_dev, p_exlen, p_progress = MOVE_PARAMS
        
        # ---- 基础奖励 ----
        reward = 0
        
        # 1. 碰撞惩罚
        if collision_flag:
            return -50  # 碰撞时直接返回大惩罚
        
        # 2. 航点到达奖励（递减式）
        if arrive_reward_flag:
            way_decay = 0.95 ** (n_points - waypoint_num)  # 后续航点奖励衰减
            reward += p_way * way_decay
        
        # 3. 终点到达奖励
        if destination_arrive_flag:
            reward += p_dest

        # ---- 连续惩罚项 ----
        # 4. 路径偏离惩罚（S型曲线）
        dev_penalty =self.calculate_penalty_with_exp(deviation)

        if len_reward_flag:
            # 5. 航程效率惩罚
            exlen_penalty = p_exlen * np.log(exlen + 1+ 1e-6)  # 对数形式减缓增长
            if exlen_penalty < -6 or np.isnan(exlen_penalty):
                exlen_penalty = -6
        else:
            exlen_penalty = 0

        # 6. 进度奖励（基于相对距离）
        # progress = (self.init_goal_dist - dist_to_goat) / self.init_goal_dist
        # progress_reward = p_progress * (1 - np.exp(-2*progress))
        # print('reward=',reward)
        # print('dev_penalty=',dev_penalty)
        # print('exlen_penalty=',exlen_penalty)

        total_reward = reward + dev_penalty + exlen_penalty

        return np.round(total_reward, 3)

    def osc_reward(self, state_list):#避免轨迹振荡（oscillation）检查状态列表中的角度变化，如果出现振荡则返回负奖励
        # to avoid oscillation
        dif_rad_list = []
        
        if len(state_list) < 3:#状态列表 state_list,三个以上
            return 0

        for i in range(1,len(state_list) - 1):#计算相邻状态之间的角度变化（差值）
            angle1 = self.calculate_angle_between_vectors(state_list[i+1][3:6], state_list[i][3:6])#[x, y, z, vx, vy, vz, radius, pra, vx_des, vy_des, vz_des]
            angle2 = self.calculate_angle_between_vectors(state_list[i][3:6], state_list[i-1][3:6])
            dif = self.wraptopi(angle1 - angle2)
            dif_rad_list.append(round(dif, 2))
            
        for j in range(len(dif_rad_list) - 3):  
            # 检测连续三个角度变化的方向是否相反   
            if (dif_rad_list[j] > 0 and dif_rad_list[j+1] < 0 and dif_rad_list[j+2] > 0) or \
                (dif_rad_list[j] < 0 and dif_rad_list[j+1] > 0 and dif_rad_list[j+2] < 0):
                print('osc', dif_rad_list[j], dif_rad_list[j+1], dif_rad_list[j+2], dif_rad_list[j+3])  
                return -10  
        return 0

    def observation(self, drone, other_drone_state_list):#计算观测
        drone_state = drone.dronestate() #提取当前位置、速度、大小、优先级、期望速度return np.concatenate((self.state, self.vel, [rc_array[0]],[priority[0]], v_des, [deviation[0]]))
        des_vel = np.squeeze(drone_state[-3:])# state: [x, y, z, vx, vy, vz, radius, pra, vx_des, vy_des, vz_des]
        
        obs_vo_list, _, min_dis, _ , obs_building_list= self.rvo.config_vo_inf(drone_state, other_drone_state_list, self.building_list)#如果存在速度障碍物（VO），还计算外部观测
        cur_vel = np.squeeze(drone.vel)
        radius = drone.radius_collision* np.ones(1,)

        if len(obs_vo_list) == 0:
            exter_obs = np.zeros((self.rvo_state_dim,))
        else:
            exter_obs = np.concatenate(obs_vo_list) # vo list外部观测变量

        if len(obs_building_list) == 0:
            exter_obs_building = np.zeros((self.rvo_state_dim,))
        else:
            exter_obs_building = np.concatenate(obs_building_list) # vo list外部观测

        
        propri_obs = drone_state #内部变量[state, cur_vel, radius, pra, des_vel, deviation])
        #return np.concatenate((self.state, self.vel, [rc_array[0]],[priority[0]], v_des, [deviation[0]]))
        observation = np.round(np.concatenate([propri_obs, exter_obs,]), 2)#将内部和外部观测连接成最终的观测向量。


        return observation

    def env_reset(self):#重置环境

        self.components['drones'].drones_reset()
        drone_state_list = self.components['drones'].total_states()# state: [x, y, z, vx, vy, vz, radius, pra, vx_des, vy_des, vz_des]
        obs_list = list(map(lambda drone: self.observation(drone, drone_state_list ), self.drone_list))
        rvo_observation_list = []
        self.render()
        return obs_list

    def env_reset_one(self, id):#重置环境中的特定无人机，在多智能体场景中逐个重置无人机
        self.drone_reset(id)

    def env_observation(self):#计算环境中所有无人机的观测，类似于 env_reset，但不重置环境或无人机状态
        drone_list = self.components['drones'].drone_list
        drone_state_list = self.components['drones'].total_states() 
        observation_list = []
        for i, drone in enumerate(drone_list):  
        # 排除当前无人机状态，获取其他无人机的状态列表  
            other_drone_state = [s for j, s in enumerate(drone_state_list ) if j != i]

            observation = self.observation(drone, other_drone_state)
            observation_list.append((observation))

        return observation_list
    
    def render(self, time=0.05, **kwargs):

        if self.plot:
            self.world_plot.clear_plot_elements()
            self.world_plot.draw_drones(**kwargs)
            self.world_plot.pause(time)

        # if polt_cone and len(rvo_observation_list) > 0:#[x, y, z, ve_x, ve_y, ve_z, α, min_dis, input_exp_time]
        #     for rvo_obs in rvo_observation_list:
        #         if not np.all(rvo_obs == 0):
        #             vertex = rvo_obs[0:3]
        #             axis = rvo_obs[3:6]
        #             angle_degrees = rvo_obs[6]
        #             self.world_plot.draw_cone(vertex, axis, angle_degrees)
                    
            
        self.time = self.time + time

    def draw_rvo(self,rvo_observation_list):
        if len(rvo_observation_list) > 0:#[x, y, z, ve_x, ve_y, ve_z, α, min_dis, input_exp_time]
            for rvo_obs in rvo_observation_list:
                if not np.all(rvo_obs == 0):
                    vertex = rvo_obs[0:3]
                    axis = rvo_obs[3:6]
                    angle_degrees = rvo_obs[6]
                    if self.plot:
                        self.world_plot.draw_cone(vertex, axis, angle_degrees)


    def indicators_deviation(self):
        deviation = self.components['drones'].deviation_indicator()
        return deviation
    
    def indicators_extra_len(self):
        extar_len =  self.components['drones'].extar_len_indicator()
        return extar_len 
    

    def calculate_angle_between_vectors(self, vec1, vec2):
        """改进的向量夹角计算（处理零向量）"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 < 1e-6 and norm2 < 1e-6:  # 均为零向量
            return 0.0
        elif norm1 < 1e-6 or norm2 < 1e-6:  # 仅一向量为零
            return np.pi  # 零向量与任何向量视为180°夹角（最大惩罚
            
        cos_theta = np.dot(vec1, vec2) / (norm1 * norm2)
        return np.arccos(np.clip(cos_theta, -1.0, 1.0))

    @staticmethod
    def wraptopi(theta):

        if theta > pi:
            theta = theta - 2*pi
        
        if theta < -pi:
            theta = theta + 2*pi

        return theta
    
    @staticmethod
    def calculate_angle_between_vectors(A, B):
        """计算两个三维向量之间的夹角（弧度）"""
        # 添加数值稳定性保护
        epsilon = 1e-8
        
        # 1. 计算向量模长（添加极小值保护）
        mag_A = sqrt(A[0]**2 + A[1]**2 + A[2]**2 + epsilon)
        mag_B = sqrt(B[0]**2 + B[1]**2 + B[2]**2 + epsilon)
        
        # 2. 计算点积（数值安全版本）
        dot_product = (
            A[0] * B[0] + 
            A[1] * B[1] + 
            A[2] * B[2]
        )
        
        # 3. 处理零向量特殊情况
        if mag_A < 1e-6 or mag_B < 1e-6:
            return 0.0  # 定义零向量的夹角为0度
        
        # 4. 计算余弦值（带二次保护）
        cos_theta = dot_product / (mag_A * mag_B)
        cos_theta = np.clip(cos_theta, -1.0 + epsilon, 1.0 - epsilon)
        
        # 5. 安全计算反余弦
        return acos(cos_theta)
    

    @staticmethod
    def calculate_penalty_with_exp(deviation):
        d = deviation * 10  
        p_dev = -1.5
        steepness = 0.3

        penalty = p_dev * (2 / (1 + np.exp(-(d-5)/steepness)))
        # d = deviation * 10  
        # if d < 2:  
        #     penalty = -0.04*d  
        # elif 2 <= d < 5 :  
        #     penalty =  -2*d - 10
        # else:
        #     penalty = -5
        return penalty    
    
    @staticmethod
    def dist_to_goat(state, current_des):
        position = state[0:3]
        dist = np.linalg.norm(position - current_des[0:3])
        return dist

    @staticmethod
    def calculate_angle(vector_a, vector_b):
        """
        Calculate the angle between two 3D vectors and scale it to the range of [-pi, pi].

        :param vector_a: The first vector as a tuple or list (x, y, z).
        :param vector_b: The second vector as a tuple or list (x, y, z).
        :return: The angle in radians between the two vectors, scaled to [-pi, pi].
        """
        # Calculate the dot product of the two vectors
        dot_product = vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1] + vector_a[2] * vector_b[2]
    
        # Calculate the magnitudes of the vectors
        magnitude_a = sqrt(vector_a[0]**2 + vector_a[1]**2 + vector_a[2]**2)
        magnitude_b = sqrt(vector_b[0]**2 + vector_b[1]**2 + vector_b[2]**2)
    
        # Calculate the cosine of the angle using the dot product theorem
        cos_theta = dot_product / (magnitude_a * magnitude_b)
        cos_theta = max(-1, min(1, cos_theta))
    
        # Calculate the angle in radians
        theta = acos(cos_theta)
    
        # Scale the angle to the range of [-pi, pi]
        # This step is not strictly necessary as acos will return a value in the range of [0, pi]
        # but it's included here for clarity and to show how you might scale a value if needed.
        theta_scaled = theta if theta <= pi else -(2 * pi - theta)
    
        return theta_scaled
    
    @staticmethod
    def contains_nan(vector):
        """
        检测向量是否包含NaN值
        支持类型：Python列表、NumPy数组、PyTorch张量
        返回：布尔值（True表示存在NaN）
        """
        # 检查输入类型
        if isinstance(vector, list):
            return any(math.isnan(x) for x in vector if isinstance(x, (float, np.floating)))
        
        elif isinstance(vector, np.ndarray):
            return np.isnan(vector).any()
        
        elif isinstance(vector, torch.Tensor):
            return torch.isnan(vector).any().item()
        
        else:
            raise TypeError(f"不支持的数据类型: {type(vector).__name__}")
