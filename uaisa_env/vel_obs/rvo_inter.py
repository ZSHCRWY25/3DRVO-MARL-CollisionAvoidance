
from math import sqrt, atan2, asin, sin, pi, cos, inf, acos
import numpy as np

from uaisa_env.vel_obs.reciprocal_vel_obs import reciprocal_vel_obs
from uaisa_env.vel_obs.vel_obs3D import get_alpha, get_PAA,  get_rvo_array, get_beta, cal_vo_exp_tim



class rvo_inter(reciprocal_vel_obs):
    def __init__(self, neighbor_region=5, neighbor_num=10, vxmax=1.5, vymax=1.5, vzmax = 1.5, acceler = 0.3, env_train=True, exp_radius=0.2, ctime_threshold=2, delta_t = 1):
        super(rvo_inter, self).__init__(neighbor_region, vxmax, vymax, vzmax, acceler)

        self.env_train = env_train
        self.exp_radius = exp_radius
        self.nm = neighbor_num
        self.ctime_threshold = ctime_threshold
        self.delta_t = delta_t

    def config_vo_inf(self, drone_state, drone_state_list, building_list, action=np.zeros((3,)), **kwargs):
        # mode: vo, rvo, hrvo
        odro_list, obs_building_list = self.preprocess(drone_state, drone_state_list, building_list)##获取建筑物障碍物与冲突无人机列表（十个）
        #drone_state=[self.state, self.vel, radius,priority, v_des, deviation]

        action = np.squeeze(action)

        info_with_budiling = list(map(lambda x:self.check_col_with_budilding(drone_state, x),obs_building_list))

        vo_list = list(map(lambda x: self.config_vo_circle2(drone_state, x, action, **kwargs), odro_list))
        #[observation_vo, vo_flag, exp_time, collision_flag, min_dis ] 
        obs_vo_list = []
        collision_flag = False
        vo_flag = False
        min_exp_time = inf
        min_dis = inf

        if len(info_with_budiling) > 0 and max(info_with_budiling):
            collision_flag = True

        for vo_inf in vo_list:
            if vo_inf[1] is True:# vo_flag[[x, y, z, ve_x, ve_y, ve_z, α, min_dis, input_exp_time], vo_flag, exp_time, collision_flag, min_dis]
                obs_vo_list.append(vo_inf[0])#[x, y, z, ve_x, ve_y, ve_z, α, min_dis, input_exp_time] 9维
                vo_flag = True#[x, y, z, rel_x, rel_y, rel_z, 0, vo_flag, 0, collision_flag, dis_mr]
                if vo_inf[2] < min_exp_time:#exp_time
                    min_exp_time = vo_inf[2]
            
            if vo_inf[3] is True: 
                collision_flag = True

        obs_vo_list.sort(reverse=True, key=lambda x: (-x[-1], x[-2]))#input_exp_time,min_dis对无人机进行排序，寻找威胁最大


        if len(obs_vo_list) > self.nm:###选十个
            obs_vo_list_nm = obs_vo_list[-self.nm:]
        else:
            obs_vo_list_nm = obs_vo_list

        if self.nm == 0:
            obs_vo_list_nm = []

        return obs_vo_list_nm, vo_flag, min_exp_time, collision_flag, obs_building_list#obs_vo_list, vo_flag, min_exp_time, collision_flag, obs_building_list
    
    def config_vo_reward(self, drone_state, other_drone_state_list,  building_list, action=np.zeros((2,)), **kwargs):#只检查是否与速度障碍物有冲突，不包括建筑物

        odro_list, obs_building_list= self.preprocess(drone_state, other_drone_state_list, building_list)
#[[x, y, z, ve_x, ve_y, ve_z, α, min_dis, input_exp_time], vo_flag, exp_time, collision_flag, min_dis]
        vo_list = list(map(lambda x: self.config_vo_circle2(drone_state, x, action, **kwargs), odro_list))
        vo_flag = False
        min_exp_time = inf
        min_dis = inf

        for vo_inf in vo_list:#[observation_vo, vo_flag, exp_time, collision_flag, min_dis]

            if vo_inf[4] < min_dis:
                min_dis = vo_inf[4]##两机距离

            if vo_inf[1] is True:
                vo_flag = True
                if vo_inf[2] < min_exp_time:##预计碰撞时间
                    min_exp_time = vo_inf[2]

        
        return vo_flag, min_exp_time, min_dis

    def preprocess(self, drone_state, dro_state_list, building_list):
        # 检测范围内障碍物
        odro_list=[]
        obs_building_list = []  
        self_drone_state = np.array(drone_state[0:3])
        for drone in  dro_state_list:
            other_drone_state = np.array(drone[0:3])
            if np.all(self_drone_state == other_drone_state):
                continue#排除自己
            dif = self_drone_state -other_drone_state
            dis = np.linalg.norm(dif)
            if dis <= 10:
                odro_list.append(drone)
#################################################################################################################################
        for building in building_list:
            building_state = np.array(building)
            if building_state[2] > self_drone_state[2] - 2:
                diff = self_drone_state[0:2] - building_state[0:2]
                diss = np.linalg.norm(diff)
                if diss <= 5:
                    obs_building_list.append(building)

        return odro_list, obs_building_list
        

    def config_vo_observe(self, drone_state, drone_state_list, obs_list):

        vo_list, _, _, _ = self.config_vo_inf(drone_state, drone_state_list, obs_list)
        
        return vo_list

    def config_vo_circle2(self, state, odro, action, **kwargs):#
        
        if np.linalg.norm(action) < 1e-5:
            action = np.zeros(3)

        x, y, z, vx, vy, vz, r = state[0:7]#state: [x, y, z, vx, vy, vz, radius, pra, vx_des, vy_des, vz_des, deviation]
        Pa = [x, y, z]
        Va = [vx, vy, vz]
        mx, my, mz,  mvx, mvy, mvz, mr = odro[0:7]#输入的状态中提取位置、速度和半径。moving_state_list: [[x, y, z, vx, vy, vz, radius, prb]]其他无人机
        Pb = [mx, my, mz]
        Vb = [mvx, mvy, mvz]

        vo_flag = False
        collision_flag = False

        rel_x = mx - x
        rel_y = my - y
        rel_z = mz - z

        dis_mr = sqrt((rel_y)**2 + (rel_x)**2 + (rel_z)**2)
        
        real_dis_mr = sqrt((rel_y)**2 + (rel_x)**2 + (rel_z)**2)
        
        env_train = kwargs.get('env_train', self.env_train)

        if env_train:#根据环境训练标志，检查是否发生碰撞。
            if dis_mr <= r + mr:
                dis_mr = r + mr
                collision_flag = True
        else:
            if dis_mr <= r - self.exp_radius + mr:
                collision_flag = True

            if dis_mr <= r + mr:
                dis_mr = r + mr

        if collision_flag == True:
            return [[x, y, z, rel_x, rel_y, rel_z, 0, 0, 0], vo_flag, 0, collision_flag, dis_mr]#[observation_vo, vo_flag, exp_time, collision_flag, min_dis]
        
        
        # 计算本机速度方向与相对向量的点积
        dot_product = vx * rel_x + vy * rel_y + vz * rel_z
        
        if dot_product <= 0:
            # 夹角 >= 90 度，本机速度背离对方，无需避障
            return [[x, y, z, rel_x, rel_y, rel_z, 0, -1, -1], vo_flag, 0, collision_flag, dis_mr]
        


        alpha = get_alpha(Pa, Pb, r, mr)
        PAA = get_PAA(Pa, state[7], odro[7], Va, Vb)#Pa, pra, prb, Va, Vb
        rvo_array = [rel_x, rel_y, rel_z]
        vo = PAA[:3]+rvo_array[:3]+[alpha]#vo: [x, y, z, ve_x, ve_y, ve_z, α]速度障碍物存储形式

 
        rel_vx = 2*action[0] - mvx - vx
        rel_vy = 2*action[1] - mvy - vy
        rel_vz = 2*action[2] - mvz - vz


        exp_time = inf

        if self.vo_out_jud_vector(state, action[0], action[1], action[2], vo):##检测此时无人机所选择速度是否在速度障碍物中agent_state, vx, vy, vz, odro_rvo
            vo_flag = False
            exp_time = inf
        else:
            exp_time = cal_vo_exp_tim(rel_x, rel_y, rel_z, rel_vx, rel_vy, rel_vz, r, mr) 
            if exp_time < self.ctime_threshold:
                vo_flag = True
            else:
                vo_flag = False
                exp_time = inf
            
        input_exp_time = 1 / (exp_time+0.2)
        min_dis = real_dis_mr-mr

        observation_vo = PAA[:3]+rvo_array[:3]+[alpha, min_dis, input_exp_time]#[x, y, z, ve_x, ve_y, ve_z, α]
        #速度障碍物的方向和速度。
        #计算期望时间，判断是否存在速度障碍物。
        #构建观测信息，包括速度障碍物的相关参数。
        return [observation_vo, vo_flag, exp_time, collision_flag, min_dis]#[[x, y, z, ve_x, ve_y, ve_z, α, min_dis, input_exp_time], vo_flag, exp_time, collision_flag, min_dis]

    def check_col_with_budilding(self, state, obs_building_list):
        x, y, z, vx, vy, vz, r = state[0:7]
        collision_flag = False
        
        for building in [obs_building_list]:
            if z <= building[2]:
                # 确保无人机在建筑物高度或以下  [xyhr]
                # 计算无人机与建筑物在地面的投影圆的距离  
                dis = self.distance2D((x,y), (building[0], building[1]))  
                if dis <= r + building[3]:  # 如果距离小于等于两圆半径之和，则碰撞  
                    collision_flag = True
        return collision_flag 


    def vo_out_jud_vector(self, agent_state, vx, vy, vz, odro_rvo):##不碰撞返回ture alpha < beta
        vector_out = True
        Panew = [0, 0, 0]
        Panew[0] = agent_state[0] + 2*vx*self.delta_t
        Panew[1] = agent_state[1] + 2*vy*self.delta_t
        Panew[2] = agent_state[2] + 2*vz*self.delta_t

        
        PAA = odro_rvo[0:3]
        rvo_array =  odro_rvo[3:6]
        alpha = odro_rvo[6]
        arr_AA_Anew = [Panew[i] - PAA[i] for i in range(len(PAA))]

        beta = get_beta(rvo_array, arr_AA_Anew)
        if alpha > beta:
            vector_out = False
        return vector_out
    @staticmethod
    def calculate_angle_between_vectors(vec1, vec2):  
    # 计算点积  
        dot_product = sum(a*b for a, b in zip(vec1, vec2))  
      
        # 计算两个向量的模长  
        norm_vec1 = sqrt(sum(x**2 for x in vec1))  
        norm_vec2 = sqrt(sum(x**2 for x in vec2))  
      
        # 防止除以零的错误  
        if norm_vec1 == 0 or norm_vec2 == 0:  
            return None  # 或者可以抛出一个异常  
      
        # 计算夹角（弧度）  
        angle_rad = acos(dot_product / (norm_vec1 * norm_vec2))

        return angle_rad  
    

    @staticmethod
    def cal_exp_tim_with_building(rel_x, rel_y, rel_vx, rel_vy, r):
        # rel_x: xa - xb
        # rel_y: ya - yb

        # (vx2 + vy2)*t2 + (2x*vx + 2*y*vy)*t+x2+y2-(r+mr)2 = 0

        a = rel_vx ** 2 + rel_vy ** 2
        b = 2* rel_x * rel_vx + 2* rel_y * rel_vy
        c = rel_x ** 2 + rel_y ** 2 - r ** 2

        if c <= 0:
            return 0

        temp = b ** 2 - 4 * a * c

        if temp <= 0:
            t = inf
        else:
            t1 = ( -b + sqrt(temp) ) / (2 * a)
            t2 = ( -b - sqrt(temp) ) / (2 * a)

            t3 = t1 if t1 >= 0 else inf
            t4 = t2 if t2 >= 0 else inf
        
            t = min(t3, t4)

        return t
    @staticmethod
    def distance2D(point1, point2):
        distance = sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)
        return distance
                 