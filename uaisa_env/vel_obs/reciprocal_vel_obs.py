
import numpy as np
from math import sin, cos, atan2, asin, pi, inf, sqrt
from time import time
# from uaisa.uaisa_env.vel_obs.vel_obs3D import get_alpha, get_PAA,  get_rvo_array, get_beta, cal_exp_tim 

from uaisa_env.vel_obs.vel_obs3D import get_alpha, get_PAA,  get_rvo_array, get_beta, cal_exp_tim 


class reciprocal_vel_obs:

    def __init__(self, neighbor_region=10, vxmax = 2, vymax = 2, vzmax = 2, acceler = 0.5, delta_t = 1):

        self.vxmax = vxmax
        self.vymax = vymax
        self.vzmax = vzmax
        self.acceler = acceler
        self.nr = neighbor_region
        self.delta_t = delta_t

    def cal_vel(self, agent_state, dro_state_list,  building_list, mode = 'rvo'):
        
        odro_list, obs_list= self.preprocess(agent_state, dro_state_list, building_list)##获取建筑物障碍物与冲突无人机列表


        vo_list = self.config_vo(agent_state, odro_list)##分别检测无人机互易速度障碍物

        
        vo_outside, vo_inside = self.vel_candidate(agent_state, vo_list)##速度边界
        rvo_vel = self.vel_select(agent_state, vo_outside, vo_inside, vo_list)##选择速度
        return rvo_vel

    def preprocess(self, agent_state, dro_state_list, building_list):
        # 检测范围内障碍物
        #待填充 
        odro_list=[]
        obs_list = []  
        agent_state = np.array(agent_state[0:3])
        for drone in  dro_state_list:
            drone_state = np.array(drone[0:3])
            dif = agent_state-drone_state
            dis = np.linalg.norm(dif)
            if dis <= 10:
                odro_list.append(drone)

        for building in building_list:
            building_state = np.array(building)
            if building_state[2] > agent_state[2] - 1:
                diff = agent_state[0:2] - building_state[0:2]
                diss = np.linalg.norm(diff)
                if diss <= 10:
                    obs_list.append(building)

        return odro_list, obs_list

    def config_vo(self, agent_state, odro_list, mode):
        # mode: vo, rvo, hrvo

        vo_list = list(map(lambda x: self.config_vo(agent_state, x, mode), odro_list))
       
        return vo_list

    def config_vo(self, state, odro, mode='vo'):##检测圆形、环形障碍物速度障碍模型
        
        Pa = state[0:4]
        Va = state[3:6]
        ra = state[6]
        pra = state[7]

        Pb = odro[0:4]
        Vb = odro[3:6]
        rb = odro[6]
        prb = odro[7]

        alpha = get_alpha(Pa, Pb, ra, rb)
        PAA = get_PAA(Pa, pra, prb, Va, Vb)
        rvo_array = get_rvo_array(Pa, Pb)
        exp_tim = cal_exp_tim(Pa, Pb, Va, Vb, ra, rb)
        odro_rvo=PAA[:3]+rvo_array[:3]+[alpha,exp_tim]

        
        return odro_rvo#返回值  位置+向量

    
    def vel_candidate(self, agent_state, vo_list):##速度候选
        
        vo_outside, vo_inside = [], []
        
        cur_vx, cur_vy, cur_vz = agent_state[3:6]##现有速度
        cur_vx_range = np.clip([cur_vx-self.acceler, cur_vx+self.acceler], -self.vxmax, self.vxmax)##加速度限制速度范围
        cur_vy_range = np.clip([cur_vy-self.acceler, cur_vy+self.acceler], -self.vymax, self.vymax)
        cur_vz_range = np.clip([cur_vz-self.acceler, cur_vz+self.acceler], -self.vzmax, self.vzmax)
        
        for new_vx in np.arange(cur_vx_range[0], cur_vx_range[1], 0.5):#使用两个嵌套的循环，遍历所有在速度范围内的新速度候选项 new_vx 和 new_vy
            for new_vy in np.arange(cur_vy_range[0], cur_vy_range[1], 0.5):
                for new_vz in np.arange(cur_vz_range[0], cur_vz_range[1], 0.5):
                
                    if sqrt(new_vx**2 + new_vy**2 + new_vz**2) < 0.3:##
                        continue

                    if self.vo_out2(agent_state, new_vx, new_vy, new_vz, vo_list):
                        vo_inside.append([new_vx, new_vy, new_vz])
                    else:
                        vo_outside.append([new_vx, new_vy, new_vz])
                    #每个速度候选项，我们检查它是否满足以下条件：速度的模长（即速度的大小）大于 0.3。通过调用 self.vo_out2(new_vx, new_vy, vo_list) 来判断是否与避障障碍物发生碰撞

        return vo_outside, vo_inside

    def vo_out2(self, agent_state, vx, vy, vz, vo_list):
        col = False
        Panew = [0, 0, 0]
        Panew[0] = agent_state[0] + vx*self.delta_t
        Panew[1] = agent_state[1] + vy*self.delta_t
        Panew[2] = agent_state[2] + vz*self.delta_t

        for odro_rvo in vo_list:##判断给定速度  是否与一组障碍物（vo_list）发生碰撞（向量法）
            PAA = odro_rvo[0:3]
            rvo_array =  odro_rvo[4:6]
            alpha = odro_rvo[6]
            arr_AA_Anew = []
            for i in range(3):
                arr_AA_Anew.append = Panew[i] - PAA[i]

            beta = get_beta(rvo_array, arr_AA_Anew)
            if alpha > beta:
                col = True

        return col

    
    def vel_select(self, agent_state, vo_outside, vo_inside, odro_list):#择机器人的速度state: [x, y, z, vx, vy, vz, radius, pra, vx_des, vy_des, vz_des]

        vel_des = [agent_state[8], agent_state[9],agent_state[10]]#期望速度 vel_des
        if (len(vo_outside) != 0):
            temp= min(vo_outside, key = lambda v: reciprocal_vel_obs.distance(v, vel_des))#vo_outside 列表不为空，它会选择其中与期望速度最接近的速度（使用 reciprocal_vel_obs.distance 计算距离
        else:
            temp = min(vo_inside, key = lambda v: self.penalty(v, vel_des, agent_state, odro_list, 1) )#否则，选择 vo_inside 列表中的速度，但是会惩罚（self.penalty
            return temp
            
    def penalty(self, vel, vel_des, agent_state, odro_list, factor):##惩罚值
        tc_list = []

        for odro in odro_list:##移动物体列表

            Pa = agent_state[0:4]
            Va = agent_state[3:6]
            ra = agent_state[6]
        
            Pb = odro[0:4]
            Vb = odro[3:6]
            rb = odro[6]


            tc = cal_exp_tim(Pa, Pb, Va, Vb, ra, rb)##self.cal_exp_tim 计算了时间冲突（tc
            tc_list.append(tc)

        tc_min = min(tc_list)

        if tc_min == 0:
            tc_inv = float('inf')
        else:
            tc_inv = 1/tc_min

        penalty_vel = factor * tc_inv + reciprocal_vel_obs.distance(vel_des, vel)

        return penalty_vel
    

    @staticmethod
    def distance(point1, point2):
        return sqrt( (point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2 + (point2[2] - point1[2]) ** 2 )

 
    @staticmethod
    def wraptopi(theta):

        if theta > pi:
            theta = theta - 2*pi
        
        if theta < -pi:
            theta = theta + 2*pi

        return theta

    @staticmethod
    def clamp(n, minn, maxn):
        return max(min(maxn, n), minn)    



