
from uaisa_env.drone_envs.drone import drone
# from uaisa.uaisa_env.drone_envs.drone import drone
from math import pi, cos, sin ,atan2, pi, sqrt
import numpy as np
import random  
from collections import namedtuple
# state: [x, y, z, vx, vy, vz, radius, pra, vx_des, vy_des, vz_des]
# moving_state_list: [[x, y, z, vx, vy, vz, radius, prb]]
# obstacle_state_list: [[x, y, z, radius]]
# rvo_vel: [x, y, z, ve_x, ve_y, ve_z, α]
class env_drone:##
    def __init__(self, waypoints_list, n_points_list, priority_list,  building_list, drone_class=drone, drone_number = 0, step_time=1, **kwargs):

        self.drone_class = drone_class#分类
        self.drone_number = drone_number#数量
        self.drone_list = []

        self.interval = kwargs.get('interval', 1)#
        self.radius = kwargs.get('radius', False)

        self.building_list = building_list
    
        starting_list, destination_list= self.init_state_distribute(waypoints_list)


        # 创建无人机对象
        for i in range(self.drone_number):
            drone = self.drone_class(id=i, starting = starting_list[i], destination = destination_list[i], waypoints=waypoints_list[i], 
                                      n_points = n_points_list[i], init_acc = 1, step_time=step_time, **kwargs)
            self.drone_list.append(drone)
            self.drone = drone if i == 0 else None 
        
    def init_state_distribute(self, waypoints_list):#将起点与终点从航路点中提取出来
        starting_list, destination_list = [], []
        for waypoints in waypoints_list:  
            start = waypoints[0]  
            goal = waypoints[-1]  
            starting_list.append(start)  
            destination_list.append(goal)  
        return starting_list, destination_list
    
   
    def distance(self, point1, point2):
        diff = point2[0:3] - point1[0:3]
        return np.linalg.norm(diff)


    def collision_check_with_building(self):
        self.collision_flag = False
        for drone in self.drone_list:
            for building_obj in self.building_list:  
                if drone.state[2] <= building_obj[2]:
                    dis = self.distance2D(drone.state, (building_obj[0], building_obj[1]))  
                    if dis <= drone.radius + building_obj[3]:
                        self.collision_flag = True  
                        print('Drone collided with a building!')  
                        break # 如果发生碰撞，跳出内部循环
            if self.collision_flag:  
                break  # 如果发生碰撞，跳出外部循环  
        return self.collision_flag



    def collision_check_with_drones(self):
        circle = namedtuple('circle', 'x y z r')
        self.collision_flag = False  
        for i, drone in enumerate(self.drone_list):  
            if drone.collision_flag == True:
                return True
            self_circle = circle(drone.state[0], drone.state[1], drone.state[2], drone.radius)
            for other_drone in self.drone_list[i+1:]:
                if other_drone is not drone and not other_drone.collision_flag: 
                    other_circle = circle(other_drone.state[0], other_drone.state[1], other_drone.state[2], other_drone.radius) 
                    if self.collision_dro_dro(self_circle, other_circle):  
                        other_drone.collision_flag = True  
                        self.collision_flag = True  
                        print('Drones collided!')  
                        return True 


    def step(self, vel_list=[], **vel_kwargs):

        for drone, vel in zip(self.drone_list, vel_list):
            drone.move_forward(vel, **vel_kwargs)

    def cal_des_list(self):
        vel_list = list(map(lambda x: x.cal_des_vel() , self.drone_list))
        return vel_list
    
    def arrive_all(self):

        for drone in self.drone_list:
            if not drone.destination_arrive():
                return False

        return True

    def drones_reset(self):
        for drone in self.drone_list:
            drone.reset()

    def drone_reset(self, id=0):
        self.drone_list[id].reset()

    def total_states(self):
        drone_state_list = list(map(lambda r: np.squeeze( r.dronestate()), self.drone_list))# state: [x, y, z, vx, vy, vz, radius, pra, vx_des, vy_des, vz_des]
        return drone_state_list
    
    def deviation_indicator(self):#计算无人机列表中最大偏差和额外长度的平均值
        deviation = []
        for drone in self.drone_list:
            deviation.append(drone.max_deviation)
        return deviation
    
    def extar_len_indicator(self):#计算无人机列表中最大偏差和额外长度的平均值
        extar_len = []
        for drone in self.drone_list:
            extar_len.append(drone.extra_len)

        return extar_len

    

    @staticmethod  
    def collision_dro_dro(circle1, circle2):  
        dis = drone.distance(circle1, circle2)   
        if 0 < dis <= circle1.r + circle2.r:  
            return True  
        return False
    
    @staticmethod
    def distance(point1, point2):
        distance = sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2+(point1[2]-point2[2])**2)
        return distance
    
    @staticmethod
    def distance2D(point1, point2):
        distance = sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)
        return distance
    # # states
    # def total_states(self, env_train=True):
        
    #     robot_state_list = list(map(lambda r: np.squeeze( r.omni_state(env_train)), self.robot_list))
    #     nei_state_list = list(map(lambda r: np.squeeze( r.omni_obs_state(env_train)), self.robot_list))
    #     obs_circular_list = list(map(lambda o: np.squeeze( o.omni_obs_state(env_train) ), self.obs_cir_list))
    #     obs_line_list = self.obs_line_list
        
    #     return [robot_state_list, nei_state_list, obs_circular_list, obs_line_list]
        
    # def render(self, time=0.1, save=False, path=None, i = 0, **kwargs):
        
    #     self.world_plot.draw_robot_diff_list(**kwargs)
    #     self.world_plot.draw_obs_cir_list()
    #     self.world_plot.pause(time)

    #     if save == True:
    #         self.world_plot.save_gif_figure(path, i)

    #     self.world_plot.com_cla()

    
    # def seg_dis(self, segment, point):
        
    #     point = np.squeeze(point[0:2])
    #     sp = np.array(segment[0:2])
    #     ep = np.array(segment[2:4])

    #     l2 = (ep - sp) @ (ep - sp)

    #     if (l2 == 0.0):
    #         return np.linalg.norm(point - sp)

    #     t = max(0, min(1, ((point-sp) @ (ep-sp)) / l2 ))

    #     projection = sp + t * (ep-sp)

    #     distance = np.linalg.norm(point - projection) 

    #     return distance