
import numpy as np
from math import sin, cos, atan2, pi, sqrt
from uaisa_env.world.line_sight_partial_3D import line_sight_partial_3D
from collections import namedtuple
# state: [x, y, z, vx, vy, vz, radius, pra, vx_des, vy_des, vz_des]
# moving_state_list: [[x, y, z, vx, vy, vz, radius, prb]]
# obstacle_state_list: [[x, y, z, radius]]
# rvo_vel: [x, y, z, ve_x, ve_y, ve_z, α]
import numpy as np 
# from uaisa_env.drone_envs.kinematic import kinematic
  
class drone():  
    def __init__(self, id, starting, destination, waypoints, n_points, init_acc,priority = 5, dt=1,    
                 vel=np.zeros((3,)), vel_max = 1*np.ones((3,)), goal_threshold=0.4, radius=0.2, **kwargs):  
  #id=i, starting = starting_list[i], destination = destination_list[i], waypoints=waypoints_list[i], 
   #                                   n_points = n_points_list[i], init_acc = 1, step_time=step_time, **kwargs)

        self.id = int(id)  # 
  
        self.vel = np.array(vel) if not isinstance(vel, np.ndarray) else vel 
        self.vel_max = np.array(vel_max) if not isinstance(vel_max, np.ndarray) else vel_max  
  
        # 转换starting和destination为NumPy数组  
        self.starting = np.array(starting)if isinstance(starting, list) else starting 
        self.destination = np.array(destination)if isinstance(destination, list) else destination

        self.state =  self.starting

        ##航路长度、初始化行程长度
        self.route_len = self.calculate_total_length(waypoints)
        self.real_route_len = 0
        self.extra_len = 0
        self.route_point = [[self.starting]]

        #初始化偏移量
        self.max_deviation = 0

        # waypoints作为列表传入，无需转换为NumPy数组  
        self.waypoints = waypoints  
        self.n_points = n_points  # 航路点个数

        self.priority = priority  
  
        # 当前目标点，初始化为destination或者waypoints的第二个点 （waypoints）
        if len(waypoints) > 0:  
            self.current_des = np.array(waypoints[1])
            self.previous_des = np.array(waypoints[0])
        else:  
            self.current_des = np.array(destination)if isinstance(destination, list) else destination
            self.previous_des = np.array(waypoints[0])

        self.i = 1  
        self.previous_state = self.state.copy()  
        self.acc = init_acc  # 加速度  
        self.dt = dt  # 时间步长  
        self.radius = radius  # 半径  
        self.rvo_vel = [0,0,0]##专门用来画图的速度向量
  
        self.arrive_flag = False  # 到达标志 
        self.destination_arrive_flag = False# 
        self.collision_flag = False  # 碰撞标志
        self.see_des_flag = True  
        self.goal_threshold = goal_threshold  # 到达目标的阈值

        # 初始状态
        self.velocity = 0.0                        # 合速度标量 (m/s)
        self.yaw = 0.0                             # 航向角 (deg)
        self.pitch = 0.0                           # 俯仰角 (deg)
        
        # 物理约束
        self.max_acc = 1.0                        # 最大加速度 (m/s²)
        self.max_angle_change = 90.0               # 最大角度变化 (deg)

        #self.components = components 
        self.radius_collision = radius # 碰撞检测半径
        
        # 添加noise参数，如果kwargs中没有提供，则默认为False  
        self.__noise = kwargs.get('noise', False)  
  
        # 添加control_std参数，如果kwargs中没有提供，则使用默认值  
        self.__control_std = kwargs.get('control_std', [0.06, 0.06, 0.06])  

  

    def update_info(self):
        ##更新行程长度
        difference = self.state - self.previous_state
        diff_len = np.linalg.norm(difference )
        self.real_route_len = self.real_route_len + diff_len
        
    # # 更新状态
    #     self.state = state
    #     self.vel = vel

    def move_forward(self, act,  E3d, map_size, stop=False, **kwargs): 
    
        if isinstance(act, list):
            act = np.array(vel)

        assert act.shape == (3,)

        self.velocity = np.linalg.norm(self.vel)
        
        vel = self.kinematicstep(act, self.velocity)

        if stop:
            if self.destination_arrive_flag or self.collision_flag:
                vel = np.zeros(3,)

        self.previous_state = self.state
        self.move(vel, self.__noise, self.__control_std)
        self.update_info()
        self.vel = self.vel
        # print(self.state)
        if self.arrive(self.state, self.current_des) == True and self.destination_arrive(self.state) == False:
            if self.i < self.n_points-1:
                self.current_des_new(E3d, map_size)
            


    def current_des_new(self, E3d, map_size):  
        if self.i < self.n_points - 1:  # 减1是因为当前已经到达了第i个点  
            next_waypoint = self.waypoints[self.i + 1]  
            #if line_sight_partial_3D(E3d, (self.state[0], next_waypoint[0]),  (self.state[1], next_waypoint[1]),  (self.state[2], next_waypoint[2]),  map_size) == 1:  
            self.i += 1
            self.previous_des = self.current_des
            self.current_des = np.array(next_waypoint)
            self.arrive_flag = False  
             

    def if_see_des(self,E3d, map_size):##没想好怎么惩罚
        if line_sight_partial_3D(E3d, (self.state[0], self.current_des[0]),  (self.state[1], self.current_des[1]),  (self.state[2], self.current_des[2]),  map_size) == 0:
            self.see_des_flag = False
        else:
            self.see_des_flag = True


    def change_current_des(self, E3d, map_size):###加到move_forward
        if self.arrive(self.state, self.current_des):
            self.current_des_new(E3d, map_size)

    

    def move(self, vel, noise=False, std=None):  
        if std is None:  
            std = self.__control_std
        #print(self.state)
        next_state = self.motion(vel, noise, std)  
        self.state = next_state  
        self.vel = vel 


    def motion(self, vel, noise = False, control_std = None):
        current_state = self.state
        sampletime = self.dt
        if control_std is None:  
            control_std = [0.06, 0.06, 0.06]
    # vel: np.array([[vel x], [vel y],[vel z]])
    # current_state: np.array([[x], [y], [z]])

        if noise:  
            vel_noise = np.round(vel + np.random.normal(np.zeros((3,)), scale=np.array(control_std)),2) 
        else:  
            vel_noise = vel
        #print("motion_vel=",vel_noise)
        next_state = current_state + vel_noise * sampletime
        #print("nextstate",next_state)
        self.route_point.append(next_state)   
        return next_state

    def arrive(self, current_position, current_des):

        position = current_position[0:3]
        dist = np.linalg.norm(position - current_des[0:3]) 
        if dist <= self.goal_threshold:
            return True
        else:
            return False
        
        
    def destination_arrive(self, current_position):
        position = current_position[0:3]
        dist = np.linalg.norm(position - self.destination[0:3]) 

        if dist <= self.goal_threshold:
            #self.destination_arrive_flag = True
            self.extra_len = self.real_route_len - self.route_len
            return True
        else:
            #self.destination_arrive_flag = False
            return False
    
    def move_to_goal(self):
        vel = self.cal_des_vel()
        self.move_forward(vel) 

    
    def cal_des_vel(self):  
        dis, angles = self.relative(self.state[0:3], self.current_des)  
          
        if dis > self.goal_threshold:  
            dir_vector = self.angles_to_direction(angles)
              
            # 缩放单位方向向量到最大速度
            vel_scaled = np.multiply(self.vel_max, dir_vector)  
            vel = np.round(vel_scaled, 3)
        else:    
            vel = np.zeros(3,)
        return vel
    
    #检查无人机是否在地图内
    def drone_out_map(self, map_size):
        dronex = self.state[0]
        droney = self.state[1]
        dronez = self.state[2]
        x = map_size[0]
        y = map_size[1]
        z = map_size[2]
        if dronex < 0 or dronex > x:  
            return True
        if droney < 0 or droney > y:  
            return True
        if dronez < 0 or dronez > z:  
            return True

    def collision_check_with_dro(self, components):  
    # 检查与其他无人机的碰撞
        circle = namedtuple('circle', 'x y z r')
        self_circle = circle(self.state[0], self.state[1], self.state[2], self.radius)
        if self.collision_flag == True:
            return True  
        for other_drone in  components['drone'].drone_list:
            if other_drone is not self and not other_drone.collision_flag: 
                 other_circle = circle(other_drone.state[0], other_drone.state[1], other_drone.state[2], other_drone.radius) 
                 if self.collision_dro_dro(self_circle, other_circle):  
                     other_drone.collision_flag = True  
                     self.collision_flag = True  
                     print('Drones collided!')  
                 return True

    def collision_check_with_building(self, building_list):
        for building_obj in building_list:  
            if self.state[2] <= building_obj[2]:
                 # 确保无人机在建筑物高度或以下  [xyhr]
                # 计算无人机与建筑物在地面的投影圆的距离  
                dis = self.distance2D(self.state, (building_obj[0], building_obj[1]))  
                if dis <= self.radius + building_obj[3]:  # 如果距离小于等于两圆半径之和，则碰撞  
                    self.collision_flag = True  
                    print('Drone collided with a building!')  
                    return True    
   

    def dronestate(self):
        v_des = self.cal_des_vel()
        rc_array = np.array([self.radius_collision])
        priority = np.array([self.priority])
        # state: [x, y, z, vx, vy, vz, radius, pra, vx_des, vy_des, vz_des]
        #return np.concatenate((self.state, self.vel, rc_array,priority, v_des), axis = 0)
        deviation = np.array([self.Deviation_from_route()])
        if deviation > self.max_deviation:
            self.max_deviation  = deviation
        return np.concatenate((self.state, self.vel, [rc_array[0]],[priority[0]], v_des, [deviation[0]]))

    def obs_state(self):
        rc_array = self.radius * np.ones((1,))
        return np.concatenate((self.state[0:3], self.vel, rc_array)) 


    def reset(self):

        self.state = self.starting
        self.previous_state = self.starting
        self.i = 1
        self.vel = np.zeros((3,))
        self.arrive_flag = False
        self.see_des_flag = True
        self.destination_arrive_flag = False# 
        self.collision_flag = False  # 碰撞标志
        self.real_route_len = 0
        self.max_deviation = 0
        self.route_point = [[self.starting]]
        self.velocity = 0.0                        
        self.yaw = 0.0                             
        self.pitch = 0.0                           
        if len(self.waypoints) > 0:  
            self.current_des = np.array(self.waypoints[1])
            self.previous_des = np.array(self.waypoints[0])
        else:  
            self.current_des = np.array(self.destination)if isinstance(self.destination, list) else self.destination
            self.previous_des = np.array(self.waypoints[0])

    def Deviation_from_route(self):

        deviation = self.calculate_deviation(self.previous_des, self.current_des, self.state)
##与航线偏差
        return deviation 



    # 检查两个无人机是否碰撞  
    @staticmethod  
    def collision_dro_dro(circle1, circle2):  
        dis = drone.distance(circle1, circle2)   
        if 0 < dis <= circle1.r + circle2.r:  
            return True  
        return False
    

    # 检查无人机是否与建筑物的圆形投影碰撞  
    @staticmethod  
    def collision_dro_building(drone_circle, building_projection):  
        dis = drone.distance2D(drone_circle, building_projection)  # 假设这个函数可以计算无人机与建筑物水平投影之间的距离  
        if 0 < dis <= drone_circle.r + building_projection.r:  
            return True  
        return False
    

    @staticmethod  
    def angles_to_direction(angles):  
        azimuth, elevation = angles  
        # 将方位角和俯仰角转换为单位方向向量  
        dir_vector = np.array([  
            np.cos(azimuth) * np.cos(elevation),  
            np.sin(azimuth) * np.cos(elevation),  
            np.sin(elevation)  
        ])  
        return dir_vector
    
    @staticmethod
    def distance(point1, point2):
        distance = sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2+(point1[2]-point2[2])**2)
        return distance
    
    @staticmethod
    def distance2D(point1, point2):
        distance = sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)
        return distance
                 
    @staticmethod  
    def relative(state1, state2):  
        dif = np.array(state2[0:3]) - np.array(state1[0:3])  
        dis = np.linalg.norm(dif)  # 计算向量长度（范数）  
        azimuth = np.arctan2(dif[1], dif[0])  # 方位角（绕z轴旋转）  
        elevation = np.arctan2(dif[2], np.linalg.norm(dif[0:2]))  # 俯仰角（绕y轴旋转）
        
        if dis != 0:  # 防止除以零  
            elevation = np.arctan2(dif[2], np.linalg.norm(dif[0:2]))  # 俯仰角（绕y轴旋转）  
        else:  
            elevation = 0  # 如果距离为零，俯仰角可以定义为零（或者任意值，因为它不影响方向）  
    
        return dis, (azimuth, elevation) ##输出是弧度制
    
    

    @staticmethod
    def to_pi(radian):

        if radian > pi:
            radian = radian - 2 * pi
        elif radian < -pi:
            radian = radian + 2 * pi
        
        return radian
    
    @staticmethod
    def calculate_deviation(start, end, drone_position):  
        # 起点、终点和无人机当前位置  
        x1, y1, z1 = start  
        x2, y2, z2 = end  
        x0, y0, z0 = drone_position  
  
        # 计算方向向量  
        dx = x2 - x1  
        dy = y2 - y1  
        dz = z2 - z1  
  
        # 计算方向向量的模（长度）  
        d_magnitude = sqrt(dx**2 + dy**2 + dz**2)  
  
        # 防止除数为0（当起点和终点相同时）  
        if d_magnitude == 0:  
            return 0  # 起点和终点相同，无人机不可能偏离航线  
  
        # 归一化方向向量  
        dx_hat = dx / d_magnitude  
        dy_hat = dy / d_magnitude  
        dz_hat = dz / d_magnitude  
  
        # 计算无人机位置到起点的向量  
        px = x0 - x1  
        py = y0 - y1  
        pz = z0 - z1  
  
        # 计算投影长度  
        t = px * dx_hat + py * dy_hat + pz * dz_hat  
  
        # 使用投影长度找到航线上离无人机最近的点  
        qx = x1 + t * dx_hat  
        qy = y1 + t * dy_hat  
        qz = z1 + t * dz_hat  
  
        # 计算无人机位置到最近点的距离（偏离程度）  
        deviation = sqrt((x0 - qx)**2 + (y0 - qy)**2 + (z0 - qz)**2) 
  
        return deviation 
    
    
    @staticmethod
    def calculate_total_length(waypoints):  
        total_length = 0.0  
        num_points = len(waypoints)  
      
        for i in range(num_points - 1):  
            # 当前点和下一个点的坐标  
            current_point = waypoints[i]  
            next_point = waypoints[i + 1]  
          
            # 计算两点之间的欧几里得距离  
            dx = next_point[0] - current_point[0]  
            dy = next_point[1] - current_point[1]  
            dz = next_point[2] - current_point[2]  
          
            distance = sqrt(dx**2 + dy**2 + dz**2)  
          
            # 累加距离  
            total_length += distance  
      
        return total_length  
    
    def _clamp(self, value, min_val, max_val):
        """数值约束函数"""
        return np.clip(value, min_val, max_val)

    def _action_to_control(self, action):
        """将[-1,1]的动作映射到物理量"""
        # 动作归一化处理
        acc_delta = action[0] * self.max_acc        # 加速度变化
        yaw_delta = action[1] * self.max_angle_change  # 航向角变化
        pitch_delta = action[2] * self.max_angle_change  # 俯仰角变化
        
        # 应用约束
        acc_delta = self._clamp(acc_delta, -self.max_acc, self.max_acc)
        yaw_delta = self._clamp(yaw_delta, -self.max_angle_change, self.max_angle_change)
        pitch_delta = self._clamp(pitch_delta, -self.max_angle_change, self.max_angle_change)
        
        return acc_delta, yaw_delta, pitch_delta

    def _update_velocity(self, acc_delta, velocity):
        """更新速度（含加速度约束）"""
        new_velocity = velocity + acc_delta * 1
        return max(new_velocity, 0)  # 速度不能为负

    def _update_angles(self, yaw_delta, pitch_delta):
        """更新角度（含周期约束）"""
        # 航向角保持0-360度范围
        new_yaw = (self.yaw + yaw_delta) % 360  
        # 俯仰角限制在-90~90度范围
        new_pitch = self._clamp(self.pitch + pitch_delta, -90, 90)
        return new_yaw, new_pitch

    def _get_velocity_components(self):
        """将合速度分解为三轴分量"""
        yaw_rad = np.deg2rad(self.yaw)
        pitch_rad = np.deg2rad(self.pitch)
        
        vx = self.velocity * np.cos(pitch_rad) * np.cos(yaw_rad)
        vy = self.velocity * np.cos(pitch_rad) * np.sin(yaw_rad)
        vz = self.velocity * np.sin(pitch_rad)
        return np.array([vx, vy, vz])

    def kinematicstep(self, action, velocity):
        """
        执行单步运动计算
        :param action: 三维动作输入 [-1,1]^3
        :param delta_t: 时间步长 (秒)
        :return: 新位置 [x, y, z]
        """
        # 1. 动作转换为物理量
        acc_delta, yaw_delta, pitch_delta = self._action_to_control(action)
        
        # 2. 更新速度和角度
        self.velocity = self._update_velocity(acc_delta, velocity)

        self.yaw, self.pitch = self._update_angles(yaw_delta, pitch_delta)
        
        # 3. 计算位移
        velocity_components = self._get_velocity_components()

        return velocity_components
