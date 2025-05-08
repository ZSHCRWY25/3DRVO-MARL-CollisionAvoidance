
import os
import yaml
import numpy as np
import json  
import sys
import matplotlib.pyplot as plt
import math
from uaisa_env.drone_envs.env_plot import env_plot
from uaisa_env.drone_envs.env_drones import env_drone
from uaisa_env.drone_envs.drone import drone

class env_base:

    def __init__(self, base_dir='E:\\01My_sudy\\01_ing\\uaisa\\uaisa_env\\world\\world_3'):  ##换环境改这里
        self.base_dir = base_dir  
        self.load_data()  
  
        self.components = dict()
        self.priority_list = []
        self.plot = True##和render一块改

  
        self.init_environment(drone_class=drone)

    def load_data(self):  
        # 拼接 JSON 文件的完整路径  
        json_path = os.path.join(self.base_dir, 'data_1.json')  
          
        # 读取 JSON 文件  
        with open(json_path, 'r') as file:  
            data = json.load(file)  
          
        # 提取所需的数据  
        self.waypoints_list = data.get('waypoints_list', [])  
        self.n_points_list = data.get('n_points_list', [])  
        self.building_list = data.get('building_list', [])
        self.map_size = data.get('map_size',[])
        self.drone_num = data.get('drone_num',[])
          
        # 拼接 NumPy 文件的完整路径  
        npy_path_E3d = os.path.join(self.base_dir, 'E3d.npy')  
        npy_path_E3d_safe = os.path.join(self.base_dir, 'E3d_safe.npy')  
          
        # 读取 NumPy 文件  
        self.E3d = np.load(npy_path_E3d)  
        self.E3d_safe = np.load(npy_path_E3d_safe)  
   

    def init_map(self):

        x_range, y_range, z_range = self.map_size
        
        fig = plt.figure(figsize=(8, 6))  
        ax = fig.add_subplot(111, projection='3d')  
      
        # 设置地图的x和y轴范围  
        ax.set_xlim(0, x_range)  
        ax.set_ylim(0, y_range)  
        ax.set_zlim(0, z_range + 3)  # z轴范围基于建筑物最高度+一点额外空间  
      
    # 绘制每个建筑物（圆柱体）  
        for b in self.building_list:  
            x, y, h, r = b  
          
            # 生成极坐标和高度  
            u = np.linspace(0, 2 * np.pi, 50)  
            h_vals = np.linspace(0, h, 20)  # 足够的高度切片以形成平滑的圆柱体  
          
            # 使用meshgrid生成二维网格上的X, Y, Z  
            U, H = np.meshgrid(u, h_vals)  
            X = x + r * np.sin(U)   
            Y = y + r * np.cos(U)    
            Z = H  
          

            plt.gca().set_prop_cycle(None)   
            ax.plot_surface(X, Y, Z, linewidth=0, facecolor='b', shade=True, alpha=0.6)

        for i in range(self.drone_num):
            starting = self.waypoints_list[i][0]
            destination = self.waypoints_list[i][-1]
            path = np.array(self.waypoints_list[i][1:-1])
            if len(path) > 0:
                ax.plot(path[:, 0], path[:, 1], path[:, 2], 'kx-')
            ax.plot(starting[1], starting[0], starting[2], 'go')
            ax.plot(destination[1], destination[0], destination[2], 'ro')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')


    def init_environment(self, drone_class=drone):
        ##这里添加：读取地图、障碍物、无人机及航路点列表

        self.components['drones'] = env_drone(waypoints_list = self.waypoints_list,n_points_list = self.n_points_list,priority_list = self.priority_list,
                                              building_list = self.building_list, drone_class=drone_class, drone_number=self.drone_num, step_time=1, 
                                              )
        self.drone_list = self.components['drones'].drone_list
        #assert 'waypoints_list' in kwargs and 'n_points_list' in kwargs and 'priority_list' in kwargs, "Missing required keyword arguments"

        self.time = 0
        if self.drone_num> 0:
            self.drone = self.components['drones'].drone_list[0]
        
        if self.plot:
            self.world_plot = env_plot(self.map_size, self.building_list, self.components)
        
        self.time = 0
        



    def collision_check(self):##检查碰撞
        collision = False
        for drone in self.components['drones'].drone_list: 
            if drone.collision_check_with_dro(self.components):
                collision = True
            if drone.collision_check_with_building(self.building_list):
                collision = True
            if drone.drone_out_map(self.map_size):
                collision = True
        return collision
    
    def arrive_check(self):##检查到没到
        arrive = False

        for drone in self.components['drones'].drone_list: 
            if not drone.arrive_flag:
                arrive = True

        return arrive

    def drone_step(self, acc_list,vo_flag_list, drone_id = None, **kwargs):

        if drone_id == None:
            if not isinstance(acc_list, list):
                self.drone.move_forward(acc_list, vo_flag_list, self.E3d,self.map_size, **kwargs)
            else:
                for i, drone in enumerate(self.components['drones'].drone_list):
                   drone.move_forward(acc_list[i], vo_flag_list, self.E3d,self.map_size, **kwargs)
        else:
            self.components['drones'].drone_list[drone_id-1].move_forward(acc_list, vo_flag_list, **kwargs)

    def see_des(self, drone_id = None):
        if drone_id == None:
            for drone in enumerate(self.components['drones'].drone_list):
                drone.if_see_des(self.E3d, self.map_size)
            else:
                self.components['drones'].drone_list[drone_id-1].if_see_des(self.E3d, self.map_size)



    def render(self, time=0.05, **kwargs):

        if self.plot:
            self.world_plot.clear_plot_elements()
            self.world_plot.draw_drones(**kwargs)
            self.world_plot.pause(time)
            
        self.time = self.time + time
        
    def save_fig(self, path, i):
        self.world_plot.save_gif_figure(path, i)
    
    def save_ani(self, image_path, ani_path, ani_name='animated', **kwargs):
        self.world_plot.create_animate(image_path, ani_path, ani_name=ani_name, **kwargs)

    def show(self, **kwargs):
        self.world_plot.draw_drones(**kwargs)
        self.world_plot.show()
    
    def show_ani(self):
        self.world_plot.show_ani()
  