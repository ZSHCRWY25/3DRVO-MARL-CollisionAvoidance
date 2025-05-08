'''
:@Author: 刘玉璞
:@Date: 2024/4/13 21:58:20
:@LastEditors: Your Name
:@LastEditTime: 2024/9/16 23:10:45
:Description: 
'''
# 输入：地图，起止点
# 输出：航路（点集）+各航路航路点数量
import os
import yaml
import numpy as np
import json  
import pandas as pd
import matplotlib.pyplot as plt
from uaisa_env.world.theta_star_3D import theta_star_3D
from uaisa_env.world.grid_3D_safe_zone import grid_3D_safe_zone
import math

def read_start_des():
    current_dir = os.getcwd()  
    # 构建.drone_paths.yaml文件的完整路径  
    file_path = os.path.join(current_dir, 'uaisa_env\\world\\world_8\\drone_paths.yaml')  

    # 检查文件是否存在  
    if not os.path.exists(file_path):  
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    try:  
        # 打开并读取YAML文件  
        with open(file_path, 'r') as file:

            data = yaml.safe_load(file)  
            # 从数据中提取起点和终点的列表  
            starting  = data['start_points']  
            destination = data['end_points']
            dron_num = len(starting)

    except yaml.YAMLError as e:  
        # 处理YAML格式错误  
        raise ValueError(f"Error parsing YAML file {file_path}: {e}")  
    
    except Exception as e:  
        # 处理其他未预期的异常  
        raise RuntimeError(f"An unexpected error occurred while reading the YAML file: {e}")
    
    return starting, destination, dron_num 

def init_map_road(map_size, dron_num, starting, destination, waypoints_list, n_points_list):
    #E, E_safe, E3d, E3d_safe ,obs_list = grid_3D_safe_zone(map_size, 1, 10, dron_num, starting, destination, 1)
#E是二维障碍物地图
#E_safe带保护区用来画图
#E3d三维障碍物，用于学习环境
#E3d_safe带保护区，用于路径规划
#obs = [[x,y,h,r]]
    x_range, y_range, z_range = map_size
    E3d_safe = np.zeros((x_range, y_range, z_range))
    z_grid = np.linspace(1, z_range, z_range//1)


    for i in range(round(x_range/2)-2,round(x_range/2)+3):
        for j in range(round(y_range/2)-2,round(y_range/2)+2):
            for k in range(len(z_grid)):  
                E3d_safe[i, j, k] = 1  
    # fig = plt.figure(figsize=(8, 6))  
    # ax = fig.add_subplot(111, projection='3d')  

    obs_list = [[round(x_range/2),round(y_range/2),5,1]]
    
    # 设置地图的x和y轴范围  
    ax.set_xlim(0, x_range)  
    ax.set_ylim(0, y_range)  
    ax.set_zlim(0, z_range + 5)  # z轴范围基于建筑物最高度+一点额外空间  
    
# 绘制每个建筑物（圆柱体）  
    for b in obs_list:  
        x, y, h, r = b  
        
        # 生成极坐标和高度  
        u = np.linspace(0, 2 * np.pi, 50)  
        h_vals = np.linspace(0, h, 20)  # 足够的高度切片以形成平滑的圆柱体  
        
        # 使用meshgrid生成二维网格上的X, Y, Z  
        U, H = np.meshgrid(u, h_vals)  
        X = x + r * np.sin(U)  # 考虑建筑物的x坐标  
        Y = y + r * np.cos(U)  # 考虑建筑物的y坐标  
        Z = H  
        
        # 绘制曲面，并设置颜色为蓝色  
        plt.gca().set_prop_cycle(None)   
        ax.plot_surface(X, Y, Z, linewidth=0, facecolor='b', shade=True, alpha=0.6)

    for i in range(dron_num):
        path, n_pionts = path_plan(map_size, starting[i], destination[i], E3d_safe)
        waypoints_list.append(path)
        n_points_list.append(n_pionts)
        ax.plot(path[:][:, 0], path[:][:, 1], path[:][:, 2], 'kx-')
        ax.plot(starting[i][0], starting[i][1], starting[i][2], 'go')
        ax.plot(destination[i][0], destination[i][1], destination[i][2], 'ro')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.auto_scale_xyz([0, map_size[0]], [0, map_size[1]], [0, map_size[2]])
        # plt.show()
    ##存储无人机航路点与航路点数量


    return E3d_safe, obs_list, waypoints_list, n_points_list

def path_plan(sizeE, P0, Pend, E3d_safe):
    ##############################################################################################################

    x0 = math.ceil(P0[0])
    y0= math.ceil(P0[1])
    z0= math.ceil(P0[2])

    xend= math.ceil(Pend[0])
    yend = math.ceil(Pend[1])
    zend = math.ceil(Pend[2])
    d_grid = 1

    kg = 1
    kh = 1.25
    ke = np.sqrt((xend-x0)**2+(yend-y0)**2+(zend-z0)**2)

    K = [kg, kh, ke]

    path, n_points = theta_star_3D(K, E3d_safe, x0, y0, z0, xend, yend, zend, sizeE)

    X = np.arange(1, sizeE[0]-1, d_grid)
    Y = np.arange(1, sizeE[1]-1, d_grid)
    X, Y = np.meshgrid(X, Y)

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    #ax.plot_surface(X, Y, E3d_safe[1:-1][:, 1:-1])
    ax.plot(path[:][:, 0], path[:][:, 1], path[:][:, 2], 'kx-')
    ax.plot([x0], [y0], [z0], 'go')
    ax.plot([xend], [yend], [zend], 'ro')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.auto_scale_xyz([0, sizeE[0]], [0, sizeE[1]], [0, sizeE[2]])

    return path, n_points

def ndarray_list_to_list(ndarray_list):  
    """  这个是文心一言写的
    将 NumPy 数组列表转换为 Python 列表列表。  
  
    参数:  
    - ndarray_list: 一个包含 NumPy 数组的列表。  
  
    返回:  
    - list_of_lists: 一个包含 Python 列表的列表，每个内部列表对应原始列表中的一个 NumPy 数组。  
    """  
    return [arr.tolist() for arr in ndarray_list]  
  
def ndarray_list_to_json(ndarray_list):  
    """  
    将 NumPy 数组列表转换为 JSON 字符串。  
  
    参数:  
    - ndarray_list: 一个包含 NumPy 数组的列表。  
  
    返回:  
    - json_str: 包含 NumPy 数组数据的 JSON 字符串。  
    """  
    list_of_lists = ndarray_list_to_list(ndarray_list)  
    json_str = json.dumps(list_of_lists)  
    return json_str  

map_height = 12
map_width = 12
map_high =6
map_size = [map_height, map_width, map_high]
waypoints_list=[]
n_points_list=[]


fig = plt.figure(figsize=(8, 6))  
ax = fig.add_subplot(111, projection='3d')  

starting, destination, dron_num = read_start_des()
E3d_safe, obs_list, waypoints_list, n_points_list = init_map_road(map_size, dron_num, starting, destination, waypoints_list, n_points_list)

plt.show()
#转化数据
waypoints_list_converted = ndarray_list_to_list(waypoints_list) 

#指定路径
base_path = 'uaisa_env\\world\\world_8'  # 可以根据障碍物密度与无人机数量创建文件夹
file_prefix = 'data_'  
file_number = 1 ##记得改编号
file_path = os.path.join(base_path, f"{file_prefix}{file_number}.json")  
if not os.path.exists(base_path):  
    os.makedirs(base_path)  
  

data = {'drone_num':dron_num, 'map_size':map_size, 'waypoints_list': waypoints_list_converted,'n_points_list': n_points_list, 'building_list':obs_list}
 
with open(file_path, 'w') as file:
    json.dump(data, file, indent=4) 
#存储地图
#np.save(os.path.join(base_path, 'E3d.npy'), E3d)
np.save(os.path.join(base_path, 'E3d_safe.npy'), E3d_safe) 
#存储障碍物列表
# building_list = {'building_list':obs_list}
# building_list_path = os.path.join(base_path, 'building_list.json')
# with open(building_list_path, 'w') as file:  
#     json.dump(building_list, file, indent=4)