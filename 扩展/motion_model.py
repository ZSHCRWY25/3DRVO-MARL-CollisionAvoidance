import numpy as np
from math import sin, cos, pi, tan,radians
######这个是用来写欧拉角表示无人机运动的，但是在Dronr类里已经用方位角+俯仰角表示了，这个文件后面再完善吧
# reference: probabilistic robotics[book], motion model P127

def motion_omni(current_state, vel, sampletime, noise = False, control_std = [0.01, 0.01, 0.01]):

    # vel: np.array([[vel x], [vel y]])
    # current_state: np.array([[x], [y]])

    if noise == True:
        vel_noise = vel + np.random.normal([[0], [0], [0]], scale = [[control_std[0]], [control_std[1]], [control_std[2]]])
    else:
        vel_noise = vel

    next_state = current_state + vel_noise * sampletime
    
    return next_state 

def rotation_matrix_roll(roll):  
    roll_rad = radians(roll)  
    return np.array([  
        [1, 0, 0],  
        [0, cos(roll_rad), -sin(roll_rad)],  
        [0, sin(roll_rad), cos(roll_rad)]  
    ])  
  
def rotation_matrix_pitch(pitch):  
    pitch_rad = radians(pitch)  
    return np.array([  
        [cos(pitch_rad), 0, sin(pitch_rad)],  
        [0, 1, 0],  
        [-sin(pitch_rad), 0, cos(pitch_rad)]  
    ])  
  
def rotation_matrix_yaw(yaw):  
    yaw_rad = radians(yaw)  
    return np.array([  
        [cos(yaw_rad), -sin(yaw_rad), 0],  
        [sin(yaw_rad), cos(yaw_rad), 0],  
        [0, 0, 1]  
    ])  
  

def motion_ackermann(state, velocity_body, steer_limit=pi/2, step_time=1, theta_trans=True):
    
    # l: wheel base
    # state: 0, x
    #        1, y
    #        2  Z
    #        绕 x 轴旋转角用 ϕ \phiϕ 表示，绕 y 轴旋转角用 θ \thetaθ 表示，绕 z 轴旋转角用 ψ \psiψ 表示。绕坐标轴逆时针旋转为正，顺时针为负
    #        3, roll angle 滚转角 
    #        4, pitch angle 俯仰角 
    #        5, yaw angle 偏航角
    #        存储形式z-y-x
    #        velocity_body = np.array([forward_speed, lateral_speed, vertical_speed])
    roll = state[2, 0]  
    pitch = state[3, 0]
    yaw = state[4, 0] 

    roll_rad = radians(roll)  
    pitch_rad = radians(pitch)  
    yaw_rad = radians(yaw)  
      
    # 构建旋转矩阵  
    R_roll = rotation_matrix_roll(roll_rad)  
    R_pitch = rotation_matrix_pitch(pitch_rad)  
    R_yaw = rotation_matrix_yaw(yaw_rad)  
      
    # 合并旋转矩阵（Z-Y-X欧拉角顺序）  
    R = R_yaw @ R_pitch @ R_roll  
      
    # 将速度从机体坐标系转换到世界坐标系  
    velocity_world = R @ velocity_body  
      
    # 更新位置  

    new_state = np.zeros((6, 1))
    
    new_state[0:3] = state[0:3] + velocity_world * step_time
    

    if theta_trans:####原始代码中可以根据当前状态更新角度，但是我没写出来
        new_state[3, 0] = wraptopi(state[3, 0]) 
        
    new_state[4, 0] = np.clip(state[4, 0], -steer_limit, steer_limit) 
    new_state[5, 0] = np.clip(state[5, 0], -steer_limit, steer_limit)

    return new_state

def motion_acker_pre(state, wheelbase=1, vel=1, psi=0, steer_limit=pi/4, pre_time=2, time_step=0.1):
    
    # l: wheel base
    # vel: linear velocity, steer
    # state: 0, x
    #        1, y
    #        2, phi, heading direction
    #        3, psi, steering angle
    
    cur_time = 0

    while cur_time < pre_time:
        phi = state[2, 0] 
        d_state = np.array([ [vel*cos(phi)], [vel*sin(phi)], [vel*tan(psi) / wheelbase], [0] ])
        print(d_state * time_step)
        new_state = state + d_state * time_step
        new_state[2, 0] = wraptopi(new_state[2, 0]) 
        new_state[3, 0] = np.clip(psi, -steer_limit, steer_limit) 

        cur_time = cur_time + time_step
        state = new_state

    return new_state

def motion_acker_step(state, gear=1, steer=0, step_size=0.5, min_radius=1, include_gear=False):
    
    # psi: steer angle
    # state: 0, x
    #        1, y
    #        2, phi, heading direction
    # gear: -1, 1
    # steer: 1, 0, -1, left, straight, right
    if not isinstance(state, np.ndarray):
        state = np.array([state]).T

    cur_x = state[0, 0]
    cur_y = state[1, 0]
    cur_theta = state[2, 0]

    curvature = steer * 1/min_radius
    
    rot_theta = abs(steer) * step_size * curvature * gear
    trans_len = (1 - abs(steer)) * step_size * gear

    rot_matrix = np.array([[cos(rot_theta), -sin(rot_theta)], [sin(rot_theta), cos(rot_theta)]])
    trans_matrix = trans_len * np.array([[cos(cur_theta)], [sin(cur_theta)]]) 

    center_x = cur_x + cos(cur_theta + steer * pi/2) * min_radius
    center_y = cur_y + sin(cur_theta + steer * pi/2) * min_radius
    center = np.array([[center_x], [center_y]])

    if include_gear:
        new_state = np.zeros((4, 1))
        new_state[0:2] = np.around(rot_matrix @ (state[0:2] - center) + center + trans_matrix, 4)
        new_state[2, 0] = np.around(mod(cur_theta + rot_theta), 4)
        new_state[3, 0] = gear
    else:
        new_state = np.zeros((3, 1))
        new_state[0:2] = np.around(rot_matrix @ (state[0:2] - center) + center + trans_matrix, 4)
        new_state[2, 0] = np.around(mod(cur_theta + rot_theta), 4)
    
    return new_state


# def motion_acker_step(state, gear=1, steer=0, step_size=0.5, min_radius=1, include_gear=False):
    
#     # psi: steer angle
#     # state: 0, x
#     #        1, y
#     #        2, phi, heading direction
#     # gear: -1, 1
#     # steer: 1, 0, -1, left, straight, right
#     if not isinstance(state, np.ndarray):
#         state = np.array([])

#     cur_x = state[0, 0]
#     cur_y = state[1, 0]
#     cur_theta = state[2, 0]

#     curvature = steer * 1/min_radius
    
#     rot_theta = abs(steer) * step_size * curvature * gear
#     trans_len = (1 - abs(steer)) * step_size * gear

#     rot_matrix = np.array([[cos(rot_theta), -sin(rot_theta)], [sin(rot_theta), cos(rot_theta)]])
#     trans_matrix = trans_len * np.array([[cos(cur_theta)], [sin(cur_theta)]]) 

#     center_x = cur_x + cos(cur_theta + steer * pi/2) * min_radius
#     center_y = cur_y + sin(cur_theta + steer * pi/2) * min_radius
#     center = np.array([[center_x], [center_y]])

#     if include_gear:
#         new_state = np.zeros((4, 1))
#         new_state[0:2] = np.around(rot_matrix @ (state[0:2] - center) + center + trans_matrix, 4)
#         new_state[2, 0] = np.around(mod(cur_theta + rot_theta), 4)
#         new_state[3, 0] = gear
#     else:
#         new_state = np.zeros((3, 1))
#         new_state[0:2] = np.around(rot_matrix @ (state[0:2] - center) + center + trans_matrix, 4)
#         new_state[2, 0] = np.around(mod(cur_theta + rot_theta), 4)
    
#     return new_state


def wraptopi(radian):
    # -pi to pi

    if radian > pi:
        radian2 = radian - 2 * pi
    elif radian < -pi:
        radian2 = radian + 2 * pi
    else:
        radian2 = radian

    return radian2

def mod(theta):
    theta = theta % (2*pi)
    if theta < - pi:
        return theta + 2*pi
    if theta >= pi:
        return theta - 2*pi
    return theta


