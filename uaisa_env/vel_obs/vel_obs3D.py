
import numpy as np
from math import sin, cos, atan2, asin, pi, inf, sqrt, acos
from time import time

####速度障碍法:Pa,Pb,ra,rb,pra,prb

def get_alpha(Pa, Pb, ra, rb):##计算圆锥张角(弧度制)
        R = ra + rb
        pa = np.array(Pa)  
        pb = np.array(Pb)
        norm_ab = np.linalg.norm(pb - pa)
        alpha = asin(R/norm_ab)
        alpha = wraptopi(alpha)
        alpha = round(alpha, 2)

        return alpha
    
def get_PAA(Pa, pra, prb, Va, Vb):##计算互惠速度障碍物起点
        t = 1 
        x1, y1, z1 = Pa
        vax, vay, vaz = Va
        vbx, vby, vbz = Vb
        pr = pra / (pra + prb)##优先级
        PAAx = pr*(2*x1 + (vax + vbx)*t)
        PAAy = pr*(2*y1 + (vay + vby)*t)
        PAAz = pr*(2*z1 + (vaz + vbz)*t)
        # PAAx = x1 + vbx*t
        # PAAy = y1 + vby*t
        # PAAz = z1 + vbz*t
        PAA = [PAAx, PAAy, PAAz]
        return PAA 


def get_rvo_array(Pa, Pb):
      Pax, Pay, Paz = Pa
      Pbx, Pby, Pbz = Pb
      Px_array = Pbx - Pax
      Py_array = Pby - Pay
      Pz_array = Pbz - Paz
      v_rvo_array = [Px_array, Py_array, Pz_array]
      return v_rvo_array

def get_beta(A, B):##计算新位置与速度障碍物的夹角
        # 计算向量A和B的点积
        dot_product = np.dot(A, B)
    
        # 计算向量A和B的模长
        magnitude_A = np.linalg.norm(A)
        magnitude_B = np.linalg.norm(B)
        AB = magnitude_A * magnitude_B
    
        # 计算夹角的cos值
        if AB != 0:
            cos_angle = dot_product / AB
        else:
         cos_angle = 0 

    
        # 求得夹角（弧度制）
        angle_radians = wraptopi(np.arccos(cos_angle))
    
        # 保留小数点后一位
        beta =  round(angle_radians, 2)
        
        return beta
    

def calculate_projection_length(a, b):##计算a在b上的投影长度
    # 计算点积
        dot_product = np.dot(a, b)
    
    #B模长
        b_length = np.linalg.norm(b)
    
    # 投影长度
        projection_length = dot_product / b_length
    
        return projection_length


# def whcol(alpha, beta, Pa, Pb, Va, Vb, t, R):##判断是否有碰撞可能
#         ## 1.计算Anew到B'距离是否小于R
#         ##      是，碰撞
#         PAnew = Pa + Va*t
#         PAnew = map(lambda x:x+Va*t,Pa)
#         PBnew = Pb + Vb*t
        
#         L = np.linalg.norm(PAnew-PBnew)
#         if L <= R:
#             COL = 1
#         elif alpha >= beta:
#             COL = 1
#         else:
#             COL = 0
#             return COL
#         ##2.比较beta和alpha大小
#         ##if alpha >= beta
#         ## 计算AAnew在A'B'上投影长度
#         ## if 投影长度小于A'B'模长
#         ##  碰撞
#         return 



def cal_exp_tim(Pa, Pb, Va, Vb, ra, rb):
        rel_x = Pa[0] - Pb[0]
        rel_y = Pa[1] - Pb[1]
        rel_z = Pa[2] - Pb[2]

        rel_vx = Va[0] -  Vb[0]
        rel_vy = Va[1] -  Vb[1]
        rel_vz = Va[2] -  Vb[2]

        r = ra + rb

        # rel_x: xa - xb
        # rel_y: ya - yb
        # rel_z: za - zb
        # (vx2 + vy2 + vz2)*t2 + (2x*vx + 2*y*vy)*t+x2+y2-(r+mr)2 = 0
        # 计算期望碰撞时间。

        a = rel_vx ** 2 + rel_vy ** 2 + rel_vz**2
        b = 2* rel_x * rel_vx + 2* rel_y * rel_vy + 2* rel_z * rel_vz 
        c = rel_x ** 2 + rel_y ** 2 + rel_z ** 2 - r ** 2

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

def cal_vo_exp_tim(rel_x, rel_y, rel_z, rel_vx_origin, rel_vy_origin, rel_vz_origin ,  ra, rb):
    r = ra + rb

        # rel_x: xa - xb
        # rel_y: ya - yb
        # rel_z: za - zb
        # (vx2 + vy2 + vz2)*t2 + (2x*vx + 2*y*vy)*t+x2+y2-(r+mr)2 = 0
        # 计算期望碰撞时间。

    rel_vx = - rel_vx_origin
    rel_vy = - rel_vy_origin
    rel_vz = - rel_vz_origin


    a = rel_vx ** 2 + rel_vy ** 2 + rel_vz**2
    b = 2* rel_x * rel_vx + 2* rel_y * rel_vy + 2* rel_z * rel_vz 
    c = rel_x ** 2 + rel_y ** 2 + rel_z ** 2 - r ** 2

    if c <= 0:
        return 0

    temp = b ** 2 - 4 * a * c

    if temp <= 0:
        t = inf
    else:
        t1 = ( -b + sqrt(temp) ) / (2 * a)
        t2 = ( -b - sqrt(temp) ) / (2 * a)

        if t1 < 0 and t2 < 0:
              return -1

        t3 = t1 if t1 >= 0 else inf
        t4 = t2 if t2 >= 0 else inf
        
        t = min(t3, t4)

    return t








def distance(point1, point2):
        return sqrt( (point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2 + (point2[2] - point1[2]) ** 2)


def wraptopi(theta):
        if theta > pi:
            theta = theta - 2*pi
        
        if theta < -pi:
            theta = theta + 2*pi

        return theta
    
@staticmethod
def clamp(n, minn, maxn):
        return max(min(maxn, n), minn)    

@staticmethod
def norm(x):
        a=0
        for i in range(len(x)):
            a = x[1]^2+a
        return a

    
# #####测试一下
# P_dr = [[12, 34, 27],
#             [48, 19, 42],
#             [7, 8, 15],
#             [33, 45, 5],
#             [25, 2, 38]]##随机生成五个无人机位置

# V_dr = [[4.5, 5.8,1.7],
#         [4.1, 3.2, 1.9],
#         [1.5, 2.7, 3.4],
#         [3.9, 1.2, 4.6],
#         [5.8, 4.3, 2.1]]##随机速度

# r = [2.5,2.5,2.5,2.5,2.5]##半径

# for i in range(5):
#     for j in range(i+1,5):
#         alpha = get_alpha(P_dr[i], P_dr[j], r[i], r[j])*(180/pi)
#         PAA = get_PAA(P_dr[i], 2, 2, V_dr[i], V_dr[j])
#         beta = get_beta(P_dr[i], P_dr[j], PAA, V_dr[j], V_dr[i], 1)
#         v = list(map(lambda x,y: x-y, V_dr[i],V_dr[j]))
#         rel_x = P_dr[i][0] - P_dr[j][0]
#         rel_y = P_dr[i][1] - P_dr[j][1]
#         rel_z = P_dr[i][2] - P_dr[j][2]
#         rel_vx = V_dr[i][0] -  V_dr[j][0]
#         rel_vy = V_dr[i][1] -  V_dr[j][1]
#         rel_vz = V_dr[i][2] -  V_dr[j][2]
#         tim = cal_exp_tim(rel_x, rel_y, rel_z, rel_vx, rel_vy, rel_vz, 20)
#         print(alpha, PAA, v, tim)##角度是一半##取整