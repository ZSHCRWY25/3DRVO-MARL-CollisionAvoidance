
import math
def line_sight_partial_3D(E3d_safe, xb_bound, yb_bound, zb_bound, sizeE):
    # 初始化
    y_size, x_size, z_size = sizeE

    
    x1_0 = xb_bound[0]
    y1_0 = yb_bound[0]
    z1_0 = zb_bound[0]

    x2 = xb_bound[1]
    y2 = yb_bound[1]
    z2 = zb_bound[1]

    x1_0=int(x1_0)
    y1_0=int(y1_0)
    z1_0=int(z1_0)

    x2=int(x2)
    y2=int(y2)
    z2=int(z2)

    # 计算距离
    dy = int(y2 - y1_0)
    dx = int(x2 - x1_0)
    dz = int(z2 - z1_0)

    # 某方向上的视线
    sy = 1 if dy >= 0 else -1
    sx = 1 if dx >= 0 else -1

    # 高度和水平轨迹之间的角度
    gamma = math.atan2(dz, math.sqrt(dx**2 + dy**2))

    # 初始化
    x1, y1, sight = x1_0, y1_0, 1

    f = 0# 
    if dy>= dx:
        while y1 != y2:
            f += dx
            if f >= dy and 0 <= y1 + (sy-1)//2 < y_size and 0 <= x1 + (sx-1)//2 < x_size:
                z = math.floor(z1_0 + math.tan(gamma) * math.sqrt((x1 + (sx-1)/2 - x1_0)**2 + (y1 + (sy-1)//2 - y1_0)**2))
                z = max(0, min(z, z_size-1))
                if E3d_safe[y1 + (sy-1)//2, x1 + (sx-1)//2, z] == 1:
                    sight = 0
                x1 += sx
                f -= dy
            if 0 <= y1 + (sy-1)/2 < y_size and 0 <= x1 + (sx-1)/2 < x_size:
                z = math.floor(z1_0 + math.tan(gamma) * math.sqrt((x1 + (sx-1)/2 - x1_0)**2 + (y1 + (sy-1)/2 - y1_0)**2))
                z = max(0, min(z, z_size-1))
                if f != 0 and E3d_safe[y1 + (sy-1)//2, x1 + (sx-1)//2, z] == 1:
                    sight = 0
            if 0 <= y1 + (sy-1)/2 <= y_size and 1 <= x1 <= x_size:
                z_1 = math.floor(z1_0 + math.tan(gamma) * math.sqrt((x1 - x1_0)**2 + (y1 + (sy-1)/2 - y1_0)**2))
                z_1 = max(0, min(z_1, z_size-1))
                z_2 = math.floor(z1_0 + math.tan(gamma) * math.sqrt((x1 - 1 - x1_0)**2 + (y1 + (sy-1)/2 - y1_0)**2))
                z_2 = max(0, min(z_2, z_size-1))
                if dx == 0 and E3d_safe[y1 + (sy-1)//2, x1-1, z_2] == 1:
                    sight = 0
            y1=y1+sy
    else:
        while x1 != x2:
            f= f + dy
            if f >= dx and 0 <= y1 + (sy-1)/2 < y_size and 0 <= x1 + (sx-1)/2 <=x_size:
                z = math.floor(z1_0 + math.tan(gamma) * math.sqrt((x1 + (sx-1)/2 - x1_0)**2 + (y1 + (sy-1)/2 - y1_0)**2))
                z = max(0, min(z, z_size-1))
                if E3d_safe[y1 + (sy-1)//2, x1 + (sx-1)//2, z] == 1:
                    sight = 0
            if 0 <= y1 + (sy-1)/2 < y_size and 0 <= x1 + (sx-1)/2 < x_size:
                z = math.floor(z1_0 + math.tan(gamma) * math.sqrt((x1 + (sx-1)/2 - x1_0)**2 + (y1 + (sy-1)/2 - y1_0)**2))
                z = max(0, min(z, z_size-1))
                if f != 0 and E3d_safe[y1 + (sy-1)//2, x1 + (sx-1)//2, z] == 1:
                    sight = 0
            if 1 <= y1 < y_size and 0 <= x1 + (sx-1)/2 < x_size:
                z_1 = math.floor(z1_0 + math.tan(gamma) * math.sqrt((x1 + (sx-1)/2 - x1_0)**2 + (y1 - y1_0)**2))
                z_1 = max(0, min(z_1, z_size-1))
                z_2 = math.floor(z1_0 + math.tan(gamma) * math.sqrt((x1 + (sx-1)/2 - x1_0)**2 + (y1 - 1 - y1_0)**2))
                z_2 = max(0, min(z_2, z_size-1))
                if dy == 0 and E3d_safe[y1 - 1, x1 + (sx-1)//2, z_2] == 1:
                    sight = 0
            x1=x1+sx
    return sight

##测试一下


        

# 
# E3d_safe = np.zeros[100,100,10]
# 
# sizeE = [100,100,10]
# result = line_sight_partial_3D(E3d_safe, xb_bound, yb_bound, zb_bound, sizeE)
