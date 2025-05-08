
import numpy as np
from uaisa_env.world.line_sight_partial_3D import line_sight_partial_3D

def theta_star_3D(K, E3d_safe, x0_old, y0_old, z0_old, xend_old, yend_old, zend_old, sizeE):
    # 初始化参数
    kg = K[0]  # 代价函数
    kh = K[1]
    ke = K[2]

    # 起止点
    y0 = int(np.floor(x0_old))
    x0 = int(np.floor(y0_old))
    z0 = int(np.floor(z0_old))
    
    yend = int(np.ceil(xend_old))
    xend = int(np.ceil(yend_old))
    zend = int(np.ceil(zend_old))

    y_size = sizeE[0]
    x_size = sizeE[1]
    z_size = sizeE[2]

    came_fromx = np.zeros((y_size, x_size, z_size), dtype=int)
    came_fromy = np.zeros((y_size, x_size, z_size), dtype=int)
    came_fromz = np.zeros((y_size, x_size, z_size), dtype=int)

    came_fromx[y0, x0, z0] = x0
    came_fromy[y0, x0, z0] = y0
    came_fromz[y0, x0, z0] = z0

    closed_list = np.array([], dtype=int)
    open_list = np.array([[y0, x0, z0]], dtype=int)
    G = float("inf") * np.ones((y_size, x_size, z_size))
    G[y0, x0, z0] = 0
    F = float("inf") * np.ones((y_size, x_size, z_size))
    F[y0, x0, z0] = np.sqrt((xend-x0)**2 + (yend-y0)**2 + (zend-z0)**2)
    f_open = np.array([[F[y0, x0, z0]]])

    exit_path = 0 

    # 搜索航路
    while len(open_list) > 0 and exit_path == 0:
        i_f_open_min = np.argmin(f_open)
        ycurrent = open_list[i_f_open_min, 0]
        xcurrent = open_list[i_f_open_min, 1]
        zcurrent = open_list[i_f_open_min, 2]
        if xcurrent == xend and ycurrent == yend and zcurrent == zend:
            exit_path = 1
        else:
            if closed_list.shape[0] == 0:
                closed_list = np.array([[ycurrent, xcurrent, zcurrent]], dtype=int)
            else:
                closed_list = np.vstack((closed_list, np.array([ycurrent, xcurrent, zcurrent], dtype=int)))
            open_list = np.delete(open_list, i_f_open_min, 0)
            f_open = np.delete(f_open, i_f_open_min, 0)
            for i in range(-1, 2):
                for j in range(-1, 2):
                    for k in range(-1, 2):
                        if 0 <= xcurrent + i < x_size and 0 <= ycurrent + j < y_size and 0 <= zcurrent + k < z_size:
                            neigh = np.array([ycurrent + j, xcurrent + i, zcurrent + k], dtype=int)
                            sum_open = np.sum(neigh == open_list, 1)
                            sum_closed = np.sum(neigh == closed_list, 1)
                            if len(sum_open) > 0:
                                max_sum_open = max(sum_open)
                            else:
                                max_sum_open = 0
                            if len(sum_closed) > 0:
                                max_sum_closed = max(sum_closed)
                            else:
                                max_sum_closed = 0
                            if max_sum_open < 3 and max_sum_closed < 3:
                                open_list = np.vstack((open_list, neigh))
                                xb = [came_fromx[ycurrent, xcurrent, zcurrent], xcurrent + i]
                                yb = [came_fromy[ycurrent, xcurrent, zcurrent], ycurrent + j]
                                zb = [came_fromz[ycurrent, xcurrent, zcurrent], zcurrent + k]
                                sight = line_sight_partial_3D(E3d_safe, xb, yb, zb, sizeE)  # 视线检查
                                if sight == 1:
                                    yg = int(came_fromy[ycurrent, xcurrent, zcurrent])
                                    xg = int(came_fromx[ycurrent, xcurrent, zcurrent])
                                    zg = int(came_fromz[ycurrent, xcurrent, zcurrent])
                                    g_try = G[yg, xg, zg] + np.sqrt((came_fromy[ycurrent, xcurrent, zcurrent] - (ycurrent+j))**2 + (came_fromx[ycurrent, xcurrent, zcurrent] - (xcurrent+i))**2 + (came_fromz[ycurrent, xcurrent, zcurrent] - (zcurrent+k))**2)
                                    if g_try < G[ycurrent + j, xcurrent + i, zcurrent + k]:
                                        came_fromy[ycurrent + j, xcurrent + i, zcurrent + k] = came_fromy[ycurrent, xcurrent, zcurrent]
                                        came_fromx[ycurrent + j, xcurrent + i, zcurrent + k] = came_fromx[ycurrent, xcurrent, zcurrent]
                                        came_fromz[ycurrent + j, xcurrent + i, zcurrent + k] = came_fromz[ycurrent, xcurrent, zcurrent]
                                        G[ycurrent + j, xcurrent + i, zcurrent + k] = g_try
                                        H = np.sqrt((xend - (xcurrent + i))**2 + (yend - (ycurrent + j))**2 + (zend - (zcurrent + k))**2)
                                        F[ycurrent + j, xcurrent + i, zcurrent + k] = kg * G[ycurrent + j, xcurrent + i, zcurrent + k] + kh * H + ke * E3d_safe[ycurrent + j, xcurrent + i, zcurrent + k]
                                else:
                                    g_try = G[ycurrent, xcurrent, zcurrent] + np.sqrt(i**2 + j**2 + k**2)
                                    if g_try < G[ycurrent + j, xcurrent + i, zcurrent + k]:
                                        came_fromy[ycurrent + j, xcurrent + i, zcurrent + k] = ycurrent
                                        came_fromx[ycurrent + j, xcurrent + i, zcurrent + k] = xcurrent
                                        came_fromz[ycurrent + j, xcurrent + i, zcurrent + k] = zcurrent
                                        G[ycurrent + j, xcurrent + i, zcurrent + k] = g_try
                                        H = np.sqrt((xend - (xcurrent + i))**2 + (yend - (ycurrent + j))**2 + (zend - (zcurrent + k))**2)
                                        F[ycurrent + j, xcurrent + i, zcurrent + k] = kg * G[ycurrent + j, xcurrent + i, zcurrent + k] + kh * H + ke * E3d_safe[ycurrent + j, xcurrent + i, zcurrent + k]
                                f_open = np.vstack((f_open, [F[ycurrent + j, xcurrent + i, zcurrent + k]]))

    # 回溯生成路径
    path_backwards = [ycurrent, xcurrent, zcurrent]

    i = 1

    while (xcurrent != x0) or (ycurrent != y0) or (zcurrent != z0):
        next_point = np.array([came_fromy[ycurrent, xcurrent, zcurrent], came_fromx[ycurrent, xcurrent, zcurrent], came_fromz[ycurrent, xcurrent, zcurrent]], dtype=int)
        path_backwards = np.vstack((path_backwards, next_point))

        ycurrent = int(path_backwards[i, 0])
        xcurrent = int(path_backwards[i, 1])
        zcurrent = int(path_backwards[i, 2])

        i = i + 1

    # 航路点数量
    n_points = path_backwards.shape[0]

    path = np.flipud(path_backwards)

   


    return path, n_points
