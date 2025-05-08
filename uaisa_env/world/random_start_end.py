
import os
import random
import math
import yaml

# 地图尺寸
width = 20
height = 20
depth = 5

# 无人机对数
num_pairs = 8

# 最小安全距离
min_distance = 20

# 存储生成的起降点
start_points = []
end_points = []

def distance(p1, p2):
    # 计算两点之间的欧几里得距离
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

def generate_point():
    # 随机生成一个点
    x = random.randint(0, width - 1)
    y = random.randint(0, height - 1)
    z = random.randint(0, depth - 1)
    return (x, y, z)

def is_valid_pair(start, end, points):
    # 检查新生成的起降点对是否有效
    for (s, e) in points:
        if (distance(start, s) < min_distance or distance(end, e) < min_distance or
                distance(start, e) < min_distance or distance(end, s) < min_distance):
            return False
    return True

# 生成起降点
points = []
for _ in range(num_pairs):
    start = generate_point()
    end = generate_point()
    while not is_valid_pair(start, end, points) or distance(start, end) < min_distance:
        end = generate_point()
    points.append((start, end))
    start_points.append(start)
    end_points.append(end)

# 构建.drone_paths.yaml文件的完整路径
current_dir = os.getcwd()  
file_path = os.path.join(current_dir, 'drone_paths.yaml')

# 写入YAML文件
data = {
    'start_points': start_points,
    'end_points': end_points
}

with open(file_path, 'w') as file:
    yaml.dump(data, file, default_flow_style=False)

print(f"Generated drone_paths.yaml file with {num_pairs} pairs of start and end points.")