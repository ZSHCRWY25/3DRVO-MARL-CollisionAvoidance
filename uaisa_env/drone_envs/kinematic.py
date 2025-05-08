import numpy as np

class kinematic:
    def __init__(self):
        # 初始状态
        self.position = np.array([0.0, 0.0, 0.0])  # [x, y, z] (m)
        self.velocity = 0.0                        # 合速度标量 (m/s)
        self.yaw = 0.0                             # 航向角 (deg)
        self.pitch = 0.0                           # 俯仰角 (deg)
        
        # 物理约束
        self.max_acc = 10.0                        # 最大加速度 (m/s²)
        self.max_angle_change = 90.0               # 最大角度变化 (deg)

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

    def step(self, action, velocity):
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

# -----------------------------------------------------------------
# if __name__ == "__main__":
#     drone = Kinematic()
#     # 测试
#     test_action = np.array([1.0, 0.5, 1.0]) 
    
#     print("初始位置:", drone.position)
#     for _ in range(10):
#         pos = drone.step(test_action)
#         print(f"位置: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
#         print(f"当前速度: {drone.velocity:.2f} m/s | 航向: {drone.yaw:.1f}° | 俯仰: {drone.pitch:.1f}°\n")