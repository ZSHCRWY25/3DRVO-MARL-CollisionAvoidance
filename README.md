# 3D RVO-Enhanced MARL for UAV Collision Avoidance
一个强化学习框架，能够使无人机在三维结构化空域中实现冲突解脱--避免碰撞、维持运行速度、并符合航路保护区约束
A reinforcement learning framework that enables drones to achieve conflict resolution in three-dimensional structured airspace - avoiding collisions, maintaining operational speed, and complying with route protection zone constraints

## 运行步骤 apply steps
1. 克隆仓库Clone project
git clone https://github.com/ZSHCRWY25/3DRVO-MARL-CollisionAvoidance.git
2、安装依赖 install requirements
pip install -r requirements.txt
3、运行训练 train
cd train
python train_process.py --use_gpu
