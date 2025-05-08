import gym
from pathlib import Path

world_name = 'gym_test_world.yaml'
# world_name = 'dynamic_obs_test.yaml'

env = gym.make('mdin-v1', world_name=world_name)
env.drone_reset(ifrender=True)
ifrender=True
for i in range(300):

    vel_list = env.ir_gym.cal_des_list()
    print(vel_list)

    obs_list, reward_list, done_list, info_list, finish_list = env.drone_step(vel_list)
    id_list=[id for id, done in enumerate(done_list) if done==True]
    
    for id in id_list: 
        env.drone_reset_one(ifrender,id)

    env.drone_render()



