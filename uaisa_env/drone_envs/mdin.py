
import gym
import numpy as np
# from uaisa.uaisa_env.drone_envs.ir_gym import ir_gym
from uaisa_env.drone_envs.ir_gym import ir_gym
class mdin(gym.Env):
    def __init__(self, world_name=None, neighbors_region=5, neighbors_num=10, **kwargs):
        
        self.ir_gym = ir_gym( world_name, neighbors_region, neighbors_num, **kwargs)
        
        self.observation_space = self.ir_gym.observation_space
        self.action_space = self.ir_gym.action_space
        
        self.neighbors_region = neighbors_region
        self.rvo_observation_list = []
        self.drow_rvo_flag = True


    def drone_step(self, action, **kwargs):

        if not isinstance(action, list):
            action = [action]

        rvo_reward_list, vo_flag_list = self.ir_gym.rvo_reward_list_cal(action)
        self.ir_gym.drone_step(action, vo_flag_list)
        obs_list, mov_reward, done_list, info_list, finish_list = self.ir_gym.obs_move_reward_list(action,)#移动奖励（碰撞+到点+终点），碰撞标准，到点标志 ，到终点标志]

        reward_list = [x+y for x, y in zip(rvo_reward_list, mov_reward)]

        return obs_list, reward_list, done_list, info_list, finish_list

    def drone_render(self, save=False, path=None, i = 0, **kwargs):
        self.ir_gym.render(0.01, **kwargs)
                                                                        
        if save:
            self.ir_gym.save_fig(path, i) 

    def drone_reset(self,ifrender):
        if ifrender:
            self.ir_gym.world_plot.clear_alltraj()
        return self.ir_gym.env_reset()

    def drone_reset_one(self, ifrender,id):
        if ifrender:
            self.ir_gym.world_plot.clear_onetraj(id)
        self.ir_gym.components['drones'].drone_reset(id)

    def drone_show(self):
        self.ir_gym.show()