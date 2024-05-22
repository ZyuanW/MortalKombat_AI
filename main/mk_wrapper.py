import collections
import gym
import numpy as np
import retro

class MkWrapper(gym.Wrapper):
    def __init__(self, env, reset_round=True, rendering=False):
        super(MkWrapper, self).__init__(env)
        self.env = env

        self

