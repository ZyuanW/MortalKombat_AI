import collections
import gym
import numpy as np
import retro
import cv2
import time
import math

class MkWrapper(gym.Wrapper):
    def __init__(self, env, reset_round=True, rendering=False):
        super(MkWrapper, self).__init__(env)
        self.env = env

        # set up a queue to store the last 9 frames
        self.max_frames = 9
        self.frame_stack = collections.deque(maxlen=self.max_frames)

        # step frames for the game
        self.num_step_frame = 6

        self.reset_round = reset_round
        self.rendering = rendering

        # reward coefficients, for scaling the rewards
        self.reward_coeff = 3.0

        # total time steps
        self.total_timesteps = 0

        # FULL_HP = 120
        self.full_hp = 120

        self.prev_player_hp = self.full_hp
        self.prev_enemy_hp = self.full_hp

        # define the observation space
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(112, 160, 3), dtype=np.uint8) # do not modify this line !!!!
        
        # self.game = 'MortalKombatII-Genesis'
        # self.state = 'Level1.LiuKangVsJax'

        # self.game_env = retro.make(
        #     game = self.game,
        #     state = self.state,
        #     use_restricted_actions=retro.Actions.FILTERED,
        #     obs_type = retro.Observations.IMAGE
        # )

    # 9 fames stack, get 3, 6, 9 then stack them
    def _stack_observation(self):
        return np.stack([self.frame_stack[i * 3 + 2][:, :, i] for i in range(3)], axis=-1)


    # def _stack_observation(self):
    #     return np.concatenate(list(self.frame_stack), axis=-1)
    
    def reset(self):
        observation = self.env.reset()
        # observation = self.preprocess(observation)

        self.prev_player_hp = self.full_hp
        self.prev_enemy_hp = self.full_hp

        self.total_timesteps = 0

        # clear the frames stack and add the first observation times
        self.frame_stack.clear()
        for _ in range(self.max_frames):
            self.frame_stack.append(observation[::2, ::2, :])
        
        return self._stack_observation()
        # return self._stack_observation()
    

    # def preprocess(self, observation): 
    #     # Grayscaling 
    #     gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
    #     # Resize 
    #     resize = cv2.resize(gray, (84,84), interpolation=cv2.INTER_CUBIC)
    #     # Add the channels value
    #     channels = np.reshape(resize, (84,84,1))
    #     return channels 
    
    def step(self, action):
        tmp_done = False

        obs, _reward, _done, info = self.env.step(action)
        # obs = self.preprocess(obs)
        self.frame_stack.append(obs[::2, ::2, :])

        # render if rendering is enabled
        if self.rendering:
            self.render()

        # update the hp values
        # curr_player_hp = info['agent_hp']
        # curr_enemy_hp = info['enemy_hp']
        curr_player_hp = info['health']
        curr_enemy_hp = info['enemy_health']

        # update the total time steps
        self.total_timesteps += self.num_step_frame

        # calculate the reward
        tmp_reward = 0
        # tmp_reward, tmp_done = self.reward_function(curr_player_hp, curr_enemy_hp, tmp_done)
        tmp_reward, tmp_done = self.reward_function_modified(curr_player_hp, curr_enemy_hp, tmp_done)
        
        return self._stack_observation(), 0.001 * tmp_reward, tmp_done, info

    def reward_function(self, curr_player_hp, curr_enemy_hp, tmp_done):
        t_reward = 0
        # if game is over and player loses
        if curr_player_hp <= 0 and curr_enemy_hp > 0:
            t_reward -= self.reward_coeff * self.full_hp
            tmp_done = True
        # if game is over and player wins
        elif curr_player_hp > 0 and curr_enemy_hp <= 0:
            t_reward += self.reward_coeff * self.full_hp
            tmp_done = True
        # attack success reward
        elif curr_enemy_hp < self.prev_enemy_hp:
            t_reward += self.reward_coeff * (self.prev_enemy_hp - curr_enemy_hp)
            tmp_done = False
        # damage taken penalty
        elif curr_enemy_hp > self.prev_enemy_hp:
            t_reward -= self.reward_coeff * (curr_enemy_hp - self.prev_enemy_hp)
            tmp_done = False
        
        # update the previous hp values
        self.prev_player_hp = curr_player_hp
        self.prev_enemy_hp = curr_enemy_hp

        if not self.reset_round:
            tmp_done = False
        
        return t_reward, tmp_done 
    
    def reward_function_modified(self, curr_player_hp, curr_enemy_hp, tmp_done):
        t_reward = 0
        # Game is over and player loses.
        if curr_player_hp <= 0:
            t_reward = -math.pow(self.full_hp, (curr_enemy_hp + 1) / (self.full_hp + 1))    # Use the remaining health points of opponent as penalty. 
                                                   # If the opponent also has negative health points, it's a even game and the reward is +1.
            tmp_done = True
        # Game is over and player wins.
        elif curr_enemy_hp <= 0:
            # custom_reward = curr_player_health * self.reward_coeff # Use the remaining health points of player as reward.
                                                                   # Multiply by reward_coeff to make the reward larger than the penalty to avoid cowardice of agent.

            # custom_reward = math.pow(self.full_hp, (5940 - self.total_timesteps) / 5940) * self.reward_coeff # Use the remaining time steps as reward.
            t_reward = math.pow(self.full_hp, (curr_player_hp + 1) / (self.full_hp + 1)) * self.reward_coeff
            tmp_done = True

        # While the fighting is still going on
        else:
            t_reward = self.reward_coeff * (self.prev_enemy_hp - curr_enemy_hp) - (self.prev_player_hp - curr_player_hp)
            self.prev_player_hp = curr_player_hp
            self.prev_enemy_hp = curr_enemy_hp
            tmp_done = False

        # When reset_round flag is set to False (never reset), the session should always keep going.
        if not self.reset_round:
            tmp_done = False
             
        # Max reward is 6 * full_hp = 1054 (damage * 3 + winning_reward * 3) norm_coefficient = 0.001
        return t_reward, tmp_done

    def render(self):
        self.env.render()
        time.sleep(0.01)

    def close(self):
        self.env.close()








