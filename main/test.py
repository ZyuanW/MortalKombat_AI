import os
import time 

import retro
from stable_baselines3 import PPO

from mk_wrapper import MkWrapper

RESET_ROUND = True  # Whether to reset the round when fight is over. 
RENDERING = True    # Whether to render the game screen.

MODEL_NAME = r"mk_cuda_Jax&Baraka&scorpion_1000000_steps" # Specify the model file to load.

RANDOM_ACTION = False
NUM_EPISODES = 5
MODEL_DIR = r"./models/"

def make_env(game, state):
    def _init():
        env = retro.make(
            game=game, 
            state=state, 
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE
        )
        env = MkWrapper(env, reset_round=RESET_ROUND, rendering=RENDERING)
        return env
    return _init

game = "MortalKombatII-Genesis"

state0 = 'Level1.LiuKangVsJax'
state1 = 'Level1.LiuKangVsBaraka'
state2 = 'Level1.LiuKangVsScorpion'
state3 = 'Level1.LiuKangVsSubZero'

env = make_env(game, state=state3)()

if not RANDOM_ACTION:
    print(os.path.join(MODEL_DIR, MODEL_NAME))
    model = PPO.load(os.path.join(MODEL_DIR, MODEL_NAME), env=env)

obs = env.reset()
done = False

num_episodes = NUM_EPISODES
episode_reward_sum = 0
num_victory = 0

print("\nFighting Begins!\n")

for _ in range(num_episodes):
    done = False
    
    if RESET_ROUND:
        obs = env.reset()

    total_reward = 0

    while not done:
        timestamp = time.time()

        if RANDOM_ACTION:
            obs, reward, done, info = env.step(env.action_space.sample())
        else:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)

        if reward != 0:
            total_reward += reward
            print("Reward: {:.3f}, playerHP: {}, enemyHP:{}".format(reward, info['health'], info['enemy_health']))
        
        if info['enemy_health'] <= 0 or info['health'] <= 0:
            done = True

    if info['enemy_health'] <= 0:
        print("Victory!")
        num_victory += 1

    print("Total reward: {}\n".format(total_reward))
    episode_reward_sum += total_reward

    if not RESET_ROUND:
        while info['enemy_health'] <= 0 or info['health'] <= 0:
        # Inter scene transition. Do nothing.
            obs, reward, done, info = env.step([0] * 12)
            env.render()

env.close()
print("Winning rate: {}".format(1.0 * num_victory / num_episodes))
if RANDOM_ACTION:
    print("Average reward for random action: {}".format(episode_reward_sum/num_episodes))
else:
    print("Average reward for {}: {}".format(MODEL_NAME, episode_reward_sum/num_episodes))