from stable_baselines3.common.callbacks import CheckpointCallback, ProgressBarCallback
from stable_baselines3.common.callbacks import BaseCallback
import os
import sys

class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)

    def _on_step(self) -> bool:

        if self.n_calls % 1000 == 0:
            print(f"Step: {self.n_calls}, Timesteps: {self.num_timesteps}, Reward: {self.locals['rewards']}")
        return True