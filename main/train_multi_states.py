import os

import retro
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, ProgressBarCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecTransposeImage
from mk_wrapper import MkWrapper
from custom_callback import CustomCallback


NUM_ENV = 16
LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)

# Linear scheduler
def linear_schedule(initial_value, final_value=0.0):

    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert (initial_value > 0.0)

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler

def create_env(game, state, seed):
    def _init():
        env = retro.make(
            game=game, 
            state=state,
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE
        )
        env = MkWrapper(env)
        env = Monitor(env)
        env.seed(seed)
        return env
    return _init


if __name__ == "__main__":
    game = 'MortalKombatII-Genesis'
    state0 = 'Level1.LiuKangVsJax'
    state1 = 'Level1.LiuKangVsBaraka'
    state2 = 'Level1.LiuKangVsScorpion'
    state3 = 'Level1.LiuKangVsSubZero'

    state = state1

    # deummy_env = DummyVecEnv([create_env(game, state, seed = 0)])
    # model = PPO('CnnPolicy', deummy_env, verbose=1)

    # print(model.policy)


    # Create the environment
    env = SubprocVecEnv([create_env(game, state, seed = i) for i in range(NUM_ENV)])
    env = VecTransposeImage(env)
    
    # Create the evaluation environment
    eval_env = DummyVecEnv([create_env(game, state, seed = NUM_ENV)])
    eval_env = VecTransposeImage(eval_env)

    # # set linear schedule for LR TODO: can be adjust
    # lr_schedule = linear_schedule(2.5e-4, 2.5e-5)
    lr_schedule = linear_schedule(2.5e-4, 2.5e-6)

    # set linear scheduler for clip range
    clip_range_schedule = linear_schedule(0.15, 0.025)


    # # load the pretrained model print error if the model is not found
    # try:
    #     model = PPO.load(os.path.join('models', 'final_cuda_pow_rew_best_test_model.zip'), env=env)
    #     print("Model loaded successfully!")
    # except:
    #     print("Model not found! Training a new model...")
    #     model = None
    #     # terminate the program
    #     exit()


    pretrained_model_path = os.path.join('models', 'mk_cuda_Jax&Baraka_1000000_steps.zip')
    # load the pretrained model  or create a new one if not found
    if os.path.exists(pretrained_model_path):
        model = PPO.load(pretrained_model_path, env=env, device='cuda')
        print(f"Model loaded successfully from {pretrained_model_path}")
    else:
        print("Model not found! Training a new model...")
        # Create the model
        model = PPO(
            "CnnPolicy", 
            env,
            device="cuda", 
            verbose=1,
            n_steps=512,
            batch_size=512,
            n_epochs=4,
            gamma=0.94,
            learning_rate=lr_schedule,
            clip_range=clip_range_schedule,
            tensorboard_log=LOG_DIR
        )

    # save the model
    model_path = "models"
    model_path = os.path.join(model_path)
    os.makedirs(model_path, exist_ok=True)

    # Create the callback: check every 62500 steps
    save_frequency = 6250
    checkpoint_callback = CheckpointCallback(save_freq=save_frequency, save_path=model_path, name_prefix='mk_cuda_Jax&Baraka&scorpion')

    # progress bar
    progress_bar_callback = ProgressBarCallback()
    custom_callback = CustomCallback()
    
    # Eval callback: evaluate every 100000 steps and save the best model
    eval_frequency = 6250
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=model_path,
        eval_freq=eval_frequency
    )

    # Train the model
    # set the number of steps to train the model
    steps = 1000000

    model.learn(
        total_timesteps=int(steps),
        callback=[checkpoint_callback, progress_bar_callback, custom_callback, eval_callback]
    )
    env.close()


    # # write the training logs from stdout to a file
    # original_stdout = sys.stdout
    # log_file_path = os.path.join(LOG_DIR, 'training.log')
    # with open(log_file_path, 'w') as log_file:
    #     sys.stdout = log_file

    #     # Train the model
    #     model.learn(
    #         total_timesteps=int(steps),
    #         callback=[checkpoint_callback, progress_bar_callback, custom_callback]
    #     )
    #     env.close()

    # # restore stdout
    # sys.stdout = original_stdout

    # Save the final model
    model_file_name = 'final_cuda_jax&baraka&scorpion.zip'
    model.save(os.path.join(model_path, model_file_name))
    print("Training completed!")
    print(f"Final model saved to {os.path.join(model_path, model_file_name)}")

    # # Test the model
    # obs = env.reset()
    # for _ in range(1000):
    #     actions, _ = model.predict(obs, deterministic=True)
    #     obs, _, _, _ = env.step(actions)
    #     env.render()
    # env.close()