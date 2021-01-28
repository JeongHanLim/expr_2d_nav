import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import os
import pickle as pkl

from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from two_dim_nav_env import TwoDimNavEnv
from reptile_callback import LowCallback, reptile
from expr_manage import ExperimentManager

env_setting = lambda goal: TwoDimNavEnv(goal=goal)
model_setting = lambda env: PPO2(MlpPolicy, env, n_steps=1024, tensorboard_log=os.path.join(manager.sub_path), full_tensorboard_log=True, verbose=1)
manager = ExperimentManager('D:/2021paper_data', 'base_model', 'base_model_2')

if __name__ == '__main__':
    goal = np.random.random((2,)) * 512 - 256
    print('goal setting: {}'.format(goal))
    env = env_setting(goal)
    model = model_setting(env)
    # model = PPO2.load(os.path.join(manager.sub_path, 'model.zip'))
    model.learn(total_timesteps=1000000)
    model.save(os.path.join(manager.sub_path, 'model'))
    with open(os.path.join(manager.sub_path, 'setting.pkl'), 'wb') as f:
        pkl.dump(goal, f)