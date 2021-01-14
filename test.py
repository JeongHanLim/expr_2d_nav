import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import os

from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from two_dim_nav_env import TwoDimNavEnv
from reptile_callback import LowCallback, reptile
from expr_manage import ExperimentManager

env_setting = lambda goal: TwoDimNavEnv(goal=goal)
model_setting = lambda env, expr_num: PPO2(MlpPolicy, env, tensorboard_log=os.path.join(manager.sub_path, str(expr_num)), full_tensorboard_log=True)
manager = ExperimentManager('D:/2021paper_data', 'reptile', 'testexpr')

if __name__ == '__main__':
    goal = np.random.random((2,)) * 512 - 256
    print('goal setting: {}'.format(goal))
    env = env_setting(goal)
    model = PPO2.load(os.path.join(manager.sub_path, 'model.zip'))
    model.learn(total_timesteps=300000)