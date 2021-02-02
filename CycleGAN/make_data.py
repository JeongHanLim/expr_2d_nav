import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import pickle as pkl
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import os

from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from two_dim_nav_env import TwoDimNavEnv
from expr_manage import ExperimentManager

env_setting = lambda goal: TwoDimNavEnv(goal=goal)
model_setting = lambda env: PPO2(MlpPolicy, env, n_steps=1024, tensorboard_log=os.path.join(manager.sub_path), full_tensorboard_log=True, verbose=1)
manager = ExperimentManager('D:/2021paper_data', 'base_model', 'base_model_1')

if __name__ == '__main__':
    with open(os.path.join(manager.sub_path, 'setting.pkl'), 'rb') as f:
        goal = pkl.load(f)
    print('goal setting: {}'.format(goal))
    env = env_setting(goal)
    model = PPO2.load(os.path.join(manager.sub_path, 'model.zip'))
    dataset = []
    save = False
    while True:
        done = False
        state = env.reset()
        while not done:
            dataset.append(state)
            action, _ = model.predict(state)
            state, reward, done, info = env.step(action)
            if len(dataset) == 500000:
                with open('./dataset/1_state.pkl', 'wb') as f:
                    pkl.dump(dataset, f)
                    save = True
        print('\r temp datset length is {}'.format(len(dataset)))
        if save:
            break