import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import os
import argparse

from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from two_dim_nav_env import TwoDimNavEnv, PartialTwoDimNavEnv
from expr_manage import ExperimentManager

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='experiment setting')
    parser.add_argument('--path', type=str)
    parser.add_argument('--midclass',  type=str)
    parser.add_argument('--subclass',  type=str)
    parser.add_argument('--total_timesteps',  type=int, default=1000000)
    parser.add_argument('--pomdp', dest='pomdp', action='store_true')
    parser.add_argument('--mdp', dest='pomdp', action='store_false')
    parser.set_defaults(pomdp=False)
    args = parser.parse_args()

    manager = ExperimentManager(args.path, args.midclass, args.subclass)
    manager.make_description(args.description)

    np.random.seed()
    goal = np.random.random((2,)) * 512 - 256

    if args.pomdp:
        env = PartialTwoDimNavEnv(goal)
    else:
        env = TwoDimNavEnv(goal)

    model =PPO2(MlpPolicy, env, tensorboard_log=manager.sub_path, full_tensorboard_log=True)
    model.learn(total_timesteps=args.total_timesteps)
    model.save(os.path.join(manager.sub_path, 'model'))