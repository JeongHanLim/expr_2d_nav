import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import os
import argparse

from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from multiprocessing import Process
from two_dim_nav_env import TwoDimNavEnv
from reptile_callback import LowCallback, reptile
from expr_manage import ExperimentManager

def run(oper_num, args):
    goal = np.random.random((2,)) * 512 - 256
    print('env {} goal setting: {}'.format(oper_num, goal))
    env = env_setting(goal)
    model = model_setting(env, oper_num)
    callback = LowCallback(oper_num)
    model.learn(total_timesteps=args.total_timesteps, callback=callback)
    print('finish')

def reptile_run(args):
    goal = np.random.random((2,)) * 512 - 256
    env = env_setting(goal)
    model = model_setting(env, args.num_workers)
    algo = reptile(num_of_operator=args.num_workers, alpha=args.alpha, model=model, env=env)
    algo.run()
    algo.save(os.path.join(manager.sub_path, 'model'))
    print('finish')
    algo.test()
    algo.adapt(args.adapt_timesteps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='experiment setting')
    parser.add_argument('--path', '-p', type=str)
    parser.add_argument('--midclass', '-mc', type=str)
    parser.add_argument('--subclass', '-sc', type=str)
    parser.add_argument('--description', '-des', type=str)
    parser.add_argument('--num_workers', '-n', type=int, default=4)
    parser.add_argument('--total_timesteps', '-tt', type=int, default=1000000)
    parser.add_argument('--adapt_timesteps', '-at', type=int, default=300000)
    parser.add_argument('--alpha', '-a', type=float, default=0.25)
    args = parser.parse_args()

    manager = ExperimentManager(args.path, args.midclass, args.subclass)
    manager.make_description(args.description)
    env_setting = lambda goal: TwoDimNavEnv(goal=goal)
    model_setting = lambda env, expr_num: PPO2(MlpPolicy, env,
                                               tensorboard_log=os.path.join(manager.sub_path, str(expr_num)),
                                               full_tensorboard_log=True)

    p_list = []
    for i in range(args.num_workers + 1):
        if not os.path.isdir(os.path.join(manager.sub_path, str(i))):
            os.mkdir(os.path.join(manager.sub_path, str(i)))
    for i in range(args.num_workers):
        p = Process(target=run, args=(i, args, ))
        p.start()
        p_list.append(p)
        print('make process')
    p = Process(target=reptile_run, args=(args, ))
    p.start()
    p_list.append(p)
    print('make reptile')
    for proc in p_list:
        proc.join()
