import warnings
warnings.filterwarnings('ignore')
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from multiprocessing import Process
import gym
from reptile_callback import LowCallback, reptile

ALPHA = 0.05
env_setting = lambda: gym.make('CartPole-v0')
model_setting = lambda env: PPO2(MlpPolicy, env)
path = './model'

def run(oper_num):
    env = env_setting()
    model = model_setting(env)
    callback = LowCallback(oper_num)
    model.learn(total_timesteps=300000, callback=callback)
    print('finish')

def reptile_run(num_oper):
    env = env_setting()
    model = model_setting(env)
    algo = reptile(num_of_operator=num_oper, alpha=ALPHA, model=model, env=env)
    algo.run()
    algo.save(path)
    print('finish')
    algo.test()

if __name__ == '__main__':
    p_list = []
    for i in range(4):
        p = Process(target=run, args=(i, ))
        p.start()
        p_list.append(p)
        print('make process')
    p = Process(target=reptile_run, args=(4, ))
    p.start()
    p_list.append(p)
    print('make reptile')
    for proc in p_list:
        proc.join()