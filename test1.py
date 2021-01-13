from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
import gym
from reptile_callback import LowCallback, reptile

ALPHA = 0.05
env_setting = lambda: gym.make('CartPole-v0')
model_setting = lambda env: PPO2(MlpPolicy, env)

if __name__ == '__main__':
    env = env_setting()
    model = model_setting(env)
    algo = reptile(1, ALPHA, model=model, env=env)
    algo.test()