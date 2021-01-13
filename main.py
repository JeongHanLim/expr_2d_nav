from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
import gym_nav2d
import gym


if __name__ == "__main__":
    env = gym.make('nav2dmdpgoal-v0')
    env.goal_setting([127, 127])
    model = PPO2(MlpPolicy, env, n_steps=1024, verbose=1, tensorboard_log='./', full_tensorboard_log=True)
    model.learn(1000000, log_interval=100)