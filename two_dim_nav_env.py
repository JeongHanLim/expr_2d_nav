import gym
import numpy as np

class TwoDimNavEnv(gym.Env):
    def __init__(self, goal):
        self.goal_x = goal[0]
        self.goal_y = goal[1]
        self.start_x = 0
        self.start_y = 0
        self.pos_x = self.start_x
        self.pos_y = self.start_y
        self.timesteps = 0
        self.grid_size = 256
        self.prev_distance = 0
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(3, ))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2, ))

    def step(self, action):
        self.timesteps += 1
        self.pos_x = np.clip(self.pos_x + action[0], a_min=-self.grid_size, a_max=self.grid_size)
        self.pos_y = np.clip(self.pos_y + action[1], a_min=-self.grid_size, a_max=self.grid_size)
        reward = 0
        done = False
        if self._distance < 1:
            done = True
            reward += 100
        if self.timesteps > 300:
            done = True
            reward += -100

        state = np.array([self.pos_x, self.pos_y, self._distance])
        reward += self.prev_distance - self._distance
        self.prev_distance = self._distance

        info = {'timesteps': self.timesteps}

        return state, reward, done, info

    def reset(self):
        self.pos_x = self.start_x
        self.pos_y = self.start_y
        self.prev_distance = self._distance
        state = np.array([self.pos_x, self.pos_y, self._distance])
        self.timesteps = 0
        return state

    def render(self, mode='human'):
        pass

    @property
    def _distance(self):
        return abs(self.goal_x - self.pos_x) + abs(self.goal_y - self.pos_y)


class PartialTwoDimNavEnv(TwoDimNavEnv):
    def step(self, action):
        state, reward, done, info = super().step(action)
        rand_var = np.random.random()
        if rand_var < 0.5:
            state[0] = 0
        else:
            state[1] = 0
        info = {'timesteps': self.timesteps}

        return state, reward, done, info

    def reset(self):
        state = super().reset()
        rand_var = np.random.random()
        if rand_var < 0.5:
            state[0] = 0
        else:
            state[1] = 0
        return state
