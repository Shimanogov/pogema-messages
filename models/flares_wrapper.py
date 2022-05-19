import gym
import numpy as np


class FlareWrapper(gym.Wrapper):
    def __init__(self, env, flares_num=2):
        super().__init__(env)
        self.env = env

        self.flares_num = flares_num
        self.flares_grid = None
        self.full_size = self.config.obs_radius * 2 + 1

        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(3 + self.flares_num,
                                                                 self.full_size,
                                                                 self.full_size))

        self.action_space = (gym.spaces.Discrete(len(self.config.MOVES)),)
        self.action_space += flares_num * (gym.spaces.Discrete(2),)
        self.action_space = gym.spaces.Tuple(self.action_space)

    def reset(self):
        grid_obs = self.env.reset()
        size = self.grid.positions.shape[1]
        self.flares_grid = np.zeros((self.flares_num, size, size))
        return [np.concatenate((i, np.zeros((self.flares_num,
                                             self.full_size,
                                             self.full_size))),
                               axis=0)
                for i in grid_obs]

    def step(self, actions):
        obs, reward, done, info = self.env.step([a[0] for a in actions])
        flares = [a[1:] for a in actions]
        size = self.grid.positions.shape[1]
        self.flares_grid = np.zeros((self.flares_num, size, size))
        for agent_id, flare in enumerate(flares):
            for color_id, color in enumerate(flare):
                x, y = self.grid.positions_xy[agent_id]
                self.flares_grid[color_id][x][y] = color
        r = self.config.obs_radius
        for i in range(self.get_num_agents()):
            x, y = self.grid.positions_xy[i]
            obs[i] = np.concatenate((obs[i], self.flares_grid[:, x - r:x + r + 1, y - r:y + r + 1]))
        return obs, reward, done, info
