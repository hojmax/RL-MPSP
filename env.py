import gym
from gym import spaces
import numpy as np


class MPSPEnv(gym.Env):
    """Environment for the Multi Port Shipping Problem"""

    def __init__(self, rows, columns, n_ports):
        super(MPSPEnv, self).__init__()
        self.R = rows
        self.C = columns
        self.N = n_ports
        # You can add or remove a container for every column
        self.action_space = spaces.Discrete(2 * self.C)
        # The observation space is (bay_matrix, transportation_matrix)
        self.observation_space = spaces.Tuple((
            spaces.Box(
                low=0,
                high=self.N,
                shape=(self.R, self.C),
                dtype=np.int32
            ),
            spaces.Box(
                low=0,
                high=float("inf"),
                shape=(self.N, self.N),
                dtype=np.int32
            )
        ))

    def step(self, action):
        # Execute one time step within the environment
        pass

    def reset(self):
        # Reset the state of the environment to an initial state
        pass

    def render(self):
        # Render the environment to the screen
        pass

    def _is_feasible(self, transportation_matrix):
        # Check if the transportation matrix is feasible
        pass
