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
        self.capacity = self.R * self.C
        # You can add or remove a container for every column
        self.action_space = spaces.Discrete(2 * self.C)
        bay_matrix_def = spaces.Box(
            low=0,
            high=self.N,
            shape=(self.R, self.C),
            dtype=np.int32
        )
        transportation_matrix_def = spaces.Box(
            low=0,
            high=float("inf"),
            shape=(self.N, self.N),
            dtype=np.int32
        )
        self.observation_space = spaces.Tuple(
            (bay_matrix_def, transportation_matrix_def)
        )
        self.transporation_matrix = None
        self.bay_matrix = None
        self.column_counts = None
        self.port = None

    def _get_last_destination_container(self):
        container = -1
        for h in range(self.C, self.port, -1):
            if self.bay_matrix[self.port, h] > 0:
                container = h
                break
        return container

    def step(self, action):
        # Execute one time step within the environment

        should_add = action < self.C
        if should_add:
            # Cannot add containers to full columns
            assert self.column_counts[action] < self.R

            # Add containers bottom up
            self.column_counts[j] += 1
            i = self.R - self.column_counts[action]
            j = action

            # Find last destination container
            container = self._get_last_destination_container()
            assert container != -1

            # Update state
            self.bay_matrix[i, j] = container
            self.transporation_matrix[self.port, container] -= 1
            sail_along = np.sum(self.transporation_matrix[self.port]) == 0
            if sail_along:
                self.port += 1

        reward = 0
        is_terminated = self.port == self.N
        info = self._get_masks()
        return (
            self._get_observation(),
            reward,
            is_terminated,
            info
        )

    def reset(self, seed=None):
        # Reset the state of the environment to an initial state
        self.seed(seed)
        self.transporation_matrix = self._get_short_distance_transportation_matrix(
            self.N
        )
        self.bay_matrix = np.zeros((self.R, self.C), dtype=np.int32)
        self.column_counts = np.zeros(self.C, dtype=np.int32)
        self.port = 0

        info = self._get_masks()
        return (
            self._get_observation(),
            info
        )

    def _get_masks(self):
        return {
            "add_mask": self.column_counts < self.R,
            "remove_mask": self.column_counts > 0
        }

    def _get_observation(self):
        return self.bay_matrix, self.transporation_matrix

    def print(self):
        # Prints the environment to the console
        pass

    def _get_short_distance_transportation_matrix(self, N):
        # Generates a feasible transportation matrix (short distance)
        output = np.zeros((N, N), dtype=np.int32)
        bay_capacity = self.capacity

        for i in range(N-1):
            for j in range(i+1, N):
                output[i, j] = np.random.randint(0, bay_capacity+1)
                bay_capacity -= output[i, j]

            # Offloaded at port
            for h in range(i+1):
                bay_capacity += output[h, i+1]

        return output
