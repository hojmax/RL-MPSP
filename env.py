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
        self.transportation_matrix = None
        self.bay_matrix = None
        self.column_counts = None
        self.port = None

    def seed(self, seed=None):
        np.random.seed(seed)

    def reset(self, seed=None):
        """Reset the state of the environment to an initial state"""
        self.seed(seed)
        self.transportation_matrix = self._get_short_distance_transportation_matrix(
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

    def step(self, action):
        """Execute one time step within the environment
        
        Args:
            action (int): The action to be executed
            The first C actions are for adding containers
            The last C actions are for removing containers
        """

        should_add = action < self.C
        reward = 0

        if should_add:
            j = action
            i = self.R - self.column_counts[j] - 1

            # Cannot add containers to full columns
            assert self.column_counts[j] < self.R

            reward += self._add_container(i, j)
        else:
            j = action - self.C
            i = self.R - self.column_counts[j]

            # Cannot remove containers from empty columns
            assert self.column_counts[action - self.C] > 0

            reward += self._remove_container(i, j)

        # Port is zero indexed
        is_terminated = self.port+1 == self.N
        info = self._get_masks()
        return (
            self._get_observation(),
            reward,
            is_terminated,
            info
        )

    def print(self):
        """Prints the environment to the console"""
        print("Port: {}".format(self.port))
        print("Bay matrix:")
        print(self.bay_matrix)
        print("Transportation matrix:")
        print(self.transportation_matrix)

    def _get_last_destination_container(self):

        container = -1
        for h in range(self.C-1, self.port, -1):
            if self.transportation_matrix[self.port, h] > 0:
                container = h
                break

        return container

    def _remove_container(self, i, j):
        """Removes container from bay and returns delta reward"""

        # Update state
        container = self.bay_matrix[i, j]
        self.bay_matrix[i, j] = 0
        self.transportation_matrix[self.port, container] += 1
        self.column_counts[j] -= 1

        # Penalize shifting containers
        return -1

    def _add_container(self, i, j):
        """Adds container to bay and returns delta reward"""

        delta_reward = 0
        self.column_counts[j] += 1

        # Find last destination container
        container = self._get_last_destination_container()
        assert container != -1

        # Update state
        self.bay_matrix[i, j] = container
        self.transportation_matrix[self.port, container] -= 1

        # Sail along for every zero-row
        while np.sum(self.transportation_matrix[self.port]) == 0:
            self.port += 1
            if self.port + 1 == self.N:
                break
            blocking_containers = self._offload_containers()
            delta_reward -= blocking_containers

        return delta_reward

    def _offload_containers(self):
        """Offloads containers to the port, updates the transportation matrix and returns the number of shifts"""
        blocking_containers = 0

        for j in range(self.C):
            off_loading_column = False
            for i in range(self.R-1, -1, -1):
                if self.bay_matrix[i, j] == 0:
                    break

                if self.bay_matrix[i, j] == self.port:
                    off_loading_column = True

                if off_loading_column:
                    if self.bay_matrix[i, j] != self.port:
                        blocking_containers += 1
                        self.transportation_matrix[
                            self.port,
                            self.bay_matrix[i, j]
                        ] += 1

                    self.bay_matrix[i, j] = 0
                    self.column_counts[j] -= 1

        return blocking_containers

    def _get_masks(self):
        return {
            "add_mask": self.column_counts < self.R,
            "remove_mask": self.column_counts > 0
        }

    def _get_observation(self):
        return self.bay_matrix, self.transportation_matrix

    def _get_short_distance_transportation_matrix(self, N):
        """Generates a feasible transportation matrix (short distance)"""
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
