import gym
from gym import spaces
import numpy as np
import types


class MPSPEnv(gym.Env):
    """Environment for the Multi Port Shipping Problem"""

    def __init__(self, rows, columns, n_ports):
        super(MPSPEnv, self).__init__()
        self.R = rows
        self.C = columns
        self.N = n_ports
        self.capacity = self.R * self.C
        self.terminated_reward = 0
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
            high=np.iinfo(np.int32).max,
            shape=(self.N, self.N),
            dtype=np.int32
        )
        self.observation_space = spaces.Dict({
            'bay_matrix': bay_matrix_def,
            'transportation_matrix': transportation_matrix_def
        })
        self.transportation_matrix = None
        self.bay_matrix = None
        self.column_counts = None
        self.port = None
        self.is_terminated = False
        self.virtual_R = None
        self.virtual_C = None

    def set_virtual_dimensions(self, virtual_R, virtual_C):
        """Limits the number of rows and columns that are accessible to the agent"""
        assert virtual_R < self.R, "Virtual R must be smaller than R"
        assert virtual_C < self.C, "Virtual C must be smaller than C"
        assert virtual_R > 0, "Virtual R must be strictly positive"
        assert virtual_C > 0, "Virtual C must be strictly positive"
        self.virtual_R = virtual_R
        self.virtual_C = virtual_C

    def reset(self, transportation_matrix=None, seed=None):
        """Reset the state of the environment to an initial state"""
        super().reset(seed=seed)
        self.transportation_matrix = self._get_short_distance_transportation_matrix(
            self.N
        ) if transportation_matrix is None else transportation_matrix
        self.bay_matrix = np.zeros((self.R, self.C), dtype=np.int32)
        self.column_counts = np.zeros(self.C, dtype=np.int32)
        self.port = 0
        self.is_terminated = False

        return self._get_observation()

    def step(self, action):
        """Execute one time step within the environment

        Args:
            action (int): The action to be executed
            The first C actions are for adding containers
            The last C actions are for removing containers
        """
        assert not self.is_terminated, "Environment is terminated"

        should_add = action < self.C
        reward = 0

        if should_add:
            j = action
            i = self.R - self.column_counts[j] - 1

            assert self.column_counts[j] < self.R, "Cannot add containers to full columns"

            reward += self._add_container(i, j)
        else:
            j = action - self.C
            i = self.R - self.column_counts[j]

            assert self.column_counts[
                action - self.C
            ] > 0, "Cannot remove containers from empty columns"

            reward += self._remove_container(i, j)

        # Port is zero indexed
        self.is_terminated = self.port+1 == self.N

        if self.is_terminated:
            reward += self.terminated_reward

        info = self._get_masks()

        return (
            self._get_observation(),
            reward,
            self.is_terminated,
            info
        )

    def get_action_mask(self):
        return self._get_masks()["mask"]

    def close(self):
        pass

    def print(self):
        """Prints the environment to the console"""
        print(f'Port: {self.port}')
        print('Bay matrix:')
        print(self.bay_matrix)
        print('Transportation matrix:')
        print(self.transportation_matrix)

    def _get_last_destination_container(self, should_print=False):

        container = -1
        for h in range(self.N-1, self.port, -1):
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

        assert container != -1, "No containers to offload"

        # Update state
        self.bay_matrix[i, j] = container
        self.transportation_matrix[self.port, container] -= 1

        # Sail along for every zero-row
        while np.sum(self.transportation_matrix[self.port]) == 0:
            self.port += 1
            blocking_containers = self._offload_containers()
            delta_reward -= blocking_containers
            if self.port + 1 == self.N:
                break

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
        """Returns the masks for the actions"""

        # Masking out full columns
        add_mask = (
            self.column_counts < self.R
            if self.virtual_R is None
            else self.column_counts < self.virtual_R
        )

        if self.virtual_C is not None:
            # Masking out columns that are not accessible
            add_mask = np.logical_and(
                add_mask,
                # Can only use first virtual_C columns
                np.arange(self.C) < self.virtual_C
            )

        # Masking out empty columns
        remove_mask = self.column_counts > 0

        mask = np.concatenate((add_mask, remove_mask), dtype=np.int8)

        return {
            "add_mask": add_mask,
            "remove_mask": remove_mask,
            "mask": mask
        }

    def _get_observation(self):
        return {
            'bay_matrix': self.bay_matrix,
            'transportation_matrix': self.transportation_matrix
        }

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

        # Make sure the first row of the transportation matrix has containers
        # Otherwise you could have skipped the first port
        if np.sum(output[0]) == 0:
            return self._get_short_distance_transportation_matrix(N)
        else:
            return output


def MPSPObservationWrapper(env):
    """Observation wrapper (flattening) for the MPSPEnv environment."""
    env.observation_space = spaces.Box(
        low=0,
        high=np.iinfo(np.int32).max,
        shape=(env.R * env.C + env.N * env.N,),
        dtype=np.int32
    )

    def _new_get_observation(self):
        return np.concatenate(
            (
                self.bay_matrix.flatten(),
                self.transportation_matrix.flatten()
            )
        )

    env._get_observation = types.MethodType(_new_get_observation, env)

    return env
