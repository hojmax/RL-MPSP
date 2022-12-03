import numpy as np


class ReplayBuffer:
    def __init__(self, mem_size, observation_space, batch_size):
        self.mem_count = 0
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.states = np.zeros((mem_size, observation_space), dtype=np.float32)
        self.actions = np.zeros(mem_size, dtype=np.int64)
        self.rewards = np.zeros(mem_size, dtype=np.float32)
        self.states_ = np.zeros((mem_size, observation_space), dtype=np.float32)
        self.dones = np.zeros(mem_size, dtype=bool)

    def add(self, state, action, reward, state_, done):
        mem_index = self.mem_count % self.mem_size

        self.states[mem_index] = state
        self.actions[mem_index] = action
        self.rewards[mem_index] = reward
        self.states_[mem_index] = state_
        self.dones[mem_index] = 1 - done

        self.mem_count += 1

    def sample(self):
        mem_max = min(self.mem_count, self.mem_size)
        batch_indices = np.random.choice(
            mem_max,
            self.batch_size,
            replace=True
        )

        states = self.states[batch_indices]
        actions = self.actions[batch_indices]
        rewards = self.rewards[batch_indices]
        states_ = self.states_[batch_indices]
        dones = self.dones[batch_indices]

        return states, actions, rewards, states_, dones
