import numpy as np
import torch
import random


class DQN_Solver:
    def __init__(self, ReplayBuffer, DQN, batch_size, exploration_max, gamma, exploration_decay, exploration_min):
        self.memory = ReplayBuffer
        self.batch_size = batch_size
        self.gamma = gamma
        self.exploration_decay = exploration_decay
        self.exploration_min = exploration_min
        self.exploration_rate = exploration_max
        self.network = DQN
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def choose_action(self, observation, mask, env):
        if random.random() < self.exploration_rate:
            return env.action_space.sample(mask)

        state = torch.tensor(observation).float().detach()
        state = state.to(self.device)
        state = state.unsqueeze(0)
        q_values = self.network(state).detach()
        q_max = q_values.abs().max()
        masked_argmax = (
            q_values - 2 * q_max * (1 - mask)
        ).argmax()
        return masked_argmax.item()

    def learn(self):
        if self.memory.mem_count < self.batch_size:
            return

        states, actions, rewards, states_, dones = self.memory.sample()
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        states_ = torch.tensor(states_, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)
        batch_indices = np.arange(self.batch_size, dtype=np.int64)

        q_values = self.network(states)
        next_q_values = self.network(states_)

        predicted_value_of_now = q_values[batch_indices, actions]
        predicted_value_of_future = torch.max(next_q_values, dim=1)[0]

        q_target = rewards + self.gamma * predicted_value_of_future * dones

        loss = self.network.loss(q_target, predicted_value_of_now)
        self.network.optimizer.zero_grad()
        loss.backward()
        self.network.optimizer.step()

        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(
            self.exploration_min, self.exploration_rate)

    def returning_epsilon(self):
        return self.exploration_rate
