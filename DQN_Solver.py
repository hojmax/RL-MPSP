import numpy as np
import torch
import random
from IPython.display import clear_output


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
            "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        )
        self.training = True

    def train(self):
        self.network.train()
        self.training = True

    def eval(self):
        self.network.eval()
        self.training = False

    def choose_action(self, observation, mask, env):
        if self.training and random.random() < self.exploration_rate:
            return env.action_space.sample(mask), None

        state = torch.tensor(observation).float().detach()
        state = state.to(self.device)
        state = state.unsqueeze(0)
        q_values = self.network(state).detach()
        q_max = q_values.abs().max()
        mask = torch.tensor(mask).to(self.device)
        masked_argmax = (
            # 3 is arbitrary, just needs to be strictly greater than 2
            q_values - 3 * q_max * (1 - mask)
        ).argmax()
        return masked_argmax.item(), q_values

    def learn(self, should_print=False):
        if self.memory.mem_count < self.batch_size:
            return 0

        states, actions, rewards, states_, dones = self.memory.sample()
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        states_ = torch.tensor(states_, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)
        batch_indices = np.arange(self.batch_size, dtype=np.int64)
        q_values = self.network(states)
        next_q_values = self.network(states_)

        predicted_value_of_now = q_values[batch_indices, actions]
        predicted_value_of_future = torch.max(next_q_values, dim=1)[0]

        q_target = rewards #+ self.gamma * predicted_value_of_future * dones
        if should_print:
            clear_output(wait=True)
            print("Rewards:")
            print(rewards)
            print('Target:')
            print(q_target)
            print('Predicted:')
            print(predicted_value_of_now)
        loss = self.network.loss(q_target, predicted_value_of_now)
        self.network.optimizer.zero_grad()
        loss.backward()
        self.network.optimizer.step()

        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(
            self.exploration_min,
            self.exploration_rate
        )

        return loss.item()

    def returning_epsilon(self):
        return self.exploration_rate
