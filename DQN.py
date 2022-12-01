import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DQN(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size_1, hidden_size_2, learning_rate):
        super().__init__()
        self.input_shape = input_size
        self.action_space = output_size

        self.fc1 = nn.Linear(self.input_shape, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, self.action_space)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()

        device = torch.device(
            "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        )
        self.to(device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
