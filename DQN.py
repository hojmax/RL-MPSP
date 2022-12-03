import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DQN(torch.nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        n_layers,
        learning_rate,
        adam_epsilon
    ):
        super().__init__()
        self.input_shape = input_size
        self.action_space = output_size

        layers = [
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        ]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_size, output_size))

        self.model = nn.Sequential(*layers)

        self.optimizer = optim.Adam(
            self.parameters(),
            lr=learning_rate,
            eps=adam_epsilon
        )
        self.loss = nn.MSELoss()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        )
        self.to(self.device)

    def forward(self, x):
        return self.model(x)

    def load(self, path):
        self.load_state_dict(
            torch.load(path, map_location=self.device)
        )
