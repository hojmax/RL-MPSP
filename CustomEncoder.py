from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch


class Transpose(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mT


class TransportationEncoder(nn.Module):
    def __init__(self, n_ports, hidden_size, device="cpu"):
        super().__init__()
        self.n_ports = n_ports
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(n_ports, hidden_size, device=device, batch_first=True)
        self.device = device

    def forward(self, x):
        x = x.to(self.device).float()

        # Pass through RNN
        # Hidden defaults to zero
        _, hidden = self.rnn(x)

        # Hidden has shape (1, batch_size, hidden_size)
        return hidden.squeeze(0)


class ToLong(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.long()


class ToFloat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.float()


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        n_ports,
        container_embedding_size,
        internal_hidden,
        output_hidden,
        device="cpu",
    ):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)
        extractors = {}
        self.Container_embedding = nn.Embedding(
            n_ports, container_embedding_size, device=device
        )

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "bay_matrix":
                rows = subspace.shape[0]
                cols = subspace.shape[1]
                extractors[key] = nn.Sequential(
                    # Long is required for embedding
                    ToFloat(),
                    # We want to encode columns, not rows
                    Transpose(),
                    nn.Flatten(2),
                    nn.Linear(rows, internal_hidden, device=device),
                    nn.Tanh(),
                    nn.Flatten(),
                )
                total_concat_size += cols * internal_hidden
            elif key == "container":
                extractors[key] = nn.Sequential(
                    # Long is required for embedding
                    ToFloat(),
                )
                total_concat_size += 1
            elif key == "transportation_matrix":
                extractors[key] = TransportationEncoder(
                    n_ports, internal_hidden, device=device
                )
                total_concat_size += internal_hidden
            elif key == "will_block":
                extractors[key] = nn.Sequential(
                    ToFloat(),
                    nn.Linear(subspace.shape[0], internal_hidden, device=device),
                    nn.Tanh(),
                )
                total_concat_size += internal_hidden

        self.extractors = nn.ModuleDict(extractors)

        self.final_layer = nn.Sequential(
            nn.Linear(total_concat_size, output_hidden, device=device),
            nn.Tanh(),
        )

        # Update the features dim manually
        self._features_dim = output_hidden

        self.device = device

    def forward(self, observations):
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():

            # We are given a (Batch, Height, Width) PyTorch tensor
            encoded_tensor_list.append(
                extractor(observations[key].to(self.device)).to(self.device)
            )

        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        output = torch.cat(encoded_tensor_list, dim=1)
        output = self.final_layer(output)
        return output
