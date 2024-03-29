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
        self.linear = nn.Linear(n_ports**2, hidden_size, device=device)
        self.device = device

    def forward(self, x):
        x = x.to(self.device).float()

        # Flatten the embedding dimension, keep batch and column
        x = x.flatten(1)

        x = self.linear(x)

        return x


class BayEncoder(nn.Module):
    def __init__(
        self,
        rows,
        internal_hidden,
        n_ports,
        device="cpu",
    ):
        super().__init__()
        self.model = nn.Sequential(
            # We want to encode columns, not rows
            Transpose(),
            # Flatten the embedding dimension, keep batch and column
            nn.Flatten(2),
            nn.Linear(rows, internal_hidden, device=device),
            nn.Tanh(),
            nn.Flatten(),
        )
        self.n_ports = n_ports
        self.device = device

    def forward(self, x):
        x = x.to(self.device).float()

        # Flatten the embedding dimension, keep batch and column
        x = x.flatten(1)

        return x


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
                extractors[key] = BayEncoder(
                    rows, internal_hidden, n_ports, device=device
                )
                total_concat_size += cols * rows
                # total_concat_size += cols * internal_hidden
            elif key == "transportation_matrix":
                extractors[key] = TransportationEncoder(
                    n_ports, internal_hidden, device=device
                )
                total_concat_size += internal_hidden
            elif key == "virtual_R":
                extractors[key] = nn.Sequential(
                    ToLong(),
                )
                total_concat_size += 1
            elif key == "virtual_C":
                extractors[key] = nn.Sequential(
                    ToLong(),
                )
                total_concat_size += 1
            elif key == "next_container":
                extractors[key] = nn.Sequential(
                    ToFloat(),
                )
                total_concat_size += 1
            elif key == "will_block":
                extractors[key] = nn.Sequential(
                    ToFloat(),
                )
                total_concat_size += subspace.shape[0]

        self.extractors = nn.ModuleDict(extractors)

        self.final_layer = nn.Sequential(
            nn.Linear(total_concat_size, output_hidden, device=device),
            nn.Tanh(),
            nn.Linear(output_hidden, output_hidden, device=device),
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
                extractor(
                    observations[key].to(self.device),
                ).to(self.device)
            )

        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        output = torch.cat(encoded_tensor_list, dim=1)
        output = self.final_layer(output)
        return output
