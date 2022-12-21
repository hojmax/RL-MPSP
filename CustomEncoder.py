from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch

class Transpose(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mT


class TransportationEncoder(nn.Module):
    def __init__(
        self,
        Container_embedding,
        container_embedding_size,
        n_ports,
        hidden_size
    ):
        super().__init__()
        self.Container_embedding = Container_embedding
        self.n_ports = n_ports
        self.linear1 = nn.Linear(
            n_ports,
            hidden_size
        )
        self.linear2 = nn.Linear(
            container_embedding_size + hidden_size,
            hidden_size
        )
        self.flatten = nn.Flatten()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = x.float()
        batch_size = x.shape[0]
        # Positional encoding
        ports = torch.arange(self.n_ports).repeat(batch_size, 1)
        ports = self.Container_embedding(ports)
        output = self.linear1(x)
        # We add a positional encoding of the ports
        output = torch.cat([output, ports], dim=2)
        output = self.tanh(output)
        output = self.linear2(output)
        output = self.tanh(output)
        output = self.flatten(output)
        return output


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        n_ports,
        container_embedding_size,
        hidden_size
    ):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomCombinedExtractor, self).__init__(
            observation_space,
            features_dim=1
        )
        extractors = {}
        self.Container_embedding = nn.Embedding(
            n_ports,
            container_embedding_size
        )

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == 'bay_matrix':
                extractors[key] = nn.Sequential(
                    # We want to encode columns, not rows
                    Transpose(),
                    self.Container_embedding,
                    # Flatten the embedding dimension, keep batch and column
                    nn.Flatten(2),
                    nn.Linear(
                        subspace.shape[0] * container_embedding_size,
                        hidden_size
                    ),
                    nn.Tanh(),
                    nn.Linear(
                        hidden_size,
                        hidden_size
                    ),
                    nn.Tanh(),
                    nn.Flatten()
                )
                total_concat_size += subspace.shape[1] * hidden_size
            elif key == 'container':
                extractors[key] = nn.Sequential(
                    self.Container_embedding,
                    nn.Tanh(),
                    nn.Flatten()
                )
                total_concat_size += container_embedding_size
            elif key == 'port':
                extractors[key] = nn.Sequential(
                    self.Container_embedding,
                    nn.Tanh(),
                    nn.Flatten()
                )
                total_concat_size += container_embedding_size
            elif key == 'transportation_matrix':
                extractors[key] = TransportationEncoder(
                    self.Container_embedding,
                    container_embedding_size,
                    n_ports,
                    hidden_size
                )
                total_concat_size += subspace.shape[0] * hidden_size

        self.extractors = nn.ModuleDict(extractors)

        self.final_layer = nn.Sequential(
            nn.Linear(total_concat_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )

        # Update the features dim manually
        self._features_dim = hidden_size

    def forward(self, observations):
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            # print(key)
            # print(observations[key].shape)
            # print(observations[key])
            # extract = extractor(observations[key].long())
            # print(extract.shape)
            # print(extract)
            # We are given a (Batch, Height, Width) PyTorch tensor
            encoded_tensor_list.append(
                extractor(
                    # Make into long for nn.Embedding
                    observations[key].long()
                )
            )

        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        output = torch.cat(encoded_tensor_list, dim=1)
        output = self.final_layer(output)
        return output
