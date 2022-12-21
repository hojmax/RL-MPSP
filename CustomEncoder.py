from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch


# class Extract_upper_triangular_batched(nn.Module):
#     def __init__(self, n_ports):
#         super().__init__()
#         self.upper_triangular_indeces = torch.triu_indices(
#             n_ports,
#             n_ports,
#             offset=1  # We don't want to include the diagonal
#         )

#     def forward(self, x):
#         return x[
#             :,  # Batch dimension
#             self.upper_triangular_indeces[0],
#             self.upper_triangular_indeces[1]
#         ].float()


class Transpose(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mT


class TransportationEncoder(nn.Module):
    def __init__(
        self,
        Port_embedding,
        port_embedding_size,
        rows,
        columns,
        hidden_size
    ):
        super().__init__()
        self.Port_embedding = Port_embedding
        self.rows = rows
        self.linear1 = nn.Linear(
            columns,
            hidden_size
        )
        self.linear2 = nn.Linear(
            port_embedding_size + hidden_size,
            hidden_size
        )
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = x.float()
        # x.shape[0] is batch size
        ports = torch.arange(self.rows).repeat(x.shape[0], 1)
        ports = self.Port_embedding(ports)
        output = self.linear1(x)
        output = torch.cat([output, ports], dim=2)
        output = self.linear2(output)
        output = self.flatten(output)
        return output


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        n_ports,
        container_embedding_size,
        port_embedding_size,
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
        self.Port_embedding = nn.Embedding(
            n_ports,
            port_embedding_size
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
                    nn.Flatten()
                )
                total_concat_size += container_embedding_size
            elif key == 'port':
                extractors[key] = nn.Sequential(
                    self.Port_embedding,
                    nn.Flatten()
                )
                total_concat_size += port_embedding_size
            elif key == 'transportation_matrix':
                extractors[key] = TransportationEncoder(
                    self.Port_embedding,
                    port_embedding_size,
                    subspace.shape[0],
                    subspace.shape[1],
                    hidden_size
                )
                total_concat_size += subspace.shape[0] * hidden_size

        self.extractors = nn.ModuleDict(extractors)

        # self.final_layer = nn.Sequential(
        #     nn.Linear(total_concat_size, encoding_size),
        #     nn.Tanh(),
        #     nn.Linear(encoding_size, encoding_size),
        #     nn.Tanh(),
        #     nn.Linear(encoding_size, encoding_size),
        #     nn.Tanh()
        # )

        # Update the features dim manually
        self._features_dim = total_concat_size

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
        # output = self.final_layer(output)
        return output
