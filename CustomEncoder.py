from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch


class Extract_upper_triangular_batched(nn.Module):
    def __init__(self, n_ports):
        super().__init__()
        self.upper_triangular_indeces = torch.triu_indices(
            n_ports,
            n_ports,
            offset=1  # We don't want to include the diagonal
        )

    def forward(self, x):
        return x[
            :,  # Batch dimension
            self.upper_triangular_indeces[0],
            self.upper_triangular_indeces[1]
        ]


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, n_ports, vocab_size, embedding_dim):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomCombinedExtractor, self).__init__(
            observation_space,
            features_dim=1
        )
        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            subspace_size = np.prod(subspace.shape)

            if key == 'bay_matrix':
                extractors[key] = nn.Sequential(
                    nn.Flatten(),
                    nn.Embedding(
                        vocab_size,
                        embedding_dim
                    ),
                    nn.Flatten(),
                )

                total_concat_size += subspace_size * embedding_dim
            elif key == 'transportation_matrix':
                extractors[key] = Extract_upper_triangular_batched(n_ports)
                upper_triangular_size = int(
                    n_ports * (n_ports - 1) / 2
                )  # Sum of 1 to n-1. Int for shape compatibility
                total_concat_size += upper_triangular_size
            elif key == 'port':
                extractors[key] = nn.Flatten()
                total_concat_size += 1  # Port is a scalar

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations):
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            # print('----', key)
            # print(observations[key].shape)
            # print(observations[key])
            # results = extractor(
            #     observations[key].long()
            # )
            # print(results.shape)
            # print(results)
            # We are given a (Batch, Height, Width) PyTorch tensor
            encoded_tensor_list.append(
                extractor(
                    # Make into long for nn.Embedding
                    observations[key].long()
                )
            )
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return torch.cat(encoded_tensor_list, dim=1)
