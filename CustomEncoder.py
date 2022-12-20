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
        ].float()


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
                    nn.Tanh(),
                    nn.Linear(subspace_size * embedding_dim, 64),
                    nn.Tanh(),
                )
                total_concat_size += 64
            elif key == 'transportation_matrix':
                upper_triangular_size = int(
                    n_ports * (n_ports - 1) / 2
                )  # Sum of 1 to n-1. Int for shape compatibility
                extractors[key] = nn.Sequential(
                    Extract_upper_triangular_batched(n_ports),
                    nn.Linear(upper_triangular_size, 128),
                    nn.Tanh(),
                )
                total_concat_size += 128
            elif key == 'port':
                extractors[key] = nn.Identity()  # No need to do anything
                total_concat_size += 1  # Port is a scalar

        self.extractors = nn.ModuleDict(extractors)

        self.final_layer = nn.Sequential(
            nn.Linear(total_concat_size, 128),
            nn.Tanh(),
        )

        # Update the features dim manually
        self._features_dim = 128

    def forward(self, observations):
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
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
