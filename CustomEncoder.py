from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, n_ports, embedding_size):
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
            if key == 'bay_matrix':
                # Change type from torch.FloatTensor to torch.LongTensor for embedding
                extractors[key] = nn.Sequential(
                    nn.Flatten(),
                    nn.Embedding(n_ports, embedding_size),
                    nn.Flatten()
                )
                bay_size = np.prod(subspace.shape)
                flat_embedding_size = embedding_size * bay_size
                total_concat_size += flat_embedding_size
            elif key == 'transportation_matrix':
                extractors[key] = nn.Flatten()
                transportation_size = np.prod(subspace.shape)
                total_concat_size += transportation_size

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations):
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(
                extractor(observations[key].type(torch.long))
            )

        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return torch.cat(encoded_tensor_list, dim=1)
