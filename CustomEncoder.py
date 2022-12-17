from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, hidden_size):
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
            extractors[key] = nn.Sequential(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=6,
                    kernel_size=3,
                    padding=2
                ),
                nn.MaxPool2d(4),
                nn.Conv2d(
                    in_channels=6,
                    out_channels=16,
                    kernel_size=5,
                    padding=4
                ),
                nn.MaxPool2d(2),
                nn.Flatten()
            )

            with torch.no_grad():
                # Get the output size of the last layer, by passing a dummy tensor
                obs = torch.tensor(
                    subspace.sample()
                ).float()
                shape = obs.shape
                # Must reshape to (Batch, Channels, Height, Width)
                obs = obs.reshape(1, 1, shape[0], shape[1])
                n_flatten = extractors[key](
                    obs
                ).shape[1]

            total_concat_size += n_flatten

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations):
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            shape = observations[key].shape
            # We are given a (Batch, Height, Width) PyTorch tensor
            # Must reshape to (Batch, Channels, Height, Width) for PyTorch conv layers (channels = 1)
            input = observations[key].reshape(shape[0], 1, shape[1], shape[2])
            encoded_tensor_list.append(
                extractor(input)
            )
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return torch.cat(encoded_tensor_list, dim=1)
