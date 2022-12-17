from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch


# class Net(nn.Module):
#     def __init__(self, width, height, hidden_size, n_input_channels):
#         super().__init__()
#         self.conv1 = nn.Conv2d(
#             in_channels=n_input_channels,
#             out_channels=6,
#             kernel_size=3,
#             padding=2
#         )
#         # Size after applying conv2
#         width = (width - 3 + 2 * 2) + 1
#         height = (height - 3 + 2 * 2) + 1
#         self.pool1 = nn.MaxPool2d(2, 2)
#         # Size after applying pool1
#         width = width // 2
#         height = height // 2
#         self.conv2 = nn.Conv2d(
#             in_channels=6,
#             out_channels=16,
#             kernel_size=5,
#             padding=2
#         )
#         # Size after applying conv2
#         width = (width - 5 + 2 * 2) + 1
#         height = (height - 5 + 2 * 2) + 1
#         self.pool2 = nn.MaxPool2d(2, 2)
#         # Size after applying pool2
#         width = width // 2
#         height = height // 2
#         # Channels * width * height
#         self.fc1 = nn.Linear(16 * width * height, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.fc3 = nn.Linear(hidden_size, hidden_size)

#     def forward(self, x):
#         x = self.pool1(F.relu(self.conv1(x)))
#         x = self.pool2(F.relu(self.conv2(x)))
#         x = torch.flatten(x, 2)  # flatten all dimensions except batch
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


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
            if key == 'bay_matrix':
                extractors[key] = nn.Sequential(
                    nn.Conv2d(
                        in_channels=1,
                        out_channels=3,
                        kernel_size=3,
                        padding=2
                    ),
                    nn.MaxPool2d(4),
                    nn.Flatten()
                )
                total_concat_size += (
                    ((
                        (subspace.shape[0] - 3 + 2 * 2) + 1
                    ) // 4) * ((
                        (subspace.shape[1] - 3 + 2 * 2) + 1
                    ) // 4) * 3
                )
            elif key == 'transportation_matrix':
                extractors[key] = nn.Sequential(nn.MaxPool2d(4), nn.Flatten())
                total_concat_size += (subspace.shape[0] //
                                      4) * (subspace.shape[1] // 4)

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
