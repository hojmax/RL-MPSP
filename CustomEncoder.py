from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch


class LoadingListEncoder(nn.Module):
    def __init__(
        self,
        Container_embedding,
        container_embedding_size,
        hidden_size,
        device='cpu'
    ):
        super().__init__()
        self.Container_embedding = Container_embedding
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            # +1 for count
            1 + container_embedding_size,
            hidden_size,
            device=device,
            batch_first=True
        )
        self.device = device

    def forward(self, loading_lists, loading_list_lengths):
        loading_lists = loading_lists.to(self.device)
        # Add container embedding, batched
        loading_lists = torch.cat([
            loading_lists[:, :, 0].unsqueeze(2),
            self.Container_embedding(loading_lists[:, :, 1].long())
        ], dim=2)

        packed_sequence = nn.utils.rnn.pack_padded_sequence(
            loading_lists,
            loading_list_lengths.flatten().cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, (hidden, cell) = self.lstm(packed_sequence)

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


class Transpose(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mT


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        n_ports,
        container_embedding_size,
        internal_hidden,
        output_hidden,
        lstm_hidden,
        device='cpu',
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
            container_embedding_size,
            device=device
        )

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == 'bay_matrix':
                rows = subspace.shape[0]
                cols = subspace.shape[1]
                extractors[key] = nn.Sequential(
                    # Long is required for embedding
                    ToLong(),
                    # We want to encode columns, not rows
                    Transpose(),
                    self.Container_embedding,
                    # Flatten the embedding dimension, keep batch and column
                    nn.Flatten(2),
                    nn.Linear(
                        rows * container_embedding_size,
                        internal_hidden,
                        device=device
                    ),
                    nn.Tanh(),
                    nn.Flatten()
                )
                total_concat_size += cols * internal_hidden
            elif key == 'port':
                extractors[key] = nn.Sequential(
                    # Long is required for embedding
                    ToLong(),
                    self.Container_embedding,
                    nn.Tanh(),
                    nn.Flatten()
                )
                total_concat_size += container_embedding_size
            elif key == 'loading_list':
                extractors[key] = LoadingListEncoder(
                    self.Container_embedding,
                    container_embedding_size,
                    lstm_hidden,
                    device=device
                )
                total_concat_size += lstm_hidden
            elif key == 'will_block':
                extractors[key] = nn.Sequential(
                    ToFloat(),
                    nn.Linear(
                        subspace.shape[0],
                        internal_hidden,
                        device=device
                    ),
                    nn.Tanh(),
                )
                total_concat_size += internal_hidden

        self.extractors = nn.ModuleDict(extractors)

        self.final_layer = nn.Sequential(
            nn.Linear(total_concat_size, output_hidden, device=device),
            nn.Tanh(),
            nn.Linear(output_hidden, output_hidden, device=device),
            nn.Tanh(),
            nn.Linear(output_hidden, output_hidden, device=device),
            nn.Tanh(),
        )

        # Update the features dim manually
        self._features_dim = output_hidden

        self.device = device

    def forward(self, observations):
        encoded_tensor_list = []
        debug_print = False

        encoded_tensor_list.append(
            self.extractors['loading_list'](
                observations['loading_list'].to(self.device),
                observations['loading_list_length'].to(self.device),
            ).to(self.device)
        )

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            if key == 'loading_list' or key == 'loading_list_length':
                # We handle these separately
                continue

            if debug_print:
                print(key)
                print(observations[key].shape)
                print(observations[key])
                extraction = extractor(observations[key].to(self.device))
                print(extraction.shape)
                print(extraction)

            encoded_tensor_list.append(
                extractor(
                    observations[key].to(self.device)
                ).to(self.device)
            )

        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        output = torch.cat(encoded_tensor_list, dim=1)
        output = self.final_layer(output)
        return output
