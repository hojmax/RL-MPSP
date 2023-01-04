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
        n_ports,
        hidden_size,
        device='cpu'
    ):
        super().__init__()
        self.n_ports = n_ports
        self.hidden_size = hidden_size
        # self.linear = nn.Linear(
        #     n_ports,
        #     hidden_size,
        #     device=device
        # )
        self.device = device

    def forward(self, x, ports):
        x = x.to(self.device).float()

        output = torch.zeros(
            x.shape[0],
            self.n_ports * self.n_ports,
            device=self.device
        )

        ports = ports.squeeze().tolist()
        ports = [int(port) for port in ports]

        for i, port in enumerate(ports):
            batch = x[i]

            # Remove the ports that already have been visited
            batch = batch[port:]

            # Flatten the batch
            batch = batch.flatten()

            # Pad with zeros to have size n_ports * n_ports
            batch = torch.nn.functional.pad(
                batch,
                (0, port * self.n_ports)
            )

            # Add to output
            output[i] = batch

        return output


class BayEncoder(nn.Module):
    def __init__(
        self,
        rows,
        internal_hidden,
        n_ports,
        device='cpu',
    ):
        super().__init__()
        self.model = nn.Sequential(
                    # We want to encode columns, not rows
                    Transpose(),
                    # Flatten the embedding dimension, keep batch and column
                    nn.Flatten(2),
                    nn.Linear(
                        rows,
                        internal_hidden,
                        device=device
                    ),
                    nn.Tanh(),
                    nn.Flatten()
                )
        self.n_ports = n_ports
        self.device = device

    def forward(self, x, ports):
        x = x.to(self.device).float()

        # Subtract the port from each bay
        x = x - ports.unsqueeze(1)

        # Make all negative values zero
        x = torch.nn.functional.relu(x)

        # Flatten the embedding dimension, keep batch and column
        x = x.flatten(1)
        return x
        # return self.model(x)


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
        device='cpu'
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
                extractors[key] = BayEncoder(
                    rows,
                    internal_hidden,
                    n_ports,
                    device=device
                )
                total_concat_size += cols * rows
                # total_concat_size += cols * internal_hidden
            elif key == 'transportation_matrix':
                extractors[key] = TransportationEncoder(
                    n_ports,
                    internal_hidden,
                    device=device
                )
                total_concat_size += n_ports * n_ports

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

        ports = observations['port']

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():

            # We are given a (Batch, Height, Width) PyTorch tensor
            encoded_tensor_list.append(
                extractor(
                    observations[key].to(self.device),
                    ports
                ).to(self.device)
            )

        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        output = torch.cat(encoded_tensor_list, dim=1)
        output = self.final_layer(output)
        return output
