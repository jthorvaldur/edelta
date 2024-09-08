import torch.nn as nn


def make_mlp(input_dim, hidden_dims, output_dim, activation=nn.ReLU, use_skip=False):
    """
    Generalized MLP builder for both encoder and decoder with optional skip connections.

    Args:
        input_dim (int): Input dimension to the first layer.
        hidden_dims (list of int): List containing the dimensions of the hidden layers.
        output_dim (int): Output dimension of the last layer.
        activation (nn.Module): Activation function class (default: nn.ReLU).
        use_skip (bool): Whether to use skip connections (default: False).

    Returns:
        nn.Module: A sequential model or model with skip connections.
    """
    layers = []
    in_dim = input_dim
    skip_connections = []  # To track skip connections between layers

    for i, hidden_dim in enumerate(hidden_dims):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(activation())

        # If skip connections are enabled, store the input of the first layer
        if use_skip and i == 0:
            skip_connections.append(in_dim)

        in_dim = hidden_dim

    # Add the final output layer
    layers.append(nn.Linear(in_dim, output_dim))

    # If skip connections are enabled, we wrap the model with the SkipMLP class
    if use_skip:
        return SkipMLP(nn.Sequential(*layers), input_dim, output_dim, skip_connections)

    return nn.Sequential(*layers)


class SkipMLP(nn.Module):
    """
    A feedforward network with skip connections between the input and the output layers.
    """

    def __init__(self, sequential_model, input_dim, output_dim, skip_connections):
        super(SkipMLP, self).__init__()
        self.model = sequential_model
        self.skip_layer = nn.Linear(input_dim, output_dim) if skip_connections else None

    def forward(self, x):
        if self.skip_layer is not None:
            return self.model(x) + self.skip_layer(x)  # Skip connection
        return self.model(x)
