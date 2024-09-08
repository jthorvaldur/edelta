import torch
import torch.nn as nn


class Config:
    """Configuration class for managing all parameters and settings."""

    def __init__(self):
        # Data processing parameters
        self.n_steps = 32
        self.n_basis = 3
        self.data_directory = "~/svtmp/data/"
        self.data_file = "spy.parquet"

        # Model parameters
        self.input_dim = self.n_basis
        self.embedding_dim = 32
        self.num_embeddings = 512
        self.commitment_cost = 0.25

        # Encoder and Decoder layers configuration
        self.encoder_layers = [32, 64, 128]  # Example: 3 layers with specified sizes
        self.decoder_layers = self.encoder_layers[
            ::-1
        ]  # Must mirror the encoder in reverse
        self.activation = "tanh"  # Can be "relu", "tanh", or "sigmoid"
        self.use_skip = True

        # Training parameters
        self.batch_size = 32
        self.num_epochs = 150
        self.learning_rate = 0.005
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Transformer Params
        self.num_heads = 4
        self.dim_feedforward = 128
        self.num_layers = 2


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.normal_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
