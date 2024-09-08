import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from qvocab.quantizer.vector_quantizer import VectorQuantizer


class VQVAE(nn.Module):
    def __init__(self, config, dataloader):
        super(VQVAE, self).__init__()

        self.config = config
        self.dataloader = dataloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Encoder Network
        self.encoder = self.build_network(
            config.input_dim, config.encoder_layers, config.activation
        )

        # Vector Quantizer
        self.quantizer = VectorQuantizer(
            config.num_embeddings, config.embedding_dim, config.commitment_cost
        )

        # Decoder Network
        self.decoder = self.build_network(
            config.embedding_dim, config.decoder_layers, config.activation, reverse=True
        )

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.config.learning_rate)

    def build_network(self, input_dim, layers, activation, reverse=False):
        """Build a sequential neural network based on the given layer configuration."""
        network = []
        in_dim = input_dim

        for layer_dim in layers:
            network.append(nn.Linear(in_dim, layer_dim))
            if activation == "relu":
                network.append(nn.ReLU())
            elif activation == "tanh":
                network.append(nn.Tanh())
            elif activation == "sigmoid":
                network.append(nn.Sigmoid())
            else:
                raise ValueError(f"Unsupported activation function: {activation}")
            in_dim = layer_dim

        if reverse:
            network.append(nn.Linear(layers[-1], input_dim))
        else:
            network.append(nn.Linear(layers[-1], self.config.embedding_dim))

        return nn.Sequential(*network)

    def forward(self, x):
        # Forward pass through encoder, quantizer, and decoder
        z = self.encoder(x)
        z_quantized, vq_loss, _ = self.quantizer(z)
        x_reconstructed = self.decoder(z_quantized)
        return x_reconstructed, vq_loss

    def train_model(self):
        """Training loop for the VQ-VAE model."""
        # Set the model to training mode
        super().train()

        for epoch in range(self.config.num_epochs):
            total_loss = 0
            for batch in self.dataloader:
                inputs = batch[0].to(self.device)

                # Zero the gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs, vq_loss = self(inputs)
                recon_loss = F.mse_loss(outputs, inputs)
                loss = recon_loss + vq_loss

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            # Calculate average loss for the epoch
            avg_loss = total_loss / len(self.dataloader)
            print(f"Epoch [{epoch + 1}/{self.config.num_epochs}], Loss: {avg_loss:.4f}")

        print("Training complete.")
