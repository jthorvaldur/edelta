import re
import sys
import time

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from genutil.perf import timeit

from flow.config import initialize_weights
from qvocab.quantizer.vector_quantizer import VectorQuantizer
from qvocab.skip_sequential.make_layers import SkipMLP, make_mlp


class SVQVAE(nn.Module):
    def __init__(self, config, dataloader):
        super(SVQVAE, self).__init__()

        self.config = config
        self.dataloader = dataloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.activation = nn.Tanh

        self.encoder_mu = make_mlp(
            config.input_dim,
            config.encoder_layers,
            config.embedding_dim,
            activation=self.activation,
        )
        self.encoder_logvar = make_mlp(
            config.input_dim,
            config.encoder_layers,
            config.embedding_dim,
            activation=self.activation,
        )
        self.decoder = make_mlp(
            config.embedding_dim,
            config.decoder_layers,
            config.input_dim,
            activation=self.activation,
        )

        # Vector Quantizer
        self.quantizer = VectorQuantizer(
            config.num_embeddings, config.embedding_dim, config.commitment_cost
        )

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.config.learning_rate)

        self.apply(initialize_weights)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # Calculate standard deviation
        epsilon = torch.randn_like(std)  # Sample from standard normal distribution
        return mu + std * epsilon  # Reparameterization trick

    def forward(self, x):
        # Forward pass through the encoder to get mean and log variance
        mu = self.encoder_mu(x)
        logvar = self.encoder_logvar(x)

        # Reparameterize to sample from the latent space
        z = self.reparameterize(mu, logvar)

        # Vector Quantization
        z_quantized, vq_loss, _ = self.quantizer(z)

        # Decoder
        x_reconstructed = self.decoder(z_quantized)

        # Compute the KL divergence loss between the learned latent distribution and the prior
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return x_reconstructed, vq_loss, kl_loss

    def train_model(self):
        self.train()

        for epoch in range(self.config.num_epochs):
            total_loss = 0
            for batch in self.dataloader:
                inputs = batch[0].to(self.device)

                # Zero the gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs, vq_loss, kl_loss = self(inputs)

                # Reconstruction loss (MSE or other options)
                recon_loss = F.mse_loss(outputs, inputs)

                # Total loss: reconstruction + vector quantization + KL divergence
                loss = recon_loss + vq_loss + kl_loss

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.dataloader)
            if epoch < 10 or (epoch + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{self.config.num_epochs}], Loss: {avg_loss:.4f}"
                )

        print("Training complete.")

    def get_vocabulary(self):
        return self.quantizer.embeddings.weight.detach().cpu().numpy()

    def map_new_vector(self, new_vector):
        return map_new_vector(
            self.encoder_mu,
            self.encoder_logvar,
            self.quantizer,
            new_vector,
            self.device,
        )


def get_vocabulary(quantizer):
    """
    Get the learned vocabulary (quantized embeddings) from the quantizer.

    Args:
        quantizer (nn.Module): The quantizer module, which must have an 'embeddings' attribute.

    Returns:
        np.ndarray: The vocabulary of quantized embeddings.
    """
    return quantizer.embeddings.weight.detach().cpu().numpy()


def reparameterize(mu, logvar):
    """
    Perform the reparameterization trick to sample from the latent space.

    Args:
        mu (torch.Tensor): Mean of the latent Gaussian.
        logvar (torch.Tensor): Log variance of the latent Gaussian.

    Returns:
        torch.Tensor: Sampled latent vector using the reparameterization trick.
    """
    std = torch.exp(0.5 * logvar)  # Calculate the standard deviation
    eps = torch.randn_like(std)  # Sample from a standard normal distribution
    return mu + std * eps  # Reparameterization trick


def map_new_vector(encoder_mu, encoder_logvar, quantizer, new_vector, device):
    """
    Map a new vector to the learned vocabulary using the encoder (mu, logvar) and quantizer.

    Args:
        encoder_mu (nn.Module): The encoder network that outputs the mean.
        encoder_logvar (nn.Module): The encoder network that outputs the log variance.
        quantizer (nn.Module): The quantizer module to quantize the latent representation.
        new_vector (np.ndarray or torch.Tensor): The input vector to map.
        device (torch.device): The device to use for computation (e.g., 'cpu' or 'cuda').

    Returns:
        tuple: (quantized vector, quantization loss, encoding indices).
    """
    new_vector_tensor = torch.tensor(new_vector).unsqueeze(0).to(device)

    with torch.no_grad():
        # Pass through the encoder to get mu and logvar
        mu = encoder_mu(new_vector_tensor)
        logvar = encoder_logvar(new_vector_tensor)

        # Use the reparameterization trick to sample the latent vector
        z_e = reparameterize(mu, logvar)

        # Pass the sampled latent vector through the quantizer
        z_q, loss, encoding_indices = quantizer(z_e)

        # Return the quantized representation, loss, and encoding indices
        return (
            z_q.detach().cpu().numpy().squeeze(),
            loss,
            encoding_indices.detach().cpu().numpy().squeeze(),
        )
