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


class VQVAE(nn.Module):
    def __init__(self, config, dataloader):
        super(VQVAE, self).__init__()

        self.config = config
        self.dataloader = dataloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.activation = nn.LeakyReLU()
        # Encoder Network
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, 2 * config.embedding_dim),
            self.activation,
            nn.Linear(2 * config.embedding_dim, config.embedding_dim),
        )

        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, 2 * config.embedding_dim),
            # nn.BatchNorm1d(2 * config.embedding_dim),  # Add batch normalization
            self.activation,
            nn.Linear(2 * config.embedding_dim, config.embedding_dim),
        )

        # Vector Quantizer
        self.quantizer = VectorQuantizer(
            config.num_embeddings, config.embedding_dim, config.commitment_cost
        )

        # Decoder Network
        self.decoder = nn.Sequential(
            nn.Linear(config.embedding_dim, 2 * config.embedding_dim),
            self.activation,
            nn.Linear(2 * config.embedding_dim, config.input_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(config.embedding_dim, 2 * config.embedding_dim),
            # nn.BatchNorm1d(2 * config.embedding_dim),  # Add batch normalization
            self.activation,
            nn.Linear(2 * config.embedding_dim, config.input_dim),
        )

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.config.learning_rate)
        # self.optimizer = optim.SGD(self.parameters(), lr=self.config.learning_rate)

        self.apply(initialize_weights)

    def forward(self, x):
        # Forward pass through encoder, quantizer, and decoder
        z = self.encoder(x)
        z_quantized, vq_loss, _ = self.quantizer(z)
        x_reconstructed = self.decoder(z_quantized)
        return x_reconstructed, vq_loss

    @timeit
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
                # recon_loss = 1 - nn.functional.cosine_similarity(outputs, inputs).mean()
                # recon_loss = nn.functional.smooth_l1_loss(outputs, inputs)
                loss = recon_loss + 0.25 * vq_loss

                # if epoch == 12:
                #     print(outputs)
                #     print(inputs)
                #     sys.exit()

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            # Calculate average loss for the epoch
            avg_loss = total_loss / len(self.dataloader)
            if epoch < 10 or (epoch + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{self.config.num_epochs}], Loss: {avg_loss:.4f}"
                )

        print("Training complete.")

    def get_vocabulary(self):
        """Get the learned vocabulary (quantized embeddings)."""
        embedding_dict = self.quantizer.embeddings.weight.detach().cpu().numpy()
        return embedding_dict

    def map_new_vector(self, new_vector):
        """Map a new vector to the learned vocabulary."""
        new_vector_tensor = torch.tensor(new_vector).unsqueeze(0).to(self.device)
        with torch.no_grad():
            z_e = self.encoder(new_vector_tensor)

            z_q, loss, encoding_indices = self.quantizer(z_e)

            # Return the final quantized representation
            return (
                z_q.detach().cpu().numpy().squeeze(),
                loss,
                encoding_indices.detach().cpu().numpy().squeeze(),
            )

            # z_quantized, _, _ = self.quantizer(z)
            # return z_quantized.cpu().numpy().squeeze()
