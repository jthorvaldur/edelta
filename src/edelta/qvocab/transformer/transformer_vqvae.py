import sys

import numpy as np
import torch
import torch.nn as nn

from flow.config import initialize_weights
from qvocab.quantizer.vector_quantizer import VectorQuantizer


class TransformerVQVAE(nn.Module):
    def __init__(self, config, dataloader):
        super(TransformerVQVAE, self).__init__()

        self.config = config
        self.dataloader = dataloader
        self.device = config.device

        # Linear layer to map from input_dim to embedding_dim
        self.input_to_embedding = nn.Linear(config.input_dim, config.embedding_dim)

        # Transformer Encoders
        self.encoder1 = TransformerEncoderLayer(
            config.embedding_dim,
            num_heads=config.num_heads,  # These could also be made configurable in config
            dim_feedforward=config.dim_feedforward,  # Same here, based on config
            num_layers=config.num_layers,  # And here as well
        )
        self.encoder2 = TransformerEncoderLayer(
            config.embedding_dim,
            num_heads=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            num_layers=config.num_layers,
        )

        # Quantizers
        self.quantizer1 = VectorQuantizer(
            config.num_embeddings, config.embedding_dim, config.commitment_cost
        )
        self.quantizer2 = VectorQuantizer(
            config.num_embeddings, config.embedding_dim, config.commitment_cost
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(
                config.embedding_dim, config.input_dim
            ),  # Maps back to the original input_dim
            # nn.Tanh(),
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.learning_rate)

        self.apply(initialize_weights)

    def get_vocabulary(self):
        """Get the learned vocabulary (quantized embeddings)."""
        embedding_dict1 = self.quantizer1.embeddings.weight.detach().cpu().numpy()
        embedding_dict2 = self.quantizer2.embeddings.weight.detach().cpu().numpy()
        # Combine both dictionaries if you want a unified view
        combined_embedding_dict = np.concatenate(
            (embedding_dict1, embedding_dict2), axis=0
        )
        return combined_embedding_dict

    def forward(self, x):
        # Convert input dimension to embedding dimension
        x = self.input_to_embedding(x)  # Shape: (batch_size, embedding_dim)

        # Pass through the first transformer encoder
        z_e1 = self.encoder1(x)  # Shape: (batch_size, embedding_dim)

        # Quantize the output of the first transformer encoder
        z_q1, vq_loss1, _ = self.quantizer1(z_e1)  # Shape: (batch_size, embedding_dim)

        # Pass through the second transformer encoder
        z_e2 = self.encoder2(z_q1)  # Shape: (batch_size, embedding_dim)

        # Quantize the output of the second transformer encoder
        z_q2, vq_loss2, _ = self.quantizer2(z_e2)  # Shape: (batch_size, embedding_dim)

        # Decode the quantized representation
        x_reconstructed = self.decoder(z_q2)  # Shape: (batch_size, input_dim)

        # Combine the losses from both quantizers
        total_vq_loss = vq_loss1 + vq_loss2

        return x_reconstructed, total_vq_loss

    def map_new_vector(self, new_vector):
        """Map a new vector to the learned vocabulary, symmetric to VQVAE."""
        new_vector_tensor = torch.tensor(new_vector).unsqueeze(0).to(self.device)

        # Convert input dimension to embedding dimension
        z = self.input_to_embedding(new_vector_tensor)

        # Pass through both encoders and quantizers
        z_e1 = self.encoder1(z)
        z_q1, _, _ = self.quantizer1(z_e1)
        z_e2 = self.encoder2(z_q1)
        z_q2, _, _ = self.quantizer2(z_e2)

        # Return the final quantized representation
        return z_q2.detach().cpu().numpy().squeeze()

    def train_model(self):
        """Training loop for the Transformer VQ-VAE model."""
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
                if epoch == 12:
                    print(outputs)
                    print(inputs)
                    sys.exit()
                recon_loss = nn.functional.mse_loss(outputs, inputs)
                loss = recon_loss + vq_loss

                # Backward pass and optimize
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

                self.optimizer.step()

                total_loss += loss.item()

            # Calculate average loss for the epoch
            avg_loss = total_loss / len(self.dataloader)
            print(f"Epoch [{epoch + 1}/{self.config.num_epochs}], Loss: {avg_loss:.4f}")

        print("Training complete.")


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, dim_feedforward, num_layers):
        super(TransformerEncoderLayer, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                batch_first=True,
                norm_first=not True,
            ),
            num_layers=num_layers,
        )

    def forward(self, x):
        return self.transformer(x)
