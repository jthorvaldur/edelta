import math

import torch
import torch.nn as nn

from flow.config import initialize_weights
from qvocab.quantizer.vector_quantizer import VectorQuantizer
from qvocab.transformer.t_encoder import TransformerEncoderLayer


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.embedding_dim = embedding_dim

        # Create a matrix of shape (max_len, embedding_dim) to hold the positional encodings
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-math.log(10000.0) / embedding_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add a batch dimension

        self.register_buffer("pe", pe)

    def forward(self, x):
        # x has shape (batch_size, sequence_length, embedding_dim)
        # Ensure positional encodings are the correct length
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return x


# Example usage within a Transformer model
class TransformerVQVAE(nn.Module):
    def __init__(self, config, dataloader):
        super(TransformerVQVAE, self).__init__()

        self.config = config
        self.dataloader = dataloader
        self.device = config.device

        # Linear layer to map from input_dim to embedding_dim
        self.input_to_embedding = nn.Linear(config.input_dim, config.embedding_dim)

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(config.embedding_dim)

        # Transformer Encoder
        self.encoder = TransformerEncoderLayer(
            config.embedding_dim,
            num_heads=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            num_layers=config.num_layers,
        )

        # Single Quantizer
        self.quantizer = VectorQuantizer(
            config.num_embeddings, config.embedding_dim, config.commitment_cost
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(config.embedding_dim, config.input_dim),
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.learning_rate)

        # Apply the weight initialization
        self.apply(initialize_weights)

    def forward(self, x):
        # Convert input dimension to embedding dimension
        x = self.input_to_embedding(x)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Pass through the transformer encoder
        z_e = self.encoder(x)

        # Quantize and decode
        z_q, vq_loss, _ = self.quantizer(z_e)
        x_reconstructed = self.decoder(z_q)

        return x_reconstructed, vq_loss

    def get_vocabulary(self):
        """Get the learned vocabulary (quantized embeddings)."""
        embedding_dict = self.quantizer.embeddings.weight.detach().cpu().numpy()
        return embedding_dict

    def map_new_vector(self, new_vector):
        """Map a new vector to the learned vocabulary, symmetric to VQVAE."""
        new_vector_tensor = torch.tensor(new_vector).unsqueeze(0).to(self.device)

        # Convert input dimension to embedding dimension
        z = self.input_to_embedding(new_vector_tensor)

        # Pass through the transformer encoder and quantizer
        z_e = self.encoder(z)
        z_q, _, _ = self.quantizer(z_e)

        # Return the final quantized representation
        return z_q.detach().cpu().numpy().squeeze()

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
                # if epoch == 12:
                #     print(outputs)
                #     print(inputs)
                #     sys.exit()
                recon_loss = nn.functional.mse_loss(outputs, inputs)
                # recon_loss = nn.functional.kl_div(outputs.log_softmax(dim=-1), inputs.softmax(dim=-1), reduction='batchmean')
                # recon_loss = 1 - nn.functional.cosine_similarity(outputs, inputs).mean()
                # recon_loss = nn.functional.smooth_l1_loss(outputs, inputs)
                recon_loss = nn.functional.l1_loss(outputs, inputs)

                # def log_cosh_loss(outputs, inputs):
                #     return torch.mean(torch.log(torch.cosh(outputs - inputs)))

                # recon_loss = log_cosh_loss(outputs, inputs)

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
