import math

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import MultiheadAttention


class LatentPriorTransformer(nn.Module):
    def __init__(
        self, d_model, nhead, num_encoder_layers, dim_feedforward, codebook_size
    ):
        super(LatentPriorTransformer, self).__init__()
        self.embedding = nn.Embedding(
            codebook_size, d_model
        )  # Embed the discrete codes
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, num_encoder_layers
        )
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, codebook_size)

    def forward(self, src, src_mask=None):
        # src shape: [seq_len, batch_size]
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(0), :]


class GaussianPrior(nn.Module):
    def __init__(self, latent_dim):
        super(GaussianPrior, self).__init__()
        self.mu = nn.Linear(latent_dim, latent_dim)
        self.log_var = nn.Linear(latent_dim, latent_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, z):
        mu = self.mu(z)
        log_var = self.log_var(z)
        z_sample = self.reparameterize(mu, log_var)
        return z_sample, mu, log_var


# Here, during training, you'd minimize KL divergence between this learned distribution and a standard normal,
# plus perhaps a reconstruction loss if you're using this to refine or alter latent codes.
