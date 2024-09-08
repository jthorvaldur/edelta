import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


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
        pe = pe.unsqueeze(0)  # Change to batch_first format
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]  # Correct dimension matching
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, dim_feedforward, max_len=5000):
        super(TransformerEncoder, self).__init__()
        self.pos_encoder = PositionalEncoding(input_dim, max_len)
        encoder_layers = nn.TransformerEncoderLayer(
            input_dim, num_heads, dim_feedforward, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

    def forward(self, src):
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, dim_feedforward, max_len=5000):
        super(TransformerDecoder, self).__init__()
        self.pos_encoder = PositionalEncoding(input_dim, max_len)
        decoder_layers = nn.TransformerDecoderLayer(
            input_dim, num_heads, dim_feedforward, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)

    def forward(self, tgt, memory):
        tgt = self.pos_encoder(tgt)
        output = self.transformer_decoder(tgt, memory)
        return output


class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, dim_feedforward, max_len=5000):
        super(TransformerAutoencoder, self).__init__()
        self.encoder = TransformerEncoder(
            input_dim, num_heads, num_layers, dim_feedforward, max_len
        )
        self.decoder = TransformerDecoder(
            input_dim, num_heads, num_layers, dim_feedforward, max_len
        )
        self.linear = nn.Linear(
            input_dim, input_dim
        )  # Final linear layer to map to the original input_dim

    def forward(self, src):
        memory = self.encoder(src)
        output = self.decoder(src, memory)
        output = self.linear(output)
        return output


def train_autoencoder(model, dataloader, num_epochs, learning_rate):
    criterion = nn.MSELoss()  # Mean Squared Error Loss for reconstruction
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()  # Set model to training mode

    for epoch in range(num_epochs):
        total_loss = 0
        for sequences, _ in dataloader:
            sequences = sequences.cuda()

            optimizer.zero_grad()
            reconstructed_sequences = model(sequences)
            loss = criterion(reconstructed_sequences, sequences)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    print("Training complete.")


if __name__ == "__main__":
    # Model parameters
    input_dim = 16
    seq_len = 10
    num_heads = 4
    num_layers = 4
    dim_feedforward = 64
    batch_size = 32
    num_epochs = 30
    learning_rate = 0.003

    # Generate synthetic data
    num_samples = 1000
    sequences = torch.rand(
        num_samples, seq_len, input_dim
    )  # (batch_size, seq_len, input_dim)
    dataset = TensorDataset(
        sequences, torch.zeros(num_samples)
    )  # Dummy labels for DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Instantiate the model
    model = TransformerAutoencoder(
        input_dim, num_heads, num_layers, dim_feedforward, max_len=seq_len
    ).cuda()

    # Train the model
    train_autoencoder(model, dataloader, num_epochs, learning_rate)

    # Save the trained model
    torch.save(model.state_dict(), "transformer_autoencoder.pth")
