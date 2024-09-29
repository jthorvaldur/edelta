import math

import torch
import torch.nn as nn
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
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(0), :]


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, dim_feedforward, max_len=5000):
        super(TransformerEncoder, self).__init__()
        self.pos_encoder = PositionalEncoding(input_dim, max_len)
        encoder_layers = nn.TransformerEncoderLayer(
            input_dim, num_heads, dim_feedforward
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

    def forward(self, src):
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output


class SequenceEmbedding(nn.Module):
    def __init__(
        self,
        input_dim,
        num_heads,
        num_layers,
        dim_feedforward,
        max_len=5000,
        pooling="mean",
    ):
        super(SequenceEmbedding, self).__init__()
        self.transformer_encoder = TransformerEncoder(
            input_dim, num_heads, num_layers, dim_feedforward, max_len
        )
        self.pooling = pooling

    def forward(self, src):
        encoded_sequence = self.transformer_encoder(src)

        if self.pooling == "mean":
            return encoded_sequence.mean(dim=0)  # Mean pooling
        elif self.pooling == "max":
            return encoded_sequence.max(dim=0)[0]  # Max pooling
        elif self.pooling == "cls":
            return encoded_sequence[0]  # Use the first token as the embedding
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")


def train_transformer(model, dataloader, num_epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()  # For classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()  # Set model to training mode

    for epoch in range(num_epochs):
        total_loss = 0
        for sequences, labels in dataloader:
            sequences, labels = sequences.cuda(), labels.cuda()

            optimizer.zero_grad()
            logits = model(sequences)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    print("Training complete.")


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        num_heads,
        num_layers,
        dim_feedforward,
        max_len,
        num_classes,
        pooling="mean",
    ):
        super(TransformerClassifier, self).__init__()
        self.sequence_embedding = SequenceEmbedding(
            input_dim, num_heads, num_layers, dim_feedforward, max_len, pooling
        )
        self.classifier = nn.Linear(
            input_dim, num_classes
        )  # Linear layer for classification

    def forward(self, src):
        sequence_embedding = self.sequence_embedding(src)
        logits = self.classifier(sequence_embedding)
        return logits


if __name__ == "__main__":
    # Example sequence of vectors (batch_size=32, seq_len=10, input_dim=16)
    batch_size = 32
    seq_len = 10
    input_dim = 16
    num_classes = 5

    # Generate synthetic data
    sequences = torch.rand(seq_len, batch_size, input_dim)
    labels = torch.randint(
        0, num_classes, (batch_size,)
    )  # Random labels for classification

    # Create a DataLoader
    dataset = TensorDataset(
        sequences.permute(1, 0, 2), labels
    )  # permute to get (batch_size, seq_len, input_dim)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Example sequence of vectors (batch_size=1, seq_len=10, input_dim=16)
    batch_size = 1
    seq_len = 10
    input_dim = 16
    num_heads = 2
    num_layers = 2
    dim_feedforward = 64
    pooling_method = "cls"  # Options: "mean", "max", "cls"

    # Generate synthetic data (sequence of vectors)
    sequence_of_vectors = torch.rand(seq_len, batch_size, input_dim)

    # Model
    model = SequenceEmbedding(
        input_dim, num_heads, num_layers, dim_feedforward, pooling=pooling_method
    )

    # Forward pass to get the sequence embedding
    sequence_embedding = model(sequence_of_vectors)

    print("Sequence Embedding:", sequence_embedding)
