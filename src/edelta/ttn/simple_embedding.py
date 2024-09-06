import torch
import torch.nn as nn

# Assuming 'data' is your temporal series of float vectors,
# where each vector might represent a time step or a feature set at a point in time.


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape should be (batch_size, sequence_length, input_size)
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(
            x, h0
        )  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = self.fc(out[:, -1, :])  # Take the output from the last time step
        return out


class FloatVectorEmbedder(nn.Module):
    def __init__(self, vector_dim, embedding_dim):
        super(FloatVectorEmbedder, self).__init__()
        self.embedding = nn.Linear(vector_dim, embedding_dim)

    def forward(self, x):
        return self.embedding(x)


# Usage within a larger model:
class TemporalModel(nn.Module):
    def __init__(self, vector_dim, embedding_dim, hidden_size, output_size):
        super(TemporalModel, self).__init__()
        self.embedder = FloatVectorEmbedder(vector_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embedded = self.embedder(x)
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(embedded, h0)
        out = self.fc(out[:, -1, :])
        return out


class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128), nn.ReLU(), nn.Linear(128, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


# Use this encoder part for embedding:
# autoencoder = Autoencoder(input_dim=data.shape[-1], encoding_dim=32)
# After training the autoencoder, you can use `autoencoder.encoder` to get embeddings.


# # Example setup:
# vector_dim = data.shape[-1]  # Dimension of each float vector
# embedding_dim = 32
# hidden_size = 64
# output_size = 1
#
# model = TemporalModel(vector_dim, embedding_dim, hidden_size, output_size)


# # Example usage:
# input_size = data.shape[2] if len(data.shape) == 3 else data.shape[1]
# hidden_size = 64
# output_size = 1  # for example, predicting the next value
#
# model = SimpleRNN(input_size, hidden_size, output_size)
