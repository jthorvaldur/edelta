import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # Initializing the codebook (embedding table)
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(
            -1 / self.num_embeddings, 1 / self.num_embeddings
        )

    def forward(self, inputs):
        # No need to permute; directly flatten the input
        flat_input = inputs.view(-1, self.embedding_dim)

        # Calculate distances from embeddings
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self.embeddings.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self.embeddings.weight.t())
        )

        # Get the encoding that has the minimum distance
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.size(0), self.num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize the input
        quantized = torch.matmul(encodings, self.embeddings.weight).view(inputs.shape)

        # Calculate the loss for embedding
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Add the gradient to embedding vectors
        quantized = inputs + (quantized - inputs).detach()

        return quantized, loss, encoding_indices


class VQVAE(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_embeddings, commitment_cost):
        super(VQVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2 * embedding_dim),
            nn.ReLU(),
            nn.Linear(2 * embedding_dim, embedding_dim),
        )

        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 2 * embedding_dim),
            nn.ReLU(),
            nn.Linear(2 * embedding_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        z_quantized, vq_loss, _ = self.quantizer(z)
        x_reconstructed = self.decoder(z_quantized)
        return x_reconstructed, vq_loss


class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward, num_layers):
        super(TransformerEncoderLayer, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

    def forward(self, x):
        return self.transformer(x)


class TransformerVQVAE2(nn.Module):
    def __init__(
        self,
        input_dim,
        embedding_dim,
        num_embeddings,
        commitment_cost,
        num_heads=2,
        dim_feedforward=64,
        num_layers=2,
    ):
        super(TransformerVQVAE2, self).__init__()

        # Transformer Encoder (replaces Conv1d encoder)
        self.encoder1 = TransformerEncoderLayer(
            input_dim, num_heads, dim_feedforward, num_layers
        )
        self.encoder2 = TransformerEncoderLayer(
            embedding_dim, num_heads, dim_feedforward, num_layers
        )

        # Quantizers for two levels
        self.quantizer1 = VectorQuantizer(num_embeddings, input_dim, commitment_cost)
        self.quantizer2 = VectorQuantizer(
            num_embeddings, embedding_dim, commitment_cost
        )

        # Transformer Decoder (replaces ConvTranspose1d decoder)
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        # Encoder
        z_e1 = self.encoder1(x)
        z_q1, vq_loss1, _ = self.quantizer1(z_e1)

        z_e2 = self.encoder2(z_q1)
        z_q2, vq_loss2, _ = self.quantizer2(z_e2)

        # Decoder
        x_reconstructed = self.decoder(z_q2)

        # Total loss
        total_vq_loss = vq_loss1 + vq_loss2

        return x_reconstructed, total_vq_loss


def train_vqvae(model, dataloader, num_epochs, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            inputs = batch[0].cuda()

            optimizer.zero_grad()
            outputs, vq_loss = model(inputs)
            recon_loss = F.mse_loss(outputs, inputs)
            loss = recon_loss + vq_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    print("Training complete.")


if __name__ == "__main__":
    # Parameters
    num_samples = 10000
    input_dim = 6  # N
    embedding_dim = 32
    num_embeddings = 128  # in practice maybe 10,000
    commitment_cost = 0.25
    batch_size = 64
    num_epochs = 100
    learning_rate = 0.008
    import sys

    # Generate synthetic data
    data = np.random.randn(num_samples, input_dim).astype(np.float32)
    # data = (data - np.mean(data)) / np.std(data)
    print(data.shape)

    print(np.linalg.matrix_rank(data))
    # svd reduce dimensionality
    cov_ = np.cov(data.T)
    U, S, V = np.linalg.svd(cov_)
    # U[:, 3:] = 0 # uncomment to reduce dimensionality and converge faster (but less accurate)
    data = data @ U
    data = data.astype(np.float32)
    print(data.shape)
    # evaluate effective dimensionality of data after svd
    cov = np.cov(data.T)

    print(np.linalg.matrix_rank(data))
    print(np.linalg.matrix_rank(cov))
    # sys.exit()
    dataset = TensorDataset(torch.tensor(data))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model 1
    model = VQVAE(input_dim, embedding_dim, num_embeddings, commitment_cost).cuda()

    # Train the model
    train_vqvae(model, dataloader, num_epochs, learning_rate)

    # Save the model
    torch.save(model.state_dict(), "vqvae.pth")
