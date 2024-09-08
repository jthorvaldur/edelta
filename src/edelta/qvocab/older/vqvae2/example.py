import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def generate_sine_wave_data(num_samples, num_points, frequency_range=(1, 10)):
    x = np.linspace(0, 2 * np.pi, num_points)
    sine_waves = []
    for _ in range(num_samples):
        frequency = np.random.uniform(*frequency_range)
        sine_wave = np.sin(frequency * x)
        sine_waves.append(sine_wave)
    return np.array(sine_waves, dtype=np.float32)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # Initialize the codebook (embedding table)
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(
            -1 / self.num_embeddings, 1 / self.num_embeddings
        )

    def forward(self, inputs):
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


class VQVAE2(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_embeddings, commitment_cost):
        super(VQVAE2, self).__init__()

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv1d(
                1, 64, 4, stride=2, padding=1
            ),  # Downsample, Output: (batch, 64, 50)
            nn.ReLU(),
            nn.Conv1d(
                64, 128, 4, stride=2, padding=1
            ),  # Downsample, Output: (batch, 128, 25)
            nn.ReLU(),
            nn.Conv1d(
                128, embedding_dim, 4, stride=2, padding=1
            ),  # Final conv to embedding_dim, Output: (batch, 64, 13)
        )

        self.encoder2 = nn.Sequential(
            nn.Conv1d(embedding_dim, embedding_dim, 3, stride=1, padding=1), nn.ReLU()
        )

        # Quantizers for two levels
        self.quantizer1 = VectorQuantizer(
            num_embeddings, embedding_dim, commitment_cost
        )
        self.quantizer2 = VectorQuantizer(
            num_embeddings, embedding_dim, commitment_cost
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                embedding_dim, 128, 4, stride=2, padding=1
            ),  # Upsample, Output: (batch, 128, 25)
            nn.ReLU(),
            nn.ConvTranspose1d(
                128, 64, 4, stride=2, padding=1
            ),  # Upsample, Output: (batch, 64, 50)
            nn.ReLU(),
            nn.ConvTranspose1d(
                64, 1, 4, stride=2, padding=1
            ),  # Final conv to 1 channel, Output: (batch, 1, 100)
            nn.Tanh(),
        )

    def forward(self, x):
        # Encoder
        z_e1 = self.encoder1(x)
        z_q1, vq_loss1, _ = self.quantizer1(z_e1)

        z_e2 = self.encoder2(z_q1)
        z_q2, vq_loss2, _ = self.quantizer2(z_e2)

        # Decoder


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


def train_vqvae2(model, dataloader, num_epochs, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            inputs = batch[0].unsqueeze(1).to(device)  # Add channel dimension

            optimizer.zero_grad()
            reconstructed, vq_loss = model(inputs)
            recon_loss = F.mse_loss(reconstructed, inputs)
            loss = recon_loss + vq_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    print("Training complete.")


def evaluate_model(model, sine_waves, num_samples=5):
    model.eval()
    with torch.no_grad():
        for i in range(num_samples):
            input_wave = (
                torch.tensor(sine_waves[i]).unsqueeze(0).unsqueeze(0).to(device)
            )
            reconstructed_wave, _ = model(input_wave)
            reconstructed_wave = reconstructed_wave.squeeze().cpu().numpy()

            plt.figure(figsize=(8, 4))
            plt.plot(sine_waves[i], label="Original")
            plt.plot(reconstructed_wave, label="Reconstructed")
            plt.legend()
            plt.show()


if __name__ == "__main__":
    # Model parameters

    # Parameters
    num_samples = 10000
    num_points = 100  # Number of points in each sine wave

    # Generate the data
    sine_waves = generate_sine_wave_data(num_samples, num_points)
    # plt.plot(sine_waves[0])
    # plt.title("Sample Sine Wave")
    # plt.show()

    # Convert to a PyTorch DataLoader
    dataset = TensorDataset(torch.tensor(sine_waves))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    input_dim = num_points  # Size of each sine wave
    embedding_dim = 64
    num_embeddings = 512
    commitment_cost = 0.25
    num_epochs = 20
    learning_rate = 0.001

    # Instantiate the model
    model = VQVAE2(input_dim, embedding_dim, num_embeddings, commitment_cost).to(device)

    # Train the model
    train_vqvae2(model, dataloader, num_epochs, learning_rate)

    # Evaluate the model
    evaluate_model(model, sine_waves)
