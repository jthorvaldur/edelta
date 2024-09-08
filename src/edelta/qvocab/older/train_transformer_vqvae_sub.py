import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from qvocab.transformer.transformer_vqvae import TransformerVQVAE2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def train_transformer_vqvae_1(model, dataloader, num_epochs, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            inputs = batch[0].to(device)

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


def train_transformer_vqvae(model, dataloader, num_epochs, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            inputs = batch[0].to(device)

            # Print the shape of inputs before passing them to the model
            print(f"Batch input shape: {inputs.shape}")

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
    input_dim = 8
    embedding_dim = 64
    num_embeddings = 256
    commitment_cost = 0.25
    batch_size = 64
    num_epochs = 100
    learning_rate = 0.008

    data = np.random.randn(num_samples, input_dim).astype(np.float32)
    cov_ = np.cov(data.T)
    U, _, _ = np.linalg.svd(cov_)
    data = data @ U
    data = data.astype(np.float32)

    dataset = TensorDataset(torch.tensor(data))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # model = TransformerVQVAE2(
    #     input_dim=6,  # Ensure this matches your data
    #     embedding_dim=32,
    #     num_embeddings=128,
    #     commitment_cost=0.25,
    # ).to(device)

    model = TransformerVQVAE2(
        input_dim=embedding_dim,  # Adjusted to match embedding dimension
        embedding_dim=embedding_dim,
        num_embeddings=num_embeddings,
        commitment_cost=commitment_cost,
        num_heads=2,
    ).to(device)

    train_transformer_vqvae(model, dataloader, num_epochs, learning_rate)

    # torch.save(model.state_dict(), "transformer_vqvae.pth")
