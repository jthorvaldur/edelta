# Cell 1: Imports and Initial Setup
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from expand0 import extract_embeddings, load_and_evaluate_model
from scipy.stats import entropy
from torch.utils.data import DataLoader, TensorDataset

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Cell 2: Neural Network Definition with Dropout and Embedded Vector Extraction
class DeepAutoencoder(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(DeepAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2 * input_dim),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(2 * input_dim, embedding_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 2 * input_dim),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(2 * input_dim, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)


# Cell 3: Generate Synthetic Data
def generate_synthetic_data(num_samples, input_dim):
    data = np.random.randn(num_samples, input_dim).astype(np.float32)
    return data


# Cell 4: Training Setup
def train_autoencoder(model, dataloader, num_epochs=50, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(num_epochs):
        for batch in dataloader:
            inputs = batch[0].to(device)  # Move inputs to GPU
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    print("Training complete.")


# Cell 5: Function to Print Model Parameter Space Size
def print_model_param_space_size(model):
    param_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {param_size}")


# Cell 6: Function to Calculate Entropy of Model Weights
def calculate_model_entropy(model):
    entropies = []
    for param in model.parameters():
        if param.requires_grad:
            param_data = param.detach().cpu().numpy().flatten()
            hist, bin_edges = np.histogram(param_data, bins=256, density=True)
            entropies.append(entropy(hist))
    total_entropy = sum(entropies)
    print(f"Total entropy of model parameters: {total_entropy:.4f}")


# Cell 7: Main Function
if __name__ == "__main__":
    # Parameters
    num_samples = 10000
    input_dim = 10  # N
    embedding_dim = 20  # Higher-dimensional space
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.002

    # Generate data
    data = generate_synthetic_data(num_samples, input_dim)
    dataset = TensorDataset(torch.tensor(data).to(device))  # Move dataset to GPU
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = DeepAutoencoder(input_dim, embedding_dim).to(device)

    # Print model parameter space size
    print_model_param_space_size(model)

    # Train the model
    train_autoencoder(model, dataloader, num_epochs, learning_rate)

    # Calculate model entropy
    calculate_model_entropy(model)

    # Save the model
    torch.save(model.state_dict(), "deep_autoencoder.pth")

    # Example usage for loading and evaluating the model
    loaded_model_output = load_and_evaluate_model(
        "deep_autoencoder.pth", input_dim, data
    )
    print("Reconstructed Data:")
    print(pd.DataFrame(loaded_model_output))

    # Extract embeddings
    embeddings = extract_embeddings("deep_autoencoder.pth", input_dim, data)
    print("Embedded Vectors:")
    print(pd.DataFrame(embeddings))
