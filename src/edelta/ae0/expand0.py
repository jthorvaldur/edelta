# Cell 1: Imports and Initial Setup
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Cell 2: Neural Network Definition with Dropout and Embedded Vector Extraction
class DeepAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(DeepAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2 * input_dim),
            nn.ReLU(),
            # nn.Dropout(0.2),  # Dropout layer with 20% dropout rate
            nn.Linear(2 * input_dim, 4 * input_dim),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(4 * input_dim, 2 * input_dim),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(2 * input_dim, input_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, 2 * input_dim),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(2 * input_dim, 4 * input_dim),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(4 * input_dim, 2 * input_dim),
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


def load_and_evaluate_model(model_path, input_dim, data):
    model = DeepAutoencoder(input_dim).to(device)  # Ensure the model is on GPU
    state_dict = torch.load(
        model_path, weights_only=True
    )  # Load only the state dictionary
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(data).to(device)  # Move inputs to GPU
        outputs = model(inputs)
    return outputs.cpu().numpy()  # Move outputs to CPU


def extract_embeddings(model_path, input_dim, data):
    model = DeepAutoencoder(input_dim).to(device)  # Ensure the model is on GPU
    state_dict = torch.load(
        model_path, weights_only=True
    )  # Load only the state dictionary
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(data).to(device)  # Move inputs to GPU
        embeddings = model.encode(inputs)
    return embeddings.cpu().numpy()  # Move embeddings to CPU


# Cell 5: Main Function
if __name__ == "__main__":
    # Parameters
    num_samples = 1000
    input_dim = 10  # N
    batch_size = 128
    num_epochs = 50
    learning_rate = 0.002

    # Generate data
    data = generate_synthetic_data(num_samples, input_dim)
    dataset = TensorDataset(torch.tensor(data).to(device))  # Move dataset to GPU
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = DeepAutoencoder(input_dim).to(device)

    # Train the model
    train_autoencoder(model, dataloader, num_epochs, learning_rate)

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
