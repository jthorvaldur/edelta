import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from vqvae.older.runvq import VQVAE, VectorQuantizer
from torch.utils.data import DataLoader, TensorDataset

warnings.simplefilter(action="ignore", category=FutureWarning)


if __name__ == "__main__":
    # Parameters
    num_samples = 1000
    input_dim = 6  # N
    embedding_dim = 32
    num_embeddings = 128
    commitment_cost = 0.25
    batch_size = 16
    num_epochs = 50
    learning_rate = 0.008

    # Assuming the model has been trained and saved as "vqvae.pth"

    # Load the model and move it to the appropriate device
    model = VQVAE(input_dim, embedding_dim, num_embeddings, commitment_cost).cuda()
    model.load_state_dict(torch.load("vqvae.pth"))
    model.eval()  # Set the model to evaluation mode

    # New vector to be processed
    new_vector = np.random.randn(input_dim).astype(np.float32)  # Example new vector
    new_vector_tensor = (
        torch.tensor(new_vector).unsqueeze(0).cuda()
    )  # Convert to tensor and add batch dimension

    # Encode the vector to get the latent representation
    with torch.no_grad():
        z = model.encoder(new_vector_tensor)  # Pass through the encoder
        z_quantized, _, _ = model.quantizer(z)  # Quantize the latent representation
        print(z_quantized)

    # Decode the quantized latent representation to reconstruct the approximate vector
    with torch.no_grad():
        reconstructed_vector = model.decoder(z_quantized)

    # Convert to numpy array (optional)
    reconstructed_vector_np = reconstructed_vector.cpu().numpy().squeeze()

    print("Original Vector:", new_vector)
    print("Reconstructed Vector:", reconstructed_vector_np)

    # Access the embedding dictionary
    embedding_dict = model.quantizer.embeddings.weight.detach().cpu().numpy()

    print("Embedding Dictionary Shape:", embedding_dict.shape)
    print("Embedding Dictionary:", embedding_dict)
