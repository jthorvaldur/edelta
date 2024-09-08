import warnings

import numpy as np
import torch

from qvocab.transformer.transformer_vqvae import TransformerVQVAE
from qvocab.util.test_util import test_transformer_model_vector

warnings.simplefilter(action="ignore", category=FutureWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if __name__ == "__main__":
    num_samples = 10000
    input_dim = 6
    embedding_dim = 32
    num_embeddings = 256
    commitment_cost = 0.25
    batch_size = 64
    num_epochs = 30
    learning_rate = 0.005

    model = TransformerVQVAE(
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        num_embeddings=num_embeddings,
        commitment_cost=commitment_cost,
        num_heads=4,
        dim_feedforward=64,
        num_layers=2,
    ).to(device)
    model.load_state_dict(torch.load("transformer_vqvae.pth"))
    model.eval()

    new_vector = np.random.randn(input_dim).astype(np.float32)
    new_vector_tensor = torch.tensor(new_vector).unsqueeze(0).to(device)

    with torch.no_grad():
        # Convert input dimension to embedding dimension
        embedded_vector = model.input_to_embedding(
            new_vector_tensor
        )  # Shape: (batch_size, embedding_dim)

        # Add sequence length dimension (seq_length = 1)
        embedded_vector = embedded_vector.unsqueeze(
            1
        )  # Shape: (batch_size, 1, embedding_dim)
        z_e1 = model.encoder1(embedded_vector)
        z_q1, _, _ = model.quantizer1(z_e1)

        z_e2 = model.encoder2(z_q1)
        z_q2, _, _ = model.quantizer2(z_e2)

        reconstructed_vector = model.decoder(z_q2)

    reconstructed_vector_np = reconstructed_vector.cpu().numpy().squeeze()

    print("Original Vector:", new_vector)
    print("Reconstructed Vector:", reconstructed_vector_np)

    test_transformer_model_vector(model, input_dim, device)
