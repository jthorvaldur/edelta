import warnings

import numpy as np
import test
import torch

from qvocab.util.test_util import test_model_vector
from qvocab.vqvae.vqvae_class import VQVAE

warnings.simplefilter(action="ignore", category=FutureWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if __name__ == "__main__":
    input_dim = 6
    embedding_dim = 32
    num_embeddings = 128
    commitment_cost = 0.25

    model = VQVAE(input_dim, embedding_dim, num_embeddings, commitment_cost).to(device)
    model.load_state_dict(torch.load("vqvae.pth"))
    model.eval()

    new_vector = np.random.randn(input_dim).astype(np.float32)
    new_vector_tensor = torch.tensor(new_vector).unsqueeze(0).to(device)

    with torch.no_grad():
        z = model.encoder(new_vector_tensor)
        z_quantized, _, _ = model.quantizer(z)
        reconstructed_vector = model.decoder(z_quantized)

    reconstructed_vector_np = reconstructed_vector.cpu().numpy().squeeze()

    print("Original Vector:", new_vector)
    print("Reconstructed Vector:", reconstructed_vector_np)

    test_model_vector(model, input_dim, device)
