import numpy as np
import torch


def test_model_vector(model, input_dim, device="cuda"):
    """
    Test function to process a new vector through the model, encode, quantize, decode,
    and compare the original and reconstructed vectors. Also prints the embedding dictionary.

    Parameters:
    model (torch.nn.Module): The VQVAE or TransformerVQVAE model.
    input_dim (int): The dimension of the input vector.
    device (str): The device to run the test on ('cuda' or 'cpu').
    """
    # New vector to be processed
    new_vector = np.random.randn(input_dim).astype(np.float32)  # Example new vector
    new_vector_tensor = (
        torch.tensor(new_vector).unsqueeze(0).to(device)
    )  # Convert to tensor and add batch dimension

    # Encode the vector to get the latent representation
    with torch.no_grad():
        z = model.encoder(new_vector_tensor)  # Pass through the encoder
        z_quantized, _, _ = model.quantizer(z)  # Quantize the latent representation
        print("Quantized Latent Representation:", z_quantized)

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


def test_transformer_model_vector(model, input_dim, embedding_dim, device="cuda"):
    """
    Test function to process a new vector through the Transformer-based VQVAE model, encode, quantize, decode,
    and compare the original and reconstructed vectors. Also prints the embedding dictionary.

    Parameters:
    model (torch.nn.Module): The TransformerVQVAE model.
    input_dim (int): The dimension of the input vector.
    embedding_dim (int): The embedding dimension expected by the model.
    device (str): The device to run the test on ('cuda' or 'cpu').
    """
    # New vector to be processed
    new_vector = np.random.randn(input_dim).astype(np.float32)  # Example new vector
    new_vector_tensor = (
        torch.tensor(new_vector).unsqueeze(0).to(device)
    )  # Convert to tensor and add batch dimension

    # Ensure the input is passed through the input_to_embedding layer and add sequence dimension
    with torch.no_grad():
        embedded_vector = model.input_to_embedding(
            new_vector_tensor
        )  # Map to embedding_dim
        embedded_vector = embedded_vector.unsqueeze(1)  # Add sequence length dimension

        # Pass through the first transformer encoder
        z_e1 = model.encoder1(embedded_vector)

        # Quantize the output of the first transformer encoder
        z_q1, _, _ = model.quantizer1(z_e1)

        # Pass through the second transformer encoder
        z_e2 = model.encoder2(z_q1)

        # Quantize the output of the second transformer encoder
        z_q2, _, _ = model.quantizer2(z_e2)

        # Decode the quantized latent representation to reconstruct the approximate vector
        z_q2 = z_q2.squeeze(1)  # Remove sequence length dimension before decoding
        reconstructed_vector = model.decoder(z_q2)

    # Convert to numpy array (optional)
    reconstructed_vector_np = reconstructed_vector.cpu().numpy().squeeze()

    # Print results
    print("Original Vector:", new_vector)
    print("Reconstructed Vector:", reconstructed_vector_np)

    # Access the embedding dictionary from the second quantizer
    embedding_dict = model.quantizer2.embeddings.weight.detach().cpu().numpy()

    print("Embedding Dictionary Shape:", embedding_dict.shape)
    print("Embedding Dictionary:", embedding_dict)
