import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from genutil.perf import timeit
from torch.utils.data import DataLoader, TensorDataset

from flow.config import Config
from flow.load_data import get_test_data_df
from qvocab.stochastic.s_vq_vae import SVQVAE as VQVAE

# from qvocab.transformer.transformerpos
# from qvocab.transformer.transformer_vqvae1q import TransformerVQVAE as VQVAE
# import qvocab.transformer.transformerpos as transformerpos
# from qvocab.transformerpos import TransformerVQVAE as VQVAE
# from qvocab.vqvae.vqvae_class import VQVAE
from utils.basefunc import gencbasis_df, genweights_df, project_and_plot_single

warnings.filterwarnings("ignore")


class VQVAEPipeline:
    """Class to encapsulate the full VQ-VAE pipeline, from data processing to model training."""

    def __init__(self, config):
        self.config = config
        self.device = config.device

        # Load and process data
        self.delta, self.df = self.load_data()
        self.weights = self.process_weights()

        # Prepare DataLoader
        self.dataloader = self.prepare_dataloader()

        # Initialize or load the model
        self.model = self.initialize_or_load_model()

    def load_data(self):
        """Load data based on the configuration."""
        delta, df = get_test_data_df(
            directory=self.config.data_directory, file=self.config.data_file
        )
        return delta, df

    def process_weights(self):
        """Process weights using delta and the data frame."""
        weights = genweights_df(self.delta, self.config.n_steps, self.config.n_basis)
        weights = weights[: len(self.df) - len(self.df) % self.config.n_steps]
        weights.index = self.df[
            : len(self.df) - len(self.df) % self.config.n_steps
        ].index[:: self.config.n_steps]
        return weights.astype(np.float32)

    def prepare_dataloader(self):
        """Prepare the DataLoader from the processed weights."""
        dataset = TensorDataset(torch.tensor(self.weights.values))
        dataloader = DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=True
        )
        return dataloader

    def initialize_or_load_model(self):
        """Initialize the VQ-VAE model or load it if already trained."""
        model = VQVAE(config=self.config, dataloader=self.dataloader).to(self.device)
        model_path = "vqvae.pth"

        if os.path.exists(model_path):
            print("Loading trained model from file...")
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print("Training new model...")
            self.train_model(model)
            torch.save(model.state_dict(), model_path)

        return model

    @timeit
    def train_model(self, model):
        """Train the VQ-VAE model."""
        model.train_model()

    def get_vocabulary(self):
        """Get the learned vocabulary (quantized embeddings)."""
        return self.model.get_vocabulary()

    def map_new_vector(self, new_vector):
        """Map a new vector to the learned vocabulary."""
        return self.model.map_new_vector(new_vector)

    def map_new_vector_index(self, new_vector):
        # Calculate distances between new_vector and each vector in the codebook
        # new_vector_tensor = torch.tensor(new_vector).unsqueeze(0).to(self.device)
        self.codebook = self.get_vocabulary()
        distances = torch.norm(self.codebook - new_vector, dim=1)
        # Find the index of the closest vector in the codebook
        min_index = torch.argmin(distances)
        # Get the quantized vector from the codebook
        quantized_vector = self.codebook[min_index]
        # Return both the quantized vector and its index
        return quantized_vector.detach().cpu().numpy().squeeze(), min_index

    def reconstruct_vector(self, quantized_vector):
        """Reconstruct the vector from the quantized latent representation."""
        quantized_tensor = torch.tensor(quantized_vector).unsqueeze(0).to(self.device)
        with torch.no_grad():
            reconstructed_vector = self.model.decoder(quantized_tensor)
        return reconstructed_vector.cpu().numpy().squeeze()

    def calculate_mse_0(self):
        """Calculate the Mean Squared Error between original and reconstructed weights."""
        reconstructed_weights = []

        for batch in self.dataloader:
            inputs = batch[0].to(self.device)
            with torch.no_grad():
                z = self.model.encoder(inputs)
                z_quantized, _, _ = self.model.quantizer(z)
                reconstructed = self.model.decoder(z_quantized)
                reconstructed_weights.append(reconstructed.cpu().numpy())

        # Convert list of arrays to a single array
        reconstructed_weights = np.concatenate(reconstructed_weights, axis=0)

        # Debugging: Print shapes
        print(f"Original weights shape: {self.weights.values.shape}")
        print(f"Reconstructed weights shape: {reconstructed_weights.shape}")

        # Calculate MSE
        mse = F.mse_loss(
            torch.tensor(reconstructed_weights), torch.tensor(self.weights.values)
        ).item()

        print(f"Mean Squared Error: {mse:.4f}")
        return mse

    @timeit
    def calculate_mse(self):
        """Calculate the Mean Squared Error between original and reconstructed weights."""
        all_losses = []

        for batch in self.dataloader:
            inputs = batch[0].to(self.device)
            # print(f"Original input shape: {inputs.shape}")  # Print original input shape

            with torch.no_grad():
                # Convert input dimension to embedding dimension
                z = self.model.input_to_embedding(inputs)
                # print(f"Shape after input_to_embedding: {z.shape}")  # Print shape after embedding

                # Pass through the transformer encoder
                z = self.model.encoder(z)
                # print(f"Shape after encoder: {z.shape}")  # Print shape after encoder

                # Quantize and decode
                z_quantized, _, _ = self.model.quantizer(z)
                reconstructed = self.model.decoder(z_quantized)
                # print(f"Shape after decoder: {reconstructed.shape}")  # Print shape after decoder

            # Calculate MSE for each vector in the batch
            for original, reconstructed_vec in zip(
                inputs.cpu().numpy(), reconstructed.cpu().numpy()
            ):
                mse = F.mse_loss(
                    torch.tensor(reconstructed_vec), torch.tensor(original)
                ).item()
                all_losses.append(mse)

        # Calculate the average and standard deviation of the MSEs
        average_mse = np.mean(all_losses)
        std_mse = np.std(all_losses)
        max_mse = np.quantile(all_losses, 0.99)

        print(f"Mean Squared Error (average over vectors): {average_mse:.4f}")
        print(f"Standard Deviation of MSE: {std_mse:.4f}")
        print(f"Max MSE: {max_mse:.4f}")

        losses_series = pd.Series(all_losses)
        losses = losses_series[losses_series > 0.000001]
        losses = np.log(losses)

        plt.hist(losses, bins=50)
        plt.title("Distribution of MSEs")
        plt.xlabel("MSE")
        plt.ylabel("Frequency")
        plt.show()

        return average_mse, std_mse

    def get_model(self):
        """Get the trained model."""
        return self.model


if __name__ == "__main__":
    # Initialize configuration
    config = Config()

    # Run the VQ-VAE pipeline
    pipeline = VQVAEPipeline(config)

    # Access the model
    model = pipeline.get_model()

    # Get the learned vocabulary (embedding dictionary)
    vocabulary = pipeline.get_vocabulary()
    print("Embedding Dictionary Shape:", vocabulary.shape)
    # print("Embedding Dictionary:", vocabulary)

    # Process a new vector
    new_vector = np.random.randn(config.input_dim).astype(np.float32)

    index = 139 * 2
    # get random index

    index = np.random.randint(0, len(pipeline.weights))
    # vectors = pipeline.weights.values*0
    # for index in range(len(pipeline.weights)):
    #     new_vector = pipeline.weights.iloc[index].values
    #     quantized_vector = pipeline.map_new_vector(new_vector)
    #     reconstructed_vector = pipeline.reconstruct_vector(quantized_vector)
    #     vectors[index,:] = reconstructed_vector

    # df = pd.DataFrame(vectors, columns=pipeline.weights.columns)
    # df.index = pipeline.weights.index
    # error = (df - pipeline.weights).abs().mean(axis=1)
    # print(error)
    # print(df.shape, pipeline.weights.shape)
    # # sys.exit()
    # print(df.std(), pipeline.weights.std())
    # error.plot()
    # plt.show()
    # sys.exit()

    index_list = []
    for i in range(pipeline.weights.shape[0]):
        new_vector = pipeline.weights.iloc[i].values
        quantized_vector, loss, idx = pipeline.map_new_vector(new_vector)
        index_list.append(idx)

    hist = pd.Series(index_list).hist(bins=100)
    plt.show()

    # print(f"Quantized Vector: {idx} {loss:.4f}", quantized_vector)
    # sys.exit()

    # Reconstruct the original vector
    reconstructed_vector = pipeline.reconstruct_vector(quantized_vector)
    print("Original Vector:", new_vector)
    print("Reconstructed Vector:", reconstructed_vector)

    # Calculate the MSE between original and reconstructed weights
    # mse = pipeline.calculate_mse()

    delta, df = get_test_data_df()

    basis = gencbasis_df(
        config.n_steps / config.n_basis, config.n_steps, config.n_basis
    )

    w_old = pipeline.weights.iloc[index]
    w_new = pd.Series(reconstructed_vector, index=w_old.index)
    project_and_plot_single(w_old, w_new, basis, delta, config.n_steps, index)
