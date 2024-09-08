import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from flow.config import Config
from flow.load_data import get_test_data_df, prepare_dataloader, process_weights
from qvocab.vqvae.vqvae_class import VQVAE

# Set random seeds and configurations
torch.manual_seed(1)
np.set_printoptions(precision=4, suppress=True)
pd.options.display.float_format = "{:.4f}".format
warnings.simplefilter(action="ignore", category=FutureWarning)


def main():
    # Initialize configuration
    config = Config()

    # Load and process data
    delta, df = get_test_data_df(directory=config.data_directory, file=config.data_file)
    weights = process_weights(config, delta, df)

    # Print data shapes for verification
    print(f"DataFrame shape: {df.shape}")
    print(f"Weights shape: {weights.shape}")

    # Prepare DataLoader
    dataloader = prepare_dataloader(config, weights)

    # Initialize and train VQ-VAE model
    model = VQVAE(config, dataloader).to(config.device)
    model.train_model()


if __name__ == "__main__":
    main()
