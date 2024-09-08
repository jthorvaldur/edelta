import pathlib as pl

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from utils.basefunc import genweights_df


def get_test_data_df(directory="~/svtmp/data", file="spy.parquet") -> pd.DataFrame:
    """Load and process test data from a Parquet file in the specified directory."""

    # Define the file path using the provided directory
    file_path = pl.Path(directory).expanduser() / file

    # Check if the file exists
    if not file_path.exists():
        raise FileNotFoundError(f"No file found at {file_path}")

    # Load the data from the Parquet file
    df = pd.read_parquet(file_path)

    # Calculate delta
    df["delta"] = df["close"] - df["close"].shift(1)
    delta = df["delta"].bfill().fillna(0)

    # Add additional columns
    df["hourtime"] = df.index.strftime("%H:%M:%S")

    # Convert delta index to int64 increasing from 0 to len(delta)
    delta.index = np.arange(len(delta))

    return delta, df


# # Example usage:
# delta, df = get_test_data_df()  # Uses the default directory '~/svtmp/data'
# # Or specify a different directory
# delta, df = get_test_data_df(directory="/some/other/path")


def process_weights(config, delta, df) -> pd.DataFrame:
    """Process weights based on the configuration and data."""
    weights = genweights_df(delta, config.n_steps, config.n_basis)
    weights = weights[: len(df) - len(df) % config.n_steps]
    weights.index = df[: len(df) - len(df) % config.n_steps].index[:: config.n_steps]
    return weights.astype(np.float32)


def prepare_dataloader(config, weights):
    """Prepare the DataLoader from the processed weights."""
    dataset = TensorDataset(torch.tensor(weights.values))
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    return dataloader
