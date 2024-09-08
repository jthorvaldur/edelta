import datetime as dt
import os
import pathlib as pl
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error


def relu(x, shift=0):
    val = np.maximum(0, x - shift)
    return val


def tanh(x, shift=0):
    val = (np.tanh((x - shift) * 0.5) + 0) / 1
    return val


def sigmoid(x, shift=0):
    val = 1 / (1 + np.exp(-(x - shift)))
    return val


def activation(x, shift=0, func="relu"):
    if func == "relu":
        val = relu(x, shift)
    elif func == "tanh":
        val = tanh(x, shift)
    elif func == "sigmoid":
        val = sigmoid(x, shift)
    else:
        val = x
    return val


def cumsum_by_day(x: pd.Series):
    i_day = np.int64(x.index.day.values)
    dlist = np.where(i_day[:-1] != i_day[1:])[0]
    y = x * 0
    y.values[dlist] = x.cumsum().iloc[dlist].values[:]
    y[y == 0] = np.nan
    y = y.ffill().fillna(0)
    return x.cumsum() - y


def index_csum(x: pd.Series, count: int):
    i_day = np.int64(x.index.values)
    dlist = np.where([i % count == 0 for i in i_day])[0]
    y = x * 0
    y.values[dlist] = x.cumsum().iloc[dlist].values[:]
    y[y == 0] = np.nan
    y = y.ffill().fillna(0)
    return x.cumsum() - y


def import_if_exists(directory, file):
    directory = os.path.expanduser(directory)
    strip_file = file.replace(".csv", "").replace(".parquet", "")
    parquet_path = os.path.join(directory, f"{strip_file}.parquet")
    csv_path = os.path.join(directory, f"{strip_file}.csv")

    # Check if the .parquet file exists
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        return df
    # If .parquet doesn't exist, check for .csv
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Convert to parquet for future use
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        df.to_parquet(parquet_path)
        return df
    else:
        raise FileNotFoundError(f"No file found at {parquet_path} or {csv_path}")


def genbasis_df(dt, count, n_b):
    basis = pd.DataFrame()
    idxsample = pd.Series(np.arange(count))
    for i in range(n_b):
        basis[i] = activation(idxsample.index, int(i * dt), "relu")
    return basis


def gencbasis_df(dt, count, n_b):
    basis = genbasis_df(dt, count, n_b)
    U, S, V = np.linalg.svd(basis.cov())
    basis = basis @ U / S**0.5
    return basis


def count_nans_by_column(df):
    return df.isnull().sum()


def get_test_data_df():
    # get directory and file, then use references
    # local relative path directory = pl.Path(__file__).parent
    directory = pl.Path("~/svtmp/data/")
    file = "spy.parquet"
    df = import_if_exists(directory, file)
    df["delta"] = df["close"] - df["close"].shift(1)
    delta = pd.Series(df["delta"].bfill().fillna(0)).fillna(0)
    df["hourtime"] = df.index.strftime("%H:%M:%S")

    # convert delta index to int64 increasing from 0 to len(delta)
    delta.index = np.arange(len(delta))
    return delta, df


def genxweights_df(delta, count, n_b) -> pd.DataFrame:
    N = len(delta)

    dt = count / n_b
    csp = index_csum(delta, count)
    basis = genbasis_df(dt, count, n_b)
    lr = BayesianRidge(fit_intercept=False)

    weights = []
    index_list = []
    for i in range(0, N, count):
        if i + count - 1 >= N:
            break
        optx = csp.loc[i : i + count - 1] * 1.0
        last_idx = optx.index[-1]
        optx.index = basis.index
        lr.fit(basis, optx)
        weights.append(np.around(lr.coef_, 6))
        index_list.append(last_idx)

    weights = np.array(weights)
    weights = pd.DataFrame(weights, index=index_list)

    return weights


def genweights_df(delta: pd.DataFrame, count: int, n_b: int) -> pd.DataFrame:
    N = len(delta)

    dt = count / n_b
    csp = index_csum(delta, count)
    basis = gencbasis_df(dt, count, n_b)
    lr = BayesianRidge(fit_intercept=False)

    weights = []
    index_list = []
    for i in range(0, N, count):
        if i + count - 1 >= N:
            break
        optx = csp.loc[i : i + count - 1] * 1.0
        last_idx = optx.index[-1]
        optx.index = basis.index
        lr.fit(basis, optx)
        weights.append(np.around(lr.coef_, 6))
        index_list.append(last_idx)

    weights = np.array(weights)
    weights = pd.DataFrame(weights, index=index_list)

    return weights


def project_and_plot(weights, basis, delta, count):
    """Project weights back to the basis and plot the original vs. reconstructed series."""
    reconstructed_series = np.dot(weights, basis.T)

    # Create a DataFrame for the reconstructed series
    reconstructed_df = pd.DataFrame(reconstructed_series, index=weights.index)

    # Plot the reconstructed series
    plt.figure(figsize=(12, 6))
    for i in range(reconstructed_df.shape[0]):
        plt.plot(reconstructed_df.iloc[i], label=f"Reconstructed {i+1}")

    # Plot the original cumulative sum series
    csp = index_csum(delta, count)
    plt.plot(csp.values, label="Original Series", color="black", linestyle="--")

    plt.title("Original vs. Reconstructed Series")
    plt.legend()
    plt.show()


def project_and_plot_single(w_old, w_new, basis, delta, count, index):
    """Project a single weight vector back to the basis and plot the original vs. reconstructed series."""

    # Project the single weight vector back onto the basis
    reconstructed_series_old = np.dot(w_old, basis.T)
    reconstructed_series_new = np.dot(w_new, basis.T)

    # Create a Series for the reconstructed series
    reconstructed_series_old = pd.Series(reconstructed_series_old, index=basis.index)
    reconstructed_series_new = pd.Series(reconstructed_series_new, index=basis.index)

    # Calculate MSE between the old and new reconstructed series
    mse_old_new = mean_squared_error(
        reconstructed_series_old.values, reconstructed_series_new.values
    )

    # Plot the reconstructed series
    plt.figure(figsize=(10, 6))
    plt.plot(
        reconstructed_series_old,
        label=f"Reconstructed Series {index+1} (Old)",
        color="blue",
    )
    plt.plot(
        reconstructed_series_new,
        label=f"Reconstructed Series {index+1} (New)",
        color="red",
    )

    # Plot the original cumulative sum series
    csp = index_csum(delta, count).iloc[index * count : (index + 1) * count]
    plt.plot(csp.values, label="Original Series", color="black", linestyle="--")

    # Display MSE between old and new in the title
    plt.title(
        f"Old vs. New Reconstructed Series for Weight Vector {index+1}\nMSE (Old vs. New): {mse_old_new:.4f}"
    )
    plt.legend()
    plt.show()


# Example usage within your pipeline
if __name__ == "__main__":
    # Generate some test data
    delta, df = get_test_data_df()

    # Generate weights and basis
    count = 32
    n_basis = 6
    weights = genweights_df(delta, count, n_basis)
    basis = genbasis_df(count / n_basis, count, n_basis)

    # Project and plot the results
    # project_and_plot(weights, basis, delta, count)

    # Project and plot a single weight vector
