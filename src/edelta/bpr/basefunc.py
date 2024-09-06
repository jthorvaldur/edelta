import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime as dt
import pathlib as pl
import sys
from sklearn.linear_model import LinearRegression, BayesianRidge, LassoLars


def relu(x, shift=0):
    val = np.maximum(0, x - shift)
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


# @timeit
def import_if_exists(directory, file):
    strip_file = file.replace(".csv", "")
    file_path = os.path.join(directory, strip_file)

    # open parquet if exists
    if os.path.exists(file_path + ".parquet"):
        df = pd.read_parquet(file_path + ".parquet")
        return df
    else:
        # open csv if exists
        df = pd.read_csv(file_path + ".csv")
        # convert to parquet
        parquet_file = file_path + ".parquet"
        # convert date to datetime
        df["datetime"] = pd.to_datetime(df["datetime"])
        # set date as index
        df.set_index("datetime", inplace=True)
        # save to parquet
        df.to_parquet(os.path.join(directory, parquet_file))
        return df


def genbasis_df(dt, count, n_b):
    basis = pd.DataFrame()
    idxsample = pd.Series(np.arange(count))
    for i in range(n_b):
        basis[i] = relu(idxsample.index, int(i * dt))

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
    directory = pl.Path(__file__).parent
    print(directory)
    # sys.exit()
    file = "spy.csv"
    df = import_if_exists(directory, file)
    df["delta"] = df["close"] - df["close"].shift(1)
    delta = pd.Series(df["delta"].bfill().fillna(0))
    df["hourtime"] = df.index.strftime("%H:%M:%S")
    # df["sec"] = df.index.timestamp()

    # convert delta index to int64 increasing from 0 to len(delta)
    delta.index = np.arange(len(delta))
    return delta, df
