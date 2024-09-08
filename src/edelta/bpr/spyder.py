import sys

import matplotlib.pyplot as plt
import numpy as np
import pacmap
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.basefunc import genweights_df, get_test_data_df
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import TSNE

torch.manual_seed(1)
np.set_printoptions(precision=4, suppress=True)
pd.options.display.float_format = "{:.4f}".format


# from a df with index datetime, generate a column with minute of day
def get_minute_of_day(df):
    df["minute_of_day"] = df.index.hour * 60 + df.index.minute
    return df


# from minute of day column, rank absolute moves of delta column
def get_rank_abs_delta(df):
    df["rank_abs_delta"] = df["delta"].abs().rank()
    return df


if __name__ == "__main__":
    delta, df = get_test_data_df()
    df = df.fillna(0)

    df["minute_of_day"] = df.index.hour * 60 + df.index.minute
    df["rank_abs_delta"] = df["delta"].fillna(0).abs()

    # delta.plot(figsize=(11, 9))
    # plt.show()

    # scatter plot of minute of day vs rank of absolute delta
    # df.plot.scatter(x="minute_of_day", y="rank_abs_delta")
    # plt.show()

    print(df)

    # sys.exit()

    # delta.cumsum().plot(figsize=(11, 9))
    # delta.plot(secondary_y=True, alpha=0.6, color="red")
    # plt.show()
    # sys.exit()

    weights = genweights_df(delta, 32, 6)
    weights = weights[: len(df) - len(df) % 32]
    weights.index = df[: len(df) - len(df) % 32].index[::32]

    # weights = pd.DataFrame(
    #     np.random.standard_normal(size=weights.shape),
    #     index=range(0, len(weights)),
    # )
    print(weights)
    sys.exit()

    print(weights.std())

    print(weights.corr())

    print(weights.mean())

    print(weights.shape)

    X = weights.values
    X_embedded = TSNE(
        n_components=3, learning_rate="auto", init="random", perplexity=21
    ).fit_transform(X)
    embedding = pacmap.PaCMAP(
        n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0
    )
    X_embedded = embedding.fit_transform(X, init="pca")

    print(X_embedded.shape)
    model = KMeans(n_clusters=7)
    model.fit(weights)
    # print(model.labels_)
    # count labels
    print(pd.Series(model.labels_).value_counts())

    # plot the clusters in 2D with colors per cluster
    fig = plt.figure(figsize=(11, 9))

    plt.scatter(
        X_embedded[:, 0],
        X_embedded[:, 1],
        c=model.labels_,
        # s=1.6,
        alpha=0.7,
        cmap="Spectral",
    )
    plt.title("Pacmap + kmeans")
    plt.savefig("k_pm.png")
    plt.show()
    sys.exit()

    lin = nn.Linear(7, 14, bias=False)
    print(lin(torch.tensor(weights.values[32, :], dtype=torch.float32)))

    plt.scatter(X_embedded[:, 0], X_embedded[:, 1])
    plt.show()
    sys.exit()
    # weights.plot(figsize=(11, 9))
    # plt.show()

    # scatter plot of first two columns
    # plt.scatter(weights.iloc[:, 0], weights.iloc[:, 1])
    # plt.show()

    # get effective dimension
    X_cov = weights.T.cov()
    X_cov2 = weights.T.corr()
    N = X_cov.shape[0]

    # for i in range(N):
    #     for ii in range(N):
    #         X_cov.iloc[i, ii] = weights.iloc[i, :].corr(weights.iloc[ii, :])

    # mse diff between X_cov and X_cov2
    print(((X_cov - X_cov2) ** 2).mean().mean())

    print(X_cov.shape)
    U, S, V = np.linalg.svd(X_cov)
    S = pd.Series(S)
    S = S / S.sum()
    print(S.loc[0:11])
    # S.loc[0:100].plot(figsize=(11, 9))
    # plt.show()

    # count = 45
    # n_b = 5
    # N = count
    # dt = N / n_b
    # print(len(delta) / count)

    # csp = index_csum(delta, count)
    # basis = genbasis_df(dt, count, n_b)
    # basis.plot(figsize=(11, 9))
    # plt.show()
    # sys.exit()
    # # print(csp)
    # # print(basis)
    # # sys.exit()

    # lr = BayesianRidge(fit_intercept=False)

    # N_obs = 300
    # optx = csp.loc[N_obs * count : (N_obs + 1) * count - 1]
    # # print(optx)
    # optx.index = basis.index
    # lr.fit(basis, optx)
    # weights = lr.coef_
    # weights = np.around(weights, 4)
    # print(weights)
    # cts = (weights * basis).sum(1)

    # cts.plot(figsize=(11, 9))
    # optx.plot()
    # delta.iloc[
    #     N_obs * count : (N_obs + 1) * count - 1
    # ].reset_index().delta.cumsum().plot()
    # plt.show()

    # print(optx)
    # print(optx.shape)

    # print(csp.tail())
    # print(csp.shape)
    # csp[N_obs*count:(N_obs+1)*count].plot(figsize=(11, 9))
    # idxs[N_obs*count:(N_obs+1)*count].plot(alpha=0.6,color="red",secondary_y=True)
    # plt.show()
    # sys.exit()
    #
    # print(df.iloc[100:1000:50])
    # # print (df["sec"].value_counts())
    # daycs = df.groupby(str(df.index.strftime("%D"))).cumsum()
    # print(daycs.head())
    # print(df.index.day)
    # # sys.exit()
    #
    # print(df.resample("1D").mean())
    # print(df.resample("1D").mean().isnull().sum())
    # print(df.isnull().sum())
    # print(df.shape)
    # # sys.exit()
    #
    # delta.plot(figsize=(11, 9))
    # plt.show()
    #
    # print(df.head())
    # print(df.dtypes)
    # print(df.describe())
