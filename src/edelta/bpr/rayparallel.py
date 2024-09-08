import dask.dataframe as dd
import numpy as np
import pacmap
import pandas as pd
import ray
from utils.basefunc import *
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.linear_model import BayesianRidge
from sklearn.manifold import TSNE

# Initialize Ray
ray.init()


@ray.remote
def compute_weights(delta_chunk, count, n_b):
    dt = count / n_b
    csp = index_csum(delta_chunk, count)
    basis = genbasis_df(dt, count, n_b)
    lr = BayesianRidge(fit_intercept=False)

    weights = []
    index_list = []
    for i in range(0, len(delta_chunk), count):
        if i + count - 1 >= int(len(delta_chunk) * 0.9):
            break
        optx = csp.loc[i : i + count - 1] * 1.0
        last_idx = optx.index[-1]
        optx.index = basis.index
        lr.fit(basis, optx)
        weights.append(np.around(lr.coef_, 6))
        index_list.append(last_idx)

    weights = np.array(weights)
    weights_df = pd.DataFrame(weights, index=index_list)

    return weights_df


def gencweights_df(delta, count, n_b):
    # Split delta into chunks
    num_chunks = len(delta) // count
    delta_chunks = [delta.iloc[i * count : (i + 1) * count] for i in range(num_chunks)]

    # If there is a remainder chunk, add it as well
    if len(delta) % count != 0:
        delta_chunks.append(delta.iloc[num_chunks * count :])

    # Launch Ray tasks
    results = ray.get(
        [compute_weights.remote(chunk, count, n_b) for chunk in delta_chunks]
    )

    # Combine results
    weights = pd.concat(results)

    return weights


if __name__ == "__main__":
    delta, df = get_test_data_df()
    delta = dd.from_pandas(delta, npartitions=4)
    df = dd.from_pandas(df, npartitions=4)
    weights = gencweights_df(delta, 36, 9)

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

    ray.shutdown()
