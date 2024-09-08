import math

import numpy as np
from scipy.integrate import quad
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde


def histogram_entropy(data, bins=10):
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    hist = hist[hist > 0]  # filter out zero entries
    probabilities = hist / hist.sum()
    entropy = -np.sum(probabilities * np.log(probabilities))
    return entropy


def kde_entropy(data, bandwidth="scott"):
    kde = gaussian_kde(data, bw_method=bandwidth)

    # Function to compute the density at a point
    def density(x):
        return kde(x)

    # Integrating the density function over its domain
    result, _ = quad(
        lambda x: -density(x) * np.log(density(x)), np.min(data), np.max(data)
    )
    return result


def delta_entropy(data):
    unique, counts = np.unique(data, return_counts=True)
    probabilities = counts / len(data)
    entropy = -np.sum(probabilities * np.log(probabilities))
    return entropy


def knn_entropy(data, k=3):
    N, d = data.shape
    tree = cKDTree(data)
    distances, _ = tree.query(data, k=k + 1)
    avg_log_dist = np.mean(np.log(distances[:, -1]))
    entropy = (
        d * avg_log_dist
        + np.log(N)
        - np.log(np.pi ** (d / 2))
        + np.log(2 ** (d - 1))
        - np.log(k)
        - np.log(math.gamma(d / 2 + 1))
    ) / N
    return entropy


def plugin_entropy(data, bins=10):
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    probabilities = hist / np.sum(hist)
    entropy = -np.sum(
        probabilities * np.log(probabilities + 1e-10)
    )  # Adding a small value to avoid log(0)
    return entropy


def gaussian_mle_entropy(data):
    mu = np.mean(data)
    sigma = np.std(data)
    entropy = 0.5 * np.log(2 * np.pi * np.e * sigma**2)
    return entropy


# Normalization Functions


def max_min_normalize(entropy_values):
    min_val = np.min(entropy_values)
    max_val = np.max(entropy_values)
    return (entropy_values - min_val) / (max_val - min_val)


def z_score_normalize(entropy_values):
    mean_val = np.mean(entropy_values)
    std_val = np.std(entropy_values)
    return (entropy_values - mean_val) / std_val


def normalized_entropy(entropy, num_bins):
    max_entropy = np.log(num_bins)
    return entropy / max_entropy


# Example usage
if __name__ == "__main__":
    data = np.random.randn(10000)

    entropies = {
        "Histogram": histogram_entropy(data, bins=30),
        "KDE": kde_entropy(data),
        "Delta": delta_entropy(data),
        "KNN": knn_entropy(np.random.randn(1000, 2), k=5),
        "Plugin": plugin_entropy(data, bins=30),
        "Gaussian MLE": gaussian_mle_entropy(data),
    }

    entropy_values = np.array(list(entropies.values()))
    print("Raw Entropies:", entropies)

    normalized_max_min = max_min_normalize(entropy_values)
    print(
        "Max-Min Normalized Entropies:", dict(zip(entropies.keys(), normalized_max_min))
    )

    normalized_z_score = z_score_normalize(entropy_values)
    print(
        "Z-Score Normalized Entropies:", dict(zip(entropies.keys(), normalized_z_score))
    )

    normalized_relative = {
        k: normalized_entropy(v, num_bins=20)
        for k, v in entropies.items()
        if k != "KNN"
    }  # assuming 'KNN' entropy isn't normalized here
    print("Normalized Relative Entropies:", normalized_relative)
