# Cell 1: Imports and Initial Setup
import numpy as np
from sklearn.cluster import KMeans


# Generate synthetic data
def generate_synthetic_data(num_samples, dim):
    data = np.random.randn(num_samples, dim).astype(np.float32)
    return data


# Learn a dictionary using K-Means
def learn_dictionary(data, dictionary_size):
    kmeans = KMeans(n_clusters=dictionary_size, random_state=0).fit(data)
    return kmeans.cluster_centers_, kmeans.labels_


# Quantize the data using the learned dictionary
def quantize_data(data, dictionary):
    kmeans = KMeans(n_clusters=len(dictionary), init=dictionary, n_init=1)
    kmeans.fit(data)
    labels = kmeans.predict(data)
    quantized_data = dictionary[labels]
    return quantized_data, labels


# Cell 2: Main Function
if __name__ == "__main__":
    # Parameters
    num_samples = 10000
    dim = 10  # Dimensionality of the data
    dictionary_size = 50  # Size of the dictionary

    # Generate data
    data = generate_synthetic_data(num_samples, dim)

    # Learn the dictionary
    dictionary, labels = learn_dictionary(data, dictionary_size)
    print("Learned Dictionary:")
    print(dictionary)
    print(len(dictionary))

    # Quantize the data
    quantized_data, quantized_labels = quantize_data(data, dictionary)
    print("Quantized Data:")
    print(quantized_data)
    print(len(set(quantized_labels)))
