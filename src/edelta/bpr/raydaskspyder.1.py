import dask.dataframe as dd
import numpy as np
import pandas as pd
import ray
from utils.basefunc import genbasis_df, get_test_data_df, index_csum
from sklearn.linear_model import BayesianRidge

# Initialize Ray
ray.init()


@ray.remote
def compute_partition_weights(csp_chunk, basis_df, chunk_size):
    model = BayesianRidge(fit_intercept=False)

    weights_list = []
    index_list = []
    for start in range(0, len(csp_chunk), chunk_size):
        end = start + chunk_size
        if end > len(csp_chunk):
            break
        optx = csp_chunk.iloc[start:end] * 1.0
        last_idx = optx.index[-1]
        optx.index = basis_df.index
        model.fit(basis_df, optx)
        weights_list.append(np.around(model.coef_, 6))
        index_list.append(last_idx)

    weights_df = pd.DataFrame(weights_list, index=index_list)
    return weights_df


def generate_weights_df(csp_dask_df, basis_df, chunk_size, num_basis_vectors):
    def process_partition(partition):
        # Process using Ray
        result = ray.get(
            compute_partition_weights.remote(partition, basis_df, chunk_size)
        )
        return result

    # Define the expected metadata
    meta_columns = list(range(num_basis_vectors))
    meta = pd.DataFrame(columns=meta_columns)

    # Apply the function to each partition
    results = csp_dask_df.map_partitions(process_partition, meta=meta)

    # Compute the results
    weights_df = results.compute()

    return weights_df


if __name__ == "__main__":
    chunk_size = 35
    num_basis_vectors = 7
    dt = chunk_size / num_basis_vectors

    # Generate test data
    delta_df, original_df = get_test_data_df()
    cumsum_df = index_csum(delta_df, chunk_size)

    # Number of partitions
    num_partitions = 35

    # Convert to Dask DataFrame and partition it
    csp_dask_df = dd.from_pandas(cumsum_df, npartitions=num_partitions)

    # Generate basis DataFrame
    basis_df = genbasis_df(dt, chunk_size, num_basis_vectors)

    # Ensure basis is a Pandas DataFrame
    basis_df = basis_df.compute() if isinstance(basis_df, dd.DataFrame) else basis_df

    # Calculate weights
    weights_df = generate_weights_df(
        csp_dask_df, basis_df, chunk_size, num_basis_vectors
    )

    print(weights_df)

    # Remember to shutdown Ray after use
    ray.shutdown()
