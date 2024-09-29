# Cell 1: Imports and Initial Setup
import os

import dask.dataframe as dd
import numpy as np
import pandas as pd
import ray
from utils.basefunc import genbasis_df, get_test_data_df, index_csum
from genutil import perf
from sklearn.linear_model import BayesianRidge

# Initialize Ray
ray.init(ignore_reinit_error=True, num_gpus=1)

# NUMEXPR_MAX_THREADS = 64
os.environ["NUMEXPR_MAX_THREADS"] = str(64)


# Cell 2: Remote Function Definition
@ray.remote
def compute_partition_weights(
    data_chunk, basis_df, chunk_size, model_class, *model_args, **model_kwargs
):
    model = model_class(*model_args, **model_kwargs)

    weights_list = []
    index_list = []
    for start in range(0, len(data_chunk), chunk_size):
        end = start + chunk_size
        if end > len(data_chunk):
            break
        optx = data_chunk.iloc[start:end] * 1.0
        last_idx = optx.index[-1]
        optx.index = basis_df.index
        model.fit(basis_df, optx)
        weights_list.append(np.around(model.coef_, 6))
        index_list.append(last_idx)

    weights_df = pd.DataFrame(weights_list, index=index_list)
    return weights_df


# Cell 3: Generic Function to Apply Model
@perf.timeit  # Measure the time taken to execute this function
def apply_model(
    data_dask_df,
    basis_df,
    chunk_size,
    model_class=BayesianRidge,
    *model_args,
    **model_kwargs,
):
    def process_partition(partition):
        # Process using Ray
        result = ray.get(
            compute_partition_weights.remote(
                partition,
                basis_df,
                chunk_size,
                model_class,
                *model_args,
                **model_kwargs,
            )
        )
        return result

    # Define the expected metadata
    num_basis_vectors = basis_df.shape[1]
    meta_columns = list(range(num_basis_vectors))
    meta = pd.DataFrame(columns=meta_columns)

    # Apply the function to each partition
    results = data_dask_df.map_partitions(process_partition, meta=meta)

    # Compute the results
    weights_df = results.compute()

    return weights_df


# Cell 4: Function to Handle Specific CSP Case
@perf.timeit  # Measure the time taken to execute this function
def apply_model_with_exact_chunks(
    data_df,
    basis_df,
    chunk_size,
    model_class=BayesianRidge,
    *model_args,
    **model_kwargs,
):
    # Ensure data_df length is an exact multiple of chunk_size
    num_full_chunks = len(data_df) // chunk_size
    truncated_length = num_full_chunks * chunk_size
    truncated_df = data_df.iloc[:truncated_length]
    num_full_chunks = min(num_full_chunks, 128)
    # print(num_full_chunks)
    # sys.exit()

    # Convert to Dask DataFrame and partition it
    data_dask_df = dd.from_pandas(truncated_df, npartitions=num_full_chunks)

    # Apply the model
    weights_df = apply_model(
        data_dask_df,
        basis_df,
        chunk_size,
        model_class,
        *model_args,
        **model_kwargs,
    )

    return weights_df


# Cell 5: Example Usage
if __name__ == "__main__":
    chunk_size = 90
    num_basis_vectors = 5
    dt = chunk_size / num_basis_vectors

    cpus = int(ray.available_resources()["CPU"]) * 1
    print(f"Number of CPUs: {cpus}")
    # sys.exit()

    # Generate test data
    delta_df, original_df = get_test_data_df()

    # size_n = 1_000_000
    # # size_n mod chunk_size should be 0
    # size_n = size_n - (size_n % chunk_size)
    # delta_df = pd.Series(np.random.standard_normal(size=size_n))

    cumsum_df = index_csum(delta_df, chunk_size)

    # Generate basis DataFrame
    basis_df = genbasis_df(dt, chunk_size, num_basis_vectors)

    # Ensure basis is a Pandas DataFrame
    basis_df = basis_df.compute() if isinstance(basis_df, dd.DataFrame) else basis_df

    # Calculate weights for generic application
    weights_df_generic = apply_model(
        dd.from_pandas(cumsum_df, npartitions=cpus), basis_df, chunk_size
    )
    weights_df_generic = apply_model(
        dd.from_pandas(cumsum_df, npartitions=cpus), basis_df, chunk_size
    )
    # print("Generic Application Weights:")
    # print(weights_df_generic)

    # Calculate weights for exact chunk application
    weights_df_exact = apply_model_with_exact_chunks(cumsum_df, basis_df, chunk_size)
    # print("Exact Chunk Application Weights:")
    # print(weights_df_exact)

# Cell 6: Shutdown Ray
# ray.shutdown()
