# Cell 1: Imports and Initial Setup
import dask.dataframe as dd
import pandas as pd
import ray
import torch
import torch.nn as nn
import torch.optim as optim
from utils.basefunc import genbasis_df, get_test_data_df, index_csum

# Initialize Ray with GPU support
ray.init(num_gpus=1, ignore_reinit_error=True)


# Cell 2: PyTorch Model Definition
class SimpleLinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleLinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        return self.linear(x)


# Cell 3: Remote Function Definition
@ray.remote(num_gpus=1)
def compute_partition_weights_gpu(data_chunk, basis_df, chunk_size):
    input_dim = basis_df.shape[1]
    output_dim = data_chunk.shape[
        -1
    ]  # Assuming data_chunk has the same number of columns as basis_df
    model = SimpleLinearModel(input_dim, output_dim).cuda()  # Move model to GPU
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    weights_list = []
    index_list = []
    for start in range(0, len(data_chunk), chunk_size):
        end = start + chunk_size
        if end > len(data_chunk):
            break
        optx = torch.tensor(
            data_chunk.iloc[start:end].values, dtype=torch.float32
        ).cuda()  # Move data to GPU
        basis = torch.tensor(basis_df.values, dtype=torch.float32).cuda()

        for _ in range(10):  # Train for 10 epochs
            model.train()
            optimizer.zero_grad()
            outputs = model(basis)
            loss = criterion(outputs, optx)
            loss.backward()
            optimizer.step()

        weights_list.append(
            model.linear.weight.detach().cpu().numpy()
        )  # Move weights back to CPU
        index_list.append(data_chunk.index[end - 1])

    weights_df = pd.DataFrame(weights_list, index=index_list)
    return weights_df


# Cell 4: Generic Function to Apply Model
def apply_model(
    data_dask_df, basis_df, chunk_size, model_class=compute_partition_weights_gpu
):
    def process_partition(partition):
        # Process using Ray
        result = ray.get(model_class.remote(partition, basis_df, chunk_size))
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


# Cell 5: Function to Handle Specific CSP Case
def apply_model_with_exact_chunks(
    data_df, basis_df, chunk_size, model_class=compute_partition_weights_gpu
):
    # Ensure data_df length is an exact multiple of chunk_size
    num_full_chunks = len(data_df) // chunk_size
    truncated_length = num_full_chunks * chunk_size
    truncated_df = data_df.iloc[:truncated_length]

    # Convert to Dask DataFrame and partition it
    data_dask_df = dd.from_pandas(truncated_df, npartitions=num_full_chunks)

    # Apply the model
    weights_df = apply_model(data_dask_df, basis_df, chunk_size, model_class)

    return weights_df


# Cell 6: Example Usage
if __name__ == "__main__":
    chunk_size = 36
    num_basis_vectors = 5
    dt = chunk_size / num_basis_vectors

    # Generate test data
    delta_df, original_df = get_test_data_df()
    cumsum_df = index_csum(delta_df, chunk_size)

    # Generate basis DataFrame
    basis_df = genbasis_df(dt, chunk_size, num_basis_vectors)

    # Ensure basis is a Pandas DataFrame
    basis_df = basis_df.compute() if isinstance(basis_df, dd.DataFrame) else basis_df

    # Calculate weights for generic application
    weights_df_generic = apply_model(
        dd.from_pandas(cumsum_df, npartitions=32), basis_df, chunk_size
    )
    print("Generic Application Weights:")
    print(weights_df_generic)

    # Calculate weights for exact chunk application
    # weights_df_exact = apply_model_with_exact_chunks(cumsum_df, basis_df, chunk_size)
    # print("Exact Chunk Application Weights:")
    # print(weights_df_exact)

# Cell 7: Shutdown Ray
# ray.shutdown()
