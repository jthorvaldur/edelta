import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error

# Assuming you have your trained model and data prepared as before
# model = ... (your trained TransformerModel)
# data = ... (your dataset)

# Prepare test data
test_size = 1000  # Use a subset for testing, adjust as needed
test_data = data[-test_size:]
sequences = test_data[:-1]
targets = test_data[1:]

# Create DataLoader for test data
test_dataset = TensorDataset(sequences, targets)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Function to predict and compute MSE for each batch
def predict_and_compute_mse(model, test_loader):
    mse_errors = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(
                inputs
            )  # Assuming model outputs [batch_size, seq_len, feature_dim]
            # Extract the last prediction for each sequence
            predictions = outputs[:, -1, :]  # Shape: [batch_size, feature_dim]

            # Compute MSE for each vector prediction
            mse = mean_squared_error(
                targets.cpu().numpy(),
                predictions.cpu().numpy(),
                multioutput="raw_values",
            )
            mse_errors.append(mse)

    return np.array(mse_errors)


# Run the test
mse_errors = predict_and_compute_mse(model, test_loader)

# Analyze results
# Flatten the list of MSE errors for each feature across all batches
flattened_mse = np.concatenate(mse_errors)

# Compute statistics
avg_mse = np.mean(flattened_mse)
std_mse = np.std(flattened_mse)
min_mse = np.min(flattened_mse)
max_mse = np.max(flattened_mse)

print(f"Average MSE across all features and predictions: {avg_mse}")
print(f"Standard Deviation of MSE: {std_mse}")
print(f"Minimum MSE: {min_mse}")
print(f"Maximum MSE: {max_mse}")

# Optionally, plot the distribution of MSE errors
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(flattened_mse, bins=100, edgecolor="black")
plt.title("Distribution of MSE Errors for Next Vector Prediction")
plt.xlabel("MSE")
plt.ylabel("Frequency")
plt.show()

# If you want to see MSE for each feature separately
for i, feature_mse in enumerate(
    mse_errors[0]
):  # Assuming first batch is representative
    print(f"MSE for feature {i}: {np.mean([errors[i] for errors in mse_errors])}")
