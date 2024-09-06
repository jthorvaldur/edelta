import warnings

import matplotlib.pyplot as plt
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from attention import AttentionAutoencoder
from basecls import *

# Assuming df is your DataFrame with shape [1e6, 8]

warnings.filterwarnings("ignore")

df = pd.DataFrame(np.random.rand(1024 * 8, 8)).astype(np.float32)

N = 1024 * 4
n_basis = 8
df_data = np.random.rand(N, n_basis)
index = np.arange(N)
for i in range(n_basis):
    df_data[:, i] = np.sin(512 * index / N) + np.pi * i / 8

df = pd.DataFrame(df_data).astype(np.float32)
# df -= df.mean()
df /= df.abs().max()
df = np.tanh(df * 2)
# df.plot()
# plt.show()
# sys.exit()
data = torch.tensor(df.values, dtype=torch.float32)

# Hyperparameters
input_dim = n_basis
d_model = 32
nhead = 4
num_encoder_layers = 2
num_decoder_layers = 2
dim_feedforward = 512
dropout = 0.05
batch_size = 16
epochs = 12
learning_rate = 0.001

# Data preparation for sequence prediction (next step prediction)
sequences = data[:-1].view(-1, 1, n_basis)
targets = data[1:].view(-1, 1, n_basis)

dataset = TensorDataset(sequences, targets)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

model = TransformerModel(
    input_dim,
    d_model,
    nhead,
    num_encoder_layers,
    num_decoder_layers,
    dim_feedforward,
    dropout,
)
model = AttentionAutoencoder(input_dim, d_model, hidden_size=128)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
train_model(epochs, dataloader, optimizer, model, criterion)

with torch.no_grad():
    for inputs, targets in dataloader:
        outputs = model(
            inputs
        )  # Assuming model outputs [batch_size, seq_len, feature_dim]
        # Extract the last prediction for each sequence
        predictions = outputs[:, -1, :]  # Shape:
        inputs = inputs[:, -1, :]

predictions = pd.DataFrame(predictions, columns=np.arange(n_basis))
inputs = pd.DataFrame(inputs, columns=np.arange(n_basis))

print(inputs.shape, predictions.shape)
for i in range(n_basis):
    predictions.iloc[:, i].plot()
    inputs.iloc[:, i].plot()
plt.show()

# # Plotting entropy over time for visualization
# plt.figure(figsize=(10, 5))
# for name, _ in entropy_history[0]:
#     entropies_over_time = [epoch_data for epoch_data in entropy_history if name in dict(epoch_data)]
#     if entropies_over_time:
#         epochs, ents = zip(*[(i*10, ent) for i, epoch_data in enumerate(entropies_over_time) for n, ent in epoch_data if n == name])
#         plt.plot(epochs, ents, label=name)
# plt.xlabel('Epochs')
# plt.ylabel('Entropy')
# plt.legend()
# plt.title('Weight Entropy Over Training')
# plt.show()
