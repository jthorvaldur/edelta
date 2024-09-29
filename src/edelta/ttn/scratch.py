import numpy as np
import warnings

import torch
import torch.nn.functional as F


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)

    scores = torch.matmul(
        query,
        key.transpose(-2, -1) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32)),
    )

    attention_weights = F.softmax(scores, dim=1)
    output = torch.matmul(attention_weights, value)
    return output, attention_weights


query = torch.tensor([[1.0, 0.0, 0.0]])


# from basecls import *

# Assuming df is your DataFrame with shape [1e6, 8]

warnings.filterwarnings("ignore")
np.random.seed(42)
# df = pd.DataFrame(np.random.rand(1024, 8)).astype(np.float32)
#
# N = 1024 * 4
# n_basis = 8
# df_data = np.random.rand(N, n_basis)
# index = np.arange(N)
# for i in range(n_basis):
#     df_data[:, i] = np.sin(512 * index / N) + np.pi * i / 8
#
# df = pd.DataFrame(df_data).astype(np.float32)
# # df -= df.mean()
# df /= df.abs().max()
# df = np.tanh(df * 2)


np_q = np.random.randn(4, 1)
np_v = np.random.randn(32, 8).astype(np.float32)

# print(np_q @ np_v)


v = torch.tensor(np_v, dtype=torch.float32)
q = torch.tensor(np_q, dtype=torch.float32)
q = q.repeat(v.size(0), 1, 1)

# print(q)
# print(v)
print(q.shape, v.shape)
print(q.matmul(v))
# print(v.squeeze(-1))

# print(q.dot(v.squeeze(0)))
