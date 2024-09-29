import torch.nn as nn


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, dim_feedforward, num_layers):
        super(TransformerEncoderLayer, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=0.1,
                # batch_first=True,
                # norm_first=not True,
            ),
            num_layers=num_layers,
        )
        for name, param in self.transformer.named_parameters():
            # print(name, param.data)
            if "weight" in name and param.data.dim() == 2:
                nn.init.kaiming_uniform_(param)

    def forward(self, x):
        return self.transformer(x)
