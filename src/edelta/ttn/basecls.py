import math

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import entropy


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(0), :]


def compute_weight_entropy(model):
    entropies = []
    for name, param in model.named_parameters():
        if "weight" in name:
            flat_weights = param.data.cpu().numpy().flatten()
            hist, _ = np.histogram(flat_weights, bins=100, density=True)
            ent = entropy(hist)
            entropies.append((name, ent))
    return entropies


def train_model(epochs, dataloader, optimizer, model, criterion):
    # Training loop
    loss = 0
    for epoch in range(epochs):
        for batch in dataloader:
            inputs, targets = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(
                outputs, targets
            )  # + model.entropy_regularization(0.01)  # Using entropy as regularization
            loss.backward()
            optimizer.step()

        #     # Track entropy
        if epoch < 10 or epoch % 10 - 1 == 0:
            # weight_entropies = compute_weight_entropy(model)
            # entropy_history.append(weight_entropies)
            print(f"Epoch {epoch}, Loss: {loss.item()}")


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        dropout=0.1,
    ):
        super(TransformerModel, self).__init__()
        self.model_type = "Transformer"
        self.d_model = d_model
        self.src_mask = None

        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model,
            nhead,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            dropout,
        )
        self.encoder_input_layer = nn.Linear(input_dim, d_model)
        self.decoder_output_layer = nn.Linear(d_model, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder_input_layer.weight.data.uniform_(-initrange, initrange)
        self.decoder_output_layer.weight.data.uniform_(-initrange, initrange)

    def forward1(self, src):
        src = self.dropout(self.encoder_input_layer(src) * math.sqrt(self.d_model))
        src = self.pos_encoder(src)
        output = self.transformer(
            src, src
        )  # Using src for both src and tgt for autoencoder-like setup
        decoded = self.decoder_output_layer(output)
        return decoded

    def entropy_regularization(self, lambda_reg):
        reg_loss = 0
        for param in self.parameters():
            if param.requires_grad:
                flat_weights = param.data.view(-1)
                prob = torch.nn.functional.softmax(flat_weights, dim=0)
                reg_loss += -lambda_reg * torch.sum(prob * torch.log(prob + 1e-10))
        return reg_loss

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder_input_layer(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer(
            src, src, src_mask=self.src_mask
        )  # Here, src is used for both src and tgt for autoencoder
        decoded = self.decoder_output_layer(output)
        return decoded  # Directly return the decoded sequence for reconstruction

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask
