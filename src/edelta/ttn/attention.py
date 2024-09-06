import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, hidden, encoder_outputs):
        # hidden: (1, batch_size, hidden_size)
        # encoder_outputs: (seq_len, batch_size, hidden_size)
        seq_len = encoder_outputs.size(0)
        h = hidden.repeat(seq_len, 1, 1).transpose(0, 1)
        print(h)
        print(h.shape)
        print(encoder_outputs)
        print(encoder_outputs.shape)
        # sys.exit()
        to_attn = torch.cat([h, encoder_outputs])
        energy = torch.tanh(self.attn(to_attn, dim=-1))
        attention = F.softmax(energy.matmul(self.v), dim=1)
        return attention.transpose(1, 2)

    # attn_weight = query @ key.transpose(-2, -1) * scale_factor
    # attn_weight += attn_bias
    # attn_weight = torch.softmax(attn_weight, dim=-1)
    # attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    # return attn_weight @ value


class AttentionAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim, hidden_size):
        super(AttentionAutoencoder, self).__init__()

        # Encoder
        self.encoder_rnn = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.encoder = nn.Linear(hidden_size, encoding_dim)

        # Attention mechanism
        self.attention = Attention(hidden_size)
        # self.attention = MultiheadAttention(hidden_size,num_heads=2)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(
                encoding_dim + hidden_size, hidden_size
            ),  # +hidden_size for attention context
            nn.ReLU(),
            nn.Linear(hidden_size, input_dim),
        )

    def forward(self, x):
        # Encoder part
        encoder_outputs, (hidden, cell) = self.encoder_rnn(x)
        encoded = self.encoder(
            hidden.squeeze(0)
        )  # Use the last hidden state for encoding

        # Attention mechanism
        context_vector = self.attention(hidden, encoder_outputs)

        # Decoder part
        # Concatenate the encoded vector with the context vector from attention
        decoder_input = torch.cat((encoded.unsqueeze(1), context_vector), dim=-1)
        decoded = self.decoder(
            decoder_input.squeeze(1)
        )  # Assuming we're decoding to predict the next step or reconstruct

        return encoded, decoded
