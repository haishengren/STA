import math
import torch
import numpy as np
import torch.nn as nn


# Indexing of SMILES
def smiles_to_index(smiles, char_to_index, length):
    smiles_idx = np.zeros(length, dtype=np.int32)
    t = 0
    idx = 0
    while t < len(smiles) and idx < length:
        if t < len(smiles) - 1 and smiles[t:t + 2] in char_to_index:
            smiles_idx[idx] = char_to_index[smiles[t:t + 2]]
            t += 2
        else:
            smiles_idx[idx] = char_to_index[smiles[t]]
            t += 1
        idx += 1
    return smiles_idx


class PositionalEncoding(nn.Module):
    def __init__(self, length, d_model):
        super(PositionalEncoding, self).__init__()
        self.length = length
        self.d_model = d_model
        self.register_buffer('pos_table', self.get_sinusoid_encoding_table())

    def get_sinusoid_encoding_table(self):
        """Create sinusoid position encoding table"""
        position = torch.arange(0, self.length).float().unsqueeze(1)  # Shape: (n_position, 1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * -(math.log(10000.0) / self.d_model))

        pos_table = torch.zeros(self.length, self.d_model)  # Shape: (length, d_model)
        pos_table[:, 0::2] = torch.sin(position * div_term)  # sin for even positions
        pos_table[:, 1::2] = torch.cos(position * div_term)  # cos for odd positions

        return torch.FloatTensor(pos_table).unsqueeze(0)  # Shape: (1, length, d_model)

    def forward(self, x):
        return x + self.pos_table  # Add the positional encoding to SMILES embedding


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super(EncoderLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(d_model)
        self.ff_layers = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.ReLU(), nn.Linear(4 * d_model, d_model))

    def forward(self, attn_intput, key_padding_mask):
        attn_output, _ = self.attention(query=attn_intput, key=attn_intput, value=attn_intput,
                                        key_padding_mask=key_padding_mask)

        output_1 = self.layer_norm(attn_intput + attn_output)
        output_2 = self.layer_norm(output_1 + self.ff_layers(output_1))
        return output_2


class Model(nn.Module):
    def __init__(self, vocab_size, length, d_model, num_heads, num_layers):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(length, d_model)
        self.attention_layers = nn.ModuleList([EncoderLayer(d_model, num_heads) for _ in range(num_layers)])
        self.prediction_head = nn.Linear(d_model, 1)

    def forward(self, input_index_vector):

        # padding mask
        key_padding_mask = input_index_vector == 0

        # Get the embeddings and scale by sqrt(d_model)
        embeddings = self.embedding(input_index_vector) * math.sqrt(self.d_model)
        input_to_encoder = self.pos_encoding(embeddings)

        # Iterate through the encoder layer
        output_from_encoder = input_to_encoder
        for layer in self.attention_layers:
            output_from_encoder = layer.forward(output_from_encoder, key_padding_mask=key_padding_mask)

        prediction_output = self.prediction_head(output_from_encoder)

        # Ignore the padding
        key_padding_mask = key_padding_mask.unsqueeze(-1).to(torch.float32)
        summed_output = (prediction_output * (1 - key_padding_mask)).sum(dim=1)

        return summed_output, output_from_encoder
