import torch
import torch.nn as nn
import numpy as np
#/home/hu/Downloads/LOGDATA
class EnhancedTimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=2, model_dim=128, num_heads=8, num_layers=3, seq_length=10800):
        """
        A lightweight Transformer model for time series data.

        :param input_dim: Number of input features (e.g., 2 for CPU and GPU usage)
        :param model_dim: Dimension of the model's internal representation
        :param num_heads: Number of attention heads
        :param num_layers: Number of Transformer layers
        :param seq_length: Length of the input sequence
        """
        super(EnhancedTimeSeriesTransformer, self).__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.seq_length = seq_length
        self.position_encoding = self.create_positional_encoding(model_dim, seq_length)
        self.input_projection = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dim_feedforward=256, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(model_dim, model_dim)

    def create_positional_encoding(self, model_dim, seq_length):
        """
        Creates positional encoding for the input sequence.

        :param model_dim: Model's internal dimension
        :param seq_length: Length of the input sequence
        :return: Positional encoding tensor
        """
        position = torch.arange(seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2) * (-np.log(10000.0) / model_dim))
        pe = torch.zeros(seq_length, model_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        """
        Forward pass through the Transformer.

        :param x: Input tensor of shape [batch_size, seq_length, input_dim]
        :return: Output feature tensor
        """
        x = self.input_projection(x) + self.position_encoding
        x = self.transformer_encoder(x)
        x = self.output_layer(x.mean(dim=1))
        return x

