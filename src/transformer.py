import torch
import torch.nn as nn

class MiniTimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=2, model_dim=64, num_heads=4, num_layers=2, seq_length=28800):
        super(MiniTimeSeriesTransformer, self).__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.seq_length = seq_length
        
        # 位置编码
        self.position_embedding = nn.Parameter(torch.randn(1, seq_length, model_dim))
        
        # 输入变换
        self.input_projection = nn.Linear(input_dim, model_dim)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dim_feedforward=128)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出特征
        self.output_layer = nn.Linear(model_dim, model_dim)

    def forward(self, x):
        # x: [batch_size, seq_length, input_dim]
        x = self.input_projection(x) + self.position_embedding
        x = self.transformer_encoder(x)
        x = self.output_layer(x.mean(dim=1))  # 取时序的平均值作为全局特征
        return x

