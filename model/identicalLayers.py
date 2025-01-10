import torch
import torch.nn as nn
import torch.nn.functional as F

from subLayers import PositionWiseFeedForward, MultiHeadSelfAttention

class EncoderLayer(nn.Module):
    def __init__(self, d_model, hidden, n_head, dropout=0.1):
        super().__init__()

        self.attention = MultiHeadSelfAttention(d_model, n_head, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = PositionWiseFeedForward(d_model, hidden, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        _x = self.attention(q=x, k=x, v=x)
        _x = self.dropout1(_x)
        x = self.norm1(x + _x)

        _x = self.ffn(x)
        _x = self.dropout2(_x)
        x = self.norm2(x + _x)

        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, hidden, n_head, dropout=0.1):
        super().__init__()

        self.maskedAttention = MultiHeadSelfAttention(d_model, n_head, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.encoderDecoderAttention = MultiHeadSelfAttention(d_model, n_head, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.ffn = PositionWiseFeedForward(d_model, hidden, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, encoder, x, mask):
        _x = self.maskedAttention(q=x, k=x, v=x, mask=mask)
        _x = self.dropout1(_x)
        x = self.norm1(x + _x)

        _x = self.encoderDecoderAttention(q=x, k=encoder, v=encoder)
        _x = self.dropout2(_x)
        x = self.norm2(x + _x)

        _x = self.ffn(x)
        _x = self.dropout3(_x)
        x = self.norm3(x + _x)

        return x