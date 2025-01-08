import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings= vocab_size, embedding_dim= d_model)
    
    def forward(self, x):
        return self.embedding(x)

class PositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()

        position = torch.arange(0, max_len).unsqueeze(1)

        i = torch.arange(0, d_model, 2) / d_model
        exp_term = 10000 ** i

        self.encoding = torch.zeros(max_len, d_model, requires_grad= False)

        self.encoding[:, 0::2] = torch.sin(position / exp_term)
        self.encoding[:, 1::2] = torch.cos(position / exp_term)

    def forward(self,x):
        batch_size, seq_len = x.size()

        return self.encoding[:seq_len, :]

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, drop_prob):
        super().__init__()
        self.tokenEmbedding = TokenEmbedding(vocab_size, d_model)
        self.positionalEncoding = PositionalEncoding(max_len, d_model)
        self.dropOut = nn.Dropout(drop_prob)

    def forward(self, x):
        # x's dim : [seq_len, d_model]
        te = self.tokenEmbedding(x)
        pe = self.positionalEncoding(x)
        return self.dropOut(te+pe)