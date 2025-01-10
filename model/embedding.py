import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenEmbedding(nn.Module):
    '''
    input: [batch_size, seq_len]
    output: [batch_size, seq_len, embedding_dim]
    '''
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings= vocab_size, embedding_dim= embedding_dim)
    
    def forward(self, x):
        return self.embedding(x)

class PositionalEncoding(nn.Module):
    '''
    input: [batch_size, seq_len]
    output: [1, seq_len, embedding_dim]
    '''
    def __init__(self, max_len: int, embedding_dim: int, device):
        super().__init__()

        position = torch.arange(0, max_len, device=device).unsqueeze(1) # [max_len, 1]

        i = torch.arange(0, embedding_dim, 2, device=device).unsqueeze(0) / embedding_dim # [1, d_model]
        exp_term = 10000 ** i

        self.encoding = torch.zeros(max_len, embedding_dim, requires_grad= False, device=device)

        self.encoding[:, 0::2] = torch.sin(position / exp_term)
        self.encoding[:, 1::2] = torch.cos(position / exp_term)

    def forward(self,x):
        batch_size, seq_len = x.size()

        return self.encoding[:seq_len, :].unsqueeze(0)

class TransformerEmbedding(nn.Module):
    '''
    input: [batch_size, seq_len]
    output: [batch_size, seq_len, embedding_dim]
    '''
    def __init__(self, vocab_size, max_len, embedding_dim, drop_prob, device):
        super().__init__()
        self.tokenEmbedding = TokenEmbedding(vocab_size, embedding_dim)
        self.positionalEncoding = PositionalEncoding(max_len, embedding_dim, device=device)
        self.dropOut = nn.Dropout(drop_prob)

    def forward(self, x):
        # x's dim : [batch_size, seq_len]
        te = self.tokenEmbedding(x) # [batch_size, seq_len, embedding_dim]
        pe = self.positionalEncoding(x) # [1, seq_len, embedding_dim]
        return self.dropOut(te+pe) # broadcasting to [batch_size, seq_len, embedding_dim]