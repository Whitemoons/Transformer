import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionWiseScaleProductAttention(nn.Module):
    '''
    input:
        Q: [batch_size, attention_head, seq_len, d_k]
        K: [batch_size, attention_head, seq_len, d_k]
        V: [batch_size, attention_head, seq_len, d_k]
    output:
        attention value: [batch_size, attention_head, seq_len, d_k]
        attention distribution: [batch_size, attention_head, seq_len, d_k]
    '''
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None):
        batch_size, attention_head, seq_len, d_k = q.size()
        
        k_T = k.transpose(2,3)
        attention_scores = torch.matmul(q, k_T) / math.sqrt(d_k) # [batch_size, attention_head]

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -100000)

        attention_distribution = F.softmax(attention_scores)
        attention_distribution = self.dropout(attention_distribution)

        attention_values = torch.matmul(attention_distribution, v)

        return attention_values, attention_distribution

class MultiHeadSelfAttention(nn.Module):
    '''
    input:
        Q,K,V: [batch_size, seq_len, embedding_dim]

    output:
        output: [batch_size, seq_len, embedding_dim]
        attention distribution: [batch_size, attention_head, seq_len, d_k] # d_k * attention_head = embedding_dim

    '''
    def __init__(self, embedding_dim, head_num, dropout=0.1):
        super().__init__()
        self.head_num = head_num
        self.Q = nn.Linear(embedding_dim, embedding_dim)
        self.K = nn.Linear(embedding_dim, embedding_dim)
        self.V = nn.Linear(embedding_dim, embedding_dim)

        self.attention = PositionWiseScaleProductAttention()

        self.linear = nn.Linear(embedding_dim, embedding_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask = None):
        batch_size, seq_len, d_model = q.size()

        # 1. dot product each weight matrices
        q = self.Q(q)
        k = self.K(k)
        v = self.V(v)

        # 2. split weight matrices
        q = q.view([batch_size, seq_len, self.head_num, -1]).transpose(1,2) # [batch_size, head_num, seq_len, d_k]
        k = k.view([batch_size, seq_len, self.head_num, -1]).transpose(1,2)
        v = v.view([batch_size, seq_len, self.head_num, -1]).transpose(1,2)

        # 3. calculate attention value
        attention_values, attention_distribution = self.attention(q, k, v, mask)

        # 4. concat
        attention_values = attention_values.transpose(1,2).reshape((batch_size, seq_len, -1))

        # 5. linear
        output = self.linear(attention_values)
        output = self.dropout(output)

        return output, attention_distribution

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, dropout = 0.1):
        super().__init__()

        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 1. FFN
        output = self.linear1(x)
        output = self.relu(output)
        output = self.linear2(output)

        output = self.dropout(output)

        return output