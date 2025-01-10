import torch
import torch.nn as nn
import torch.nn.functional as F

from model.identicalLayers import EncoderLayer, DecoderLayer
from model.embedding import TransformerEmbedding

class Encoder(nn.Module):
    def __init__(self, n_layer, vocab_size, max_len, d_model, ffn_hidden, n_head, device, dropout=0.1):
        super().__init__()

        self.embedding = TransformerEmbedding(vocab_size=vocab_size, max_len=max_len, embedding_dim=d_model, drop_prob=dropout, device=device)
        self.encoder = nn.ModuleList([EncoderLayer(d_model=d_model, hidden=ffn_hidden, n_head=n_head, dropout=dropout) for _ in range(n_layer)])
    
    def forward(self, x):
        x = self.embedding(x)

        for encoderLayer in self.encoder:
            x = encoderLayer(x)

        return x
    
class Decoder(nn.Module):
    def __init__(self, n_layer, vocab_size, max_len, d_model, ffn_hidden, n_head, device, dropout=0.1):
        super().__init__()

        self.embedding = TransformerEmbedding(vocab_size=vocab_size, max_len=max_len, embedding_dim=d_model, drop_prob=dropout, device=device)
        self.decoder = nn.ModuleList([DecoderLayer(d_model=d_model, hidden=ffn_hidden, n_head=n_head, dropout=dropout) for _ in range(n_layer)])

        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, encoder, x, mask):
        x = self.embedding(x)

        for decoderLayer in self.decoder:
            x = decoderLayer(encoder, x, mask)

        return self.linear(x)
    
class Transformer(nn.Module):
    def __init__(self, n_layer, enc_vocab_size, dec_vocab_size, max_len, d_model, ffn_hidden, n_head, device, dropout=0.1):
        super().__init__()

        self.encoder = Encoder(n_layer, enc_vocab_size, max_len, d_model, ffn_hidden, n_head, device, dropout)
        self.decoder = Decoder(n_layer, dec_vocab_size, max_len, d_model, ffn_hidden, n_head, device, dropout)

    def forward(self, src, tgt, mask):
        enc_output = self.encoder(src)
        dec_output = self.decoder(enc_output, tgt, mask)

        return dec_output