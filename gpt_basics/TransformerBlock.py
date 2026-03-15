import torch
from torch import nn
from torch.nn import functional as F
from MultiHeadAttention import MultiHeadAttention
from FeedForward import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(config.n_embd)
        self.layernorm2 = nn.LayerNorm(config.n_embd)
        self.multihead = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)

    def forward(self,x):
        x = x + self.multihead(self.layernorm1(x))
        x = x + self.ffwd(self.layernorm2(x))
        return x