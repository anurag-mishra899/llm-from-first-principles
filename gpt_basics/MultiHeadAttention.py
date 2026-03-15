import torch
from torch import nn
from torch.nn import functional as F
from AttentionHead import AttentionHead

class MultiHeadAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        n_head = config.n_head
        d_head = config.n_embd // n_head
        self.heads = nn.ModuleList([AttentionHead(d_head,config) for _ in range(n_head)])
        self.proj = nn.Linear(config.n_embd,config.n_embd,bias=False)

    def forward(self,x):
        out = torch.concat([head(x) for head in self.heads],dim=-1)
        out = self.proj(out)
        return out 