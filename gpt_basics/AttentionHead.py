import torch
from torch import nn
from torch.nn import functional as F


class AttentionHead(nn.Module):
    def __init__(self,d_head,config):
        super().__init__()
        self.Q = nn.Linear(config.n_embd,d_head,bias=False)
        self.K = nn.Linear(config.n_embd,d_head,bias=False)
        self.V = nn.Linear(config.n_embd,d_head,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(config.seq_len,config.seq_len)))

    def forward(self,x):
        B,T,C = x.shape 
        q, k = self.Q(x), self.V(x)
        wei = q @ k.transpose(-2,-1) # (B,T,C) @ (B,C,T) = (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))
        wei = F.softmax(wei,dim=-1) #(B,T,T)
        v = self.V(x)
        out = wei @ v # (B,T,T) @ (B,T,C) = (B,T,C)
        return out 

