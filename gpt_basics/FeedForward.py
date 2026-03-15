import torch
from torch import nn
from torch.nn import functional as F

class FeedForward(nn.Module):
    def __init__(self,config):
        super().__init__()
        n_embd = config.n_embd 
        self.ffwd = nn.Sequential(
        nn.Linear(n_embd,4*n_embd),
        nn.ReLU(),
        nn.Linear(4*n_embd,n_embd))        

    def forward(self,x):
        out = self.ffwd(x)
        return out 