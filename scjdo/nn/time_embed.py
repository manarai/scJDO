
import torch, torch.nn as nn
class FiLMTime(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 2*hidden_dim))
    def forward(self, h, t):
        t = t.view(-1,1); gamma, beta = self.proj(t).chunk(2, dim=-1)
        return h*(1+gamma)+beta
