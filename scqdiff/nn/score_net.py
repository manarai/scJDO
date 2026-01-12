
import torch, torch.nn as nn
from .time_embed import FiLMTime
class MLPScore(nn.Module):
    def __init__(self, dim, hidden=256, depth=3):
        super().__init__()
        layers=[]; in_dim=dim
        for _ in range(depth): layers += [nn.Linear(in_dim, hidden), nn.SiLU()]; in_dim=hidden
        self.backbone=nn.Sequential(*layers); self.head=nn.Linear(hidden, dim); self.film=FiLMTime(hidden)
    def forward(self, x, t):
        h=self.backbone[0](x); h=self.backbone[1](h)
        for i in range(2, len(self.backbone), 2):
            h=self.film(h,t); h=self.backbone[i](h); h=self.backbone[i+1](h)
        return self.head(h)
