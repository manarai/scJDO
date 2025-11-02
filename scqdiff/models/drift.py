
from dataclasses import dataclass
import torch, torch.nn as nn
from ..nn.score_net import MLPScore
@dataclass
class DriftConfig:
    dim:int; beta:float=0.1; hidden:int=256; depth:int=3; laplacian_lambda:float=0.0; device:str="cpu"
class ResidualNet(nn.Module):
    def __init__(self, dim, hidden=256, depth=2):
        super().__init__(); layers=[]; in_dim=dim+1
        for _ in range(depth): layers += [nn.Linear(in_dim, hidden), nn.SiLU()]; in_dim=hidden
        layers += [nn.Linear(in_dim, dim)]; self.net=nn.Sequential(*layers)
    def forward(self, x, t): tcol=t.view(-1,1).expand(-1,x.shape[1]); return self.net(torch.cat([x,tcol], dim=-1))
class DriftField(nn.Module):
    def __init__(self, cfg: DriftConfig, laplacian=None):
        super().__init__(); self.cfg=cfg
        self.score=MLPScore(cfg.dim, hidden=cfg.hidden, depth=cfg.depth)
        self.residual=ResidualNet(cfg.dim, hidden=cfg.hidden//2, depth=max(1,cfg.depth-1))
        self.register_buffer("L", laplacian if laplacian is not None else torch.zeros(cfg.dim, cfg.dim))
    def forward(self, x, t):
        u=self.cfg.beta*self.score(x,t)+self.residual(x,t)
        if self.cfg.laplacian_lambda>0 and self.L.numel()>0: u = u - self.cfg.laplacian_lambda*(x @ self.L.T)
        return u
    def jacobian(self, x, t):
        J=[]; 
        for i in range(x.shape[0]):
            xi=x[i:i+1].requires_grad_(True); ti=t[i:i+1]; ui=self.forward(xi,ti); Ji=[]
            for k in range(ui.shape[1]): grad=torch.autograd.grad(ui[0,k], xi, retain_graph=True, create_graph=True)[0]; Ji.append(grad)
            Ji=torch.stack(Ji, dim=1)[0]; J.append(Ji)
        return torch.stack(J, dim=0)
