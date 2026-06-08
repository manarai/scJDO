
from __future__ import annotations
from typing import Optional
import torch
@torch.no_grad()
def euler_integrate(model, x0: torch.Tensor, t0: float = 0.0, t1: float = 1.0, steps: int = 100,
                    stochastic: bool = False, beta: Optional[float] = None) -> torch.Tensor:
    B,D=x0.shape; x=x0.clone(); ts=torch.linspace(t0,t1,steps,device=x0.device); dt=(t1-t0)/max(1,steps)
    if stochastic:
        if beta is None: beta=getattr(getattr(model,'cfg',None),'beta',0.1); sigma=(2.0*beta*dt)**0.5
    traj=[]
    for ti in ts:
        tvec=torch.full((B,), float(ti), device=x.device); u=model(x,tvec)
        if stochastic: x = x + u*dt + torch.randn_like(x)*sigma
        else: x = x + u*dt
        traj.append(x.clone())
    return torch.stack(traj, dim=1)
def apply_archetype_modulation(model, x: torch.Tensor, t: torch.Tensor, patterns: torch.Tensor, U_t: torch.Tensor,
                               lam: float = 0.5, which: int = 0) -> torch.Tensor:
    u=model(x,t); T_bins,R=U_t.shape; tb=torch.clamp((t*(T_bins-1)).round().long(),0,T_bins-1)
    A=patterns[which]; Ax=x @ A.T; cr=U_t[tb, which].view(-1,1); return u + lam*Ax*cr
@torch.no_grad()
def euler_integrate_with_archetype(model, x0: torch.Tensor, patterns: torch.Tensor, U_t: torch.Tensor, which: int = 0, lam: float = 0.5,
                                   t0: float = 0.0, t1: float = 1.0, steps: int = 100) -> torch.Tensor:
    B,D=x0.shape; x=x0.clone(); ts=torch.linspace(t0,t1,steps,device=x0.device); dt=(t1-t0)/max(1,steps); traj=[]
    for ti in ts:
        tvec=torch.full((B,), float(ti), device=x.device); u=apply_archetype_modulation(model,x,tvec,patterns,U_t,lam=lam,which=which); x=x+u*dt; traj.append(x.clone())
    return torch.stack(traj, dim=1)
