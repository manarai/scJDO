"""
Hybrid Drift Field with Velocity Prior Support
"""
from dataclasses import dataclass
import torch
import torch.nn as nn
from ..nn.score_net import MLPScore


@dataclass
class DriftConfig:
    """Configuration for Hybrid Drift Field"""
    dim: int
    beta: float = 0.1
    hidden: int = 256
    depth: int = 3
    laplacian_lambda: float = 0.0
    device: str = "cpu"
    
    # Velocity prior parameters
    use_velocity_prior: bool = False
    vel_k: int = 16
    vel_scale: float = 1.0
    vel_tau: float = 0.1
    vel_conf_power: float = 2.0
    vel_schedule: str = 'mid'  # 'constant', 'early', 'mid', 'late'


class ResidualNet(nn.Module):
    """Residual correction network (Neural ODE component)"""
    def __init__(self, dim, hidden=256, depth=2):
        super().__init__()
        layers = []
        in_dim = dim  # Only x, not concatenating t here
        
        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden), nn.SiLU()]
            in_dim = hidden
        
        layers += [nn.Linear(in_dim, dim)]
        self.net = nn.Sequential(*layers)
    
    def forward(self, x, t):
        """
        Forward pass with time-dependent modulation
        Args:
            x: (batch, dim) cell states
            t: (batch,) time points
        """
        # Time-dependent gating
        t_gate = torch.sigmoid(t.view(-1, 1))
        return self.net(x) * t_gate


class KNNVelocity(nn.Module):
    """KNN-based velocity field for biological priors"""
    def __init__(self, X_ref, V_ref, W_ref=None, k=16, tau=0.1):
        super().__init__()
        self.register_buffer("X_ref", X_ref)
        self.register_buffer("V_ref", V_ref)
        
        if W_ref is not None:
            self.register_buffer("W_ref", W_ref)
        else:
            self.register_buffer("W_ref", torch.ones(X_ref.shape[0]))
        
        self.k = k
        self.tau = tau
    
    def forward(self, x):
        """
        Compute velocity at query points using KNN interpolation
        Args:
            x: (batch, dim) query points
        Returns:
            v: (batch, dim) velocity vectors
            conf: (batch,) confidence scores
        """
        # Compute distances to reference points
        dists = torch.cdist(x, self.X_ref)  # (batch, n_ref)
        
        # Get k nearest neighbors
        topk_dists, topk_idx = torch.topk(dists, k=self.k, largest=False, dim=1)
        
        # Gaussian kernel weights
        weights = torch.exp(-topk_dists / self.tau)  # (batch, k)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # Weighted average of neighbor velocities
        neighbor_vels = self.V_ref[topk_idx]  # (batch, k, dim)
        v = (weights.unsqueeze(-1) * neighbor_vels).sum(dim=1)  # (batch, dim)
        
        # Confidence from neighbor confidence scores
        neighbor_conf = self.W_ref[topk_idx]  # (batch, k)
        conf = (weights * neighbor_conf).sum(dim=1)  # (batch,)
        
        return v, conf


class DriftField(nn.Module):
    """
    Hybrid Drift Field combining:
    1. Score network (diffusion component)
    2. Residual network (Neural ODE component)
    3. Velocity prior (biological component)
    """
    def __init__(self, cfg: DriftConfig, X_ref=None, V_ref=None, W_ref=None, laplacian=None):
        super().__init__()
        self.cfg = cfg
        
        # Score network (diffusion component)
        self.score = MLPScore(cfg.dim, hidden=cfg.hidden, depth=cfg.depth)
        
        # Residual network (Neural ODE component)
        self.residual = ResidualNet(cfg.dim, hidden=cfg.hidden//2, depth=max(1, cfg.depth-1))
        
        # Laplacian regularization
        if laplacian is not None:
            self.register_buffer("L", laplacian)
        else:
            self.register_buffer("L", torch.zeros(cfg.dim, cfg.dim))
        
        # Velocity prior (biological component)
        if cfg.use_velocity_prior and X_ref is not None and V_ref is not None:
            self.vel = KNNVelocity(X_ref, V_ref, W_ref, k=cfg.vel_k, tau=cfg.vel_tau)
        else:
            self.vel = None
    
    def _g(self, t):
        """
        Time schedule for velocity prior
        Args:
            t: (batch,) time points in [0, 1]
        Returns:
            g: (batch,) schedule values
        """
        if self.cfg.vel_schedule == 'constant':
            return torch.ones_like(t)
        elif self.cfg.vel_schedule == 'early':
            return torch.exp(-5 * t)
        elif self.cfg.vel_schedule == 'mid':
            return torch.exp(-5 * (t - 0.5)**2)
        elif self.cfg.vel_schedule == 'late':
            return torch.exp(-5 * (1 - t))
        else:
            return torch.ones_like(t)
    
    def forward(self, x, t):
        """
        Compute drift at (x, t)
        Args:
            x: (batch, dim) cell states
            t: (batch,) time points
        Returns:
            u: (batch, dim) drift vectors
        """
        # Base drift: score + residual
        u = self.cfg.beta * self.score(x, t) + self.residual(x, t)
        
        # Laplacian regularization
        if self.cfg.laplacian_lambda > 0 and self.L.numel() > 0:
            u = u - self.cfg.laplacian_lambda * (x @ self.L.T)
        
        # Add velocity prior if available
        if self.vel is not None:
            v, conf = self.vel(x)
            
            # Time-dependent gating
            g = self._g(t).view(-1, 1)
            
            # Confidence-weighted velocity
            gate = conf.pow(self.cfg.vel_conf_power).view(-1, 1)
            
            # Add velocity component
            b = self.cfg.vel_scale * g * gate * v
            u = u + b
        
        return u
    
    def jacobian(self, x, t):
        """
        Compute Jacobian ∂f/∂x at (x, t)
        Args:
            x: (batch, dim) cell states
            t: (batch,) time points
        Returns:
            J: (batch, dim, dim) Jacobian matrices
        """
        J = []
        
        for i in range(x.shape[0]):
            xi = x[i:i+1].requires_grad_(True)
            ti = t[i:i+1]
            ui = self.forward(xi, ti)
            
            Ji = []
            for k in range(ui.shape[1]):
                grad = torch.autograd.grad(
                    ui[0, k], xi,
                    retain_graph=True,
                    create_graph=True
                )[0]
                Ji.append(grad)
            
            Ji = torch.stack(Ji, dim=1)[0]
            J.append(Ji)
        
        return torch.stack(J, dim=0)
