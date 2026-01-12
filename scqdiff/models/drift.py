
from dataclasses import dataclass
import torch
import torch.nn as nn
from ..nn.score_net import MLPScore


@dataclass
class DriftConfig:
    """Configuration for drift field model.
    
    Attributes:
        dim: Dimensionality of the state space
        beta: Diffusion coefficient
        hidden: Hidden layer size for neural networks
        depth: Number of layers in neural networks
        laplacian_lambda: Strength of Laplacian smoothing regularization
        device: Device to run computations on
        use_velocity_prior: Whether to incorporate RNA velocity as biological prior
        vel_k: Number of nearest neighbors for velocity interpolation
        vel_tau: Temperature parameter for softmax weighting in KNN
        vel_scale: Global scaling factor for velocity magnitude
        vel_conf_power: Exponent for confidence gating (higher = stronger gating)
        vel_time_mode: Time schedule for velocity contribution ("mid" or "flat")
    """
    dim: int
    beta: float = 0.1
    hidden: int = 256
    depth: int = 3
    laplacian_lambda: float = 0.0
    device: str = "cpu"
    # RNA velocity prior parameters
    use_velocity_prior: bool = False
    vel_k: int = 32
    vel_tau: float = 1.0
    vel_scale: float = 1.0
    vel_conf_power: float = 1.0
    vel_time_mode: str = "mid"  # "mid" or "flat"


class KNNVelocity(nn.Module):
    """Interpolates velocity v(x) from reference points using soft k-nearest neighbors.
    
    This module enables the drift field to evaluate velocity at any point in state space,
    not just at the discrete reference cell locations. It uses distance-based softmax
    weighting to create a smooth velocity field.
    
    Args:
        X_ref: Reference cell states (N, D)
        V_ref: Reference velocity vectors (N, D)
        k: Number of nearest neighbors to use
        tau: Temperature for softmax weighting (lower = sharper, higher = smoother)
        W_ref: Optional per-cell confidence weights (N,), scaled to [0,1]
    """
    
    def __init__(
        self,
        X_ref: torch.Tensor,
        V_ref: torch.Tensor,
        k: int = 32,
        tau: float = 1.0,
        W_ref: torch.Tensor | None = None
    ):
        super().__init__()
        # Store as buffers so they move with .to(device) and save in state_dict
        self.register_buffer("X_ref", X_ref)
        self.register_buffer("V_ref", V_ref)
        self.register_buffer(
            "W_ref",
            W_ref if W_ref is not None else torch.ones((X_ref.shape[0],), device=X_ref.device)
        )
        self.k = int(k)
        self.tau = float(tau)
    
    def forward(self, x: torch.Tensor):
        """Interpolate velocity and confidence at query points.
        
        Args:
            x: Query points (B, D)
            
        Returns:
            v: Interpolated velocity vectors (B, D)
            conf: Interpolated confidence values (B,), in range [0, 1]
        """
        # Compute pairwise distances: (B, N)
        dist = torch.cdist(x, self.X_ref)
        
        # Get k nearest neighbors
        k = min(self.k, dist.shape[1])
        d, idx = torch.topk(dist, k=k, largest=False)  # (B, k)
        
        # Softmax weighting based on distance
        w = torch.softmax(-d / max(self.tau, 1e-6), dim=1)  # (B, k)
        
        # Interpolate velocity vectors
        Vnn = self.V_ref[idx]  # (B, k, D)
        v = (w.unsqueeze(-1) * Vnn).sum(dim=1)  # (B, D)
        
        # Interpolate confidence weights
        Wnn = self.W_ref[idx]  # (B, k)
        conf = (w * Wnn).sum(dim=1).clamp(0.0, 1.0)  # (B,)
        
        return v, conf


class ResidualNet(nn.Module):
    """Residual correction network for drift field.
    
    This network learns a correction term to the base drift (score + velocity prior).
    It takes both state x and time t as input.
    """
    
    def __init__(self, dim, hidden=256, depth=2):
        super().__init__()
        layers = []
        in_dim = dim + 1  # x + t concatenated
        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden), nn.SiLU()]
            in_dim = hidden
        layers += [nn.Linear(in_dim, dim)]
        self.net = nn.Sequential(*layers)
    
    def forward(self, x, t):
        """Compute residual correction.
        
        Args:
            x: State (B, D)
            t: Time (B,)
            
        Returns:
            Residual drift correction (B, D)
        """
        tcol = t.view(-1, 1)  # (B, 1)
        xt = torch.cat([x, tcol], dim=-1)  # (B, D+1)
        return self.net(xt)


class DriftField(nn.Module):
    """Drift field model with optional RNA velocity prior.
    
    The drift is computed as:
        f(x,t) = beta * score(x,t) + residual(x,t) + b(x,t)
    
    where b(x,t) is the velocity prior (if enabled):
        b(x,t) = vel_scale * g(t) * gate(x) * v(x)
    
    - v(x): Interpolated velocity from reference data
    - gate(x): Confidence-based gating to downweight unreliable velocities
    - g(t): Time schedule to control when velocity guidance is strongest
    - vel_scale: Global scaling factor for velocity magnitude
    
    Args:
        cfg: Configuration object
        laplacian: Optional graph Laplacian for smoothing (D, D)
        X_ref: Reference cell states for velocity interpolation (N, D)
        V_ref: Reference velocity vectors (N, D)
        W_ref: Optional per-cell confidence weights (N,)
    """
    
    def __init__(
        self,
        cfg: DriftConfig,
        laplacian=None,
        X_ref=None,
        V_ref=None,
        W_ref=None
    ):
        super().__init__()
        self.cfg = cfg
        
        # Core drift components
        self.score = MLPScore(cfg.dim, hidden=cfg.hidden, depth=cfg.depth)
        self.residual = ResidualNet(
            cfg.dim,
            hidden=cfg.hidden // 2,
            depth=max(1, cfg.depth - 1)
        )
        
        # Graph Laplacian for smoothing
        self.register_buffer(
            "L",
            laplacian if laplacian is not None else torch.zeros(cfg.dim, cfg.dim)
        )
        
        # RNA velocity prior
        self.vel = None
        if cfg.use_velocity_prior:
            assert X_ref is not None and V_ref is not None, \
                "Need X_ref and V_ref for velocity prior"
            self.vel = KNNVelocity(X_ref, V_ref, k=cfg.vel_k, tau=cfg.vel_tau, W_ref=W_ref)
    
    def _g(self, t: torch.Tensor):
        """Time schedule for velocity contribution.
        
        Args:
            t: Time values (B,), in range [0, 1]
            
        Returns:
            Schedule values (B,)
        """
        if self.cfg.vel_time_mode == "mid":
            # Peak at t=0.5, zero at endpoints
            return 4.0 * t * (1.0 - t)
        # Flat schedule (constant contribution)
        return torch.ones_like(t)
    
    def forward(self, x, t):
        """Compute drift field at given states and times.
        
        Args:
            x: States (B, D)
            t: Times (B,)
            
        Returns:
            Drift vectors (B, D)
        """
        # Learned correction: score + residual
        u = self.cfg.beta * self.score(x, t) + self.residual(x, t)
        
        # Apply Laplacian smoothing to learned term only
        # (don't distort the biological prior)
        if self.cfg.laplacian_lambda > 0 and self.L.numel() > 0:
            u = u - self.cfg.laplacian_lambda * (x @ self.L.T)
        
        # Add velocity prior drift b(x,t)
        if self.vel is not None:
            v, conf = self.vel(x)  # Interpolated velocity and confidence
            
            # Confidence gating: downweight unreliable velocities
            gate = conf.pow(self.cfg.vel_conf_power).view(-1, 1)
            
            # Time schedule: control when velocity is strongest
            g = self._g(t).view(-1, 1)
            
            # Velocity prior: scaled, gated, scheduled
            b = self.cfg.vel_scale * g * gate * v
            
            return b + u
        
        return u
    
    def jacobian(self, x, t):
        """Compute Jacobian matrix ∂f/∂x at given states and times.
        
        This captures the local gene-gene influence structure.
        
        Args:
            x: States (B, D)
            t: Times (B,)
            
        Returns:
            Jacobian matrices (B, D, D)
        """
        J = []
        for i in range(x.shape[0]):
            xi = x[i:i+1].requires_grad_(True)
            ti = t[i:i+1]
            ui = self.forward(xi, ti)
            Ji = []
            for k in range(ui.shape[1]):
                grad = torch.autograd.grad(
                    ui[0, k],
                    xi,
                    retain_graph=True,
                    create_graph=True
                )[0]
                Ji.append(grad)
            Ji = torch.stack(Ji, dim=1)[0]
            J.append(Ji)
        return torch.stack(J, dim=0)
