"""
Schrödinger Bridge implementation for scIDiff.

This module implements the Schrödinger Bridge problem for optimal transport
between two distributions (e.g., young and old cells).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np

from scqdiff.nn.score_net import MLPScore
from scqdiff.transport.sinkhorn import compute_ot_plan, sinkhorn_divergence


class SchrodingerBridgeConfig:
    """Configuration for Schrödinger Bridge."""
    
    def __init__(
        self,
        dim: int,
        hidden: int = 256,
        depth: int = 4,
        beta: float = 0.1,
        sigma: float = 0.2,
        epsilon: float = 0.1,
        sinkhorn_max_iter: int = 100,
        device: str = 'cpu'
    ):
        self.dim = dim
        self.hidden = hidden
        self.depth = depth
        self.beta = beta  # Diffusion coefficient
        self.sigma = sigma  # Noise level for score matching
        self.epsilon = epsilon  # Entropic regularization for OT
        self.sinkhorn_max_iter = sinkhorn_max_iter
        self.device = device


class SchrodingerBridge(nn.Module):
    """
    Schrödinger Bridge for optimal transport between distributions.
    
    Given two distributions ρ₀ (source) and ρ₁ (target), learns forward and
    backward drift fields that optimally transport between them.
    
    Example use case: Aging
        ρ₀ = young cells
        ρ₁ = old cells
        forward = aging process
        backward = rejuvenation process
    """
    
    def __init__(
        self,
        cfg: SchrodingerBridgeConfig,
        X_0: torch.Tensor,
        X_1: torch.Tensor
    ):
        """
        Args:
            cfg: Configuration
            X_0: Source distribution samples (N0, D) - e.g., young cells
            X_1: Target distribution samples (N1, D) - e.g., old cells
        """
        super().__init__()
        self.cfg = cfg
        
        # Store endpoint distributions
        self.register_buffer("X_0", X_0.to(cfg.device))
        self.register_buffer("X_1", X_1.to(cfg.device))
        
        # Forward drift network (ρ₀ → ρ₁)
        self.forward_net = MLPScore(
            cfg.dim,
            hidden=cfg.hidden,
            depth=cfg.depth
        ).to(cfg.device)
        
        # Backward drift network (ρ₁ → ρ₀)
        self.backward_net = MLPScore(
            cfg.dim,
            hidden=cfg.hidden,
            depth=cfg.depth
        ).to(cfg.device)
        
        # Optimal transport plan (computed during training)
        self.P = None  # (N0, N1)
        self.f = None  # Dual potential for X_0
        self.g = None  # Dual potential for X_1
    
    def forward_drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute forward drift (source → target).
        
        Args:
            x: States (B, D)
            t: Times (B,) in [0, 1]
        
        Returns:
            drift: Forward drift (B, D)
        """
        return self.cfg.beta * self.forward_net(x, t)
    
    def backward_drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute backward drift (target → source).
        
        Args:
            x: States (B, D)
            t: Times (B,) in [0, 1]
        
        Returns:
            drift: Backward drift (B, D)
        """
        return self.cfg.beta * self.backward_net(x, t)
    
    def compute_ot_plan(self):
        """
        Compute optimal transport plan between X_0 and X_1.
        
        Updates self.P, self.f, self.g.
        """
        with torch.no_grad():
            self.P, self.f, self.g = compute_ot_plan(
                self.X_0,
                self.X_1,
                epsilon=self.cfg.epsilon,
                max_iter=self.cfg.sinkhorn_max_iter
            )
    
    def sample_bridge_trajectory(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a batch of bridge trajectories for training.
        
        Returns:
            x_0: Initial states (batch_size, D)
            x_1: Final states (batch_size, D)
            t: Random times (batch_size,)
        """
        # Sample from X_0
        idx_0 = torch.randint(0, self.X_0.shape[0], (batch_size,))
        x_0 = self.X_0[idx_0]
        
        # Sample corresponding points from X_1 using OT plan
        if self.P is not None:
            # Sample from coupling
            P_marginal = self.P[idx_0]  # (batch_size, N1)
            idx_1 = torch.multinomial(P_marginal, 1).squeeze(1)
            x_1 = self.X_1[idx_1]
        else:
            # Fallback: random pairing
            idx_1 = torch.randint(0, self.X_1.shape[0], (batch_size,))
            x_1 = self.X_1[idx_1]
        
        # Sample random time
        t = torch.rand(batch_size, device=self.cfg.device)
        
        return x_0, x_1, t
    
    def forward_score_matching_loss(self, batch_size: int) -> torch.Tensor:
        """
        Compute score matching loss for forward drift.
        
        Args:
            batch_size: Number of samples
        
        Returns:
            loss: Scalar loss
        """
        # Sample bridge trajectory
        x_0, x_1, t = self.sample_bridge_trajectory(batch_size)
        
        # Linear interpolation between x_0 and x_1
        t_expand = t.view(-1, 1)
        x_t = (1 - t_expand) * x_0 + t_expand * x_1
        
        # Add noise
        noise = torch.randn_like(x_t) * self.cfg.sigma
        x_noisy = x_t + noise
        
        # Compute score
        score = self.forward_net(x_noisy, t)
        
        # Target: denoising direction
        target = -noise / (self.cfg.sigma ** 2)
        
        # MSE loss
        loss = F.mse_loss(score, target)
        
        return loss
    
    def backward_score_matching_loss(self, batch_size: int) -> torch.Tensor:
        """
        Compute score matching loss for backward drift.
        
        Args:
            batch_size: Number of samples
        
        Returns:
            loss: Scalar loss
        """
        # Sample bridge trajectory (but go backward)
        x_0, x_1, t = self.sample_bridge_trajectory(batch_size)
        
        # Linear interpolation (backward: from x_1 to x_0)
        t_expand = t.view(-1, 1)
        x_t = t_expand * x_0 + (1 - t_expand) * x_1
        
        # Add noise
        noise = torch.randn_like(x_t) * self.cfg.sigma
        x_noisy = x_t + noise
        
        # Compute score
        score = self.backward_net(x_noisy, t)
        
        # Target: denoising direction
        target = -noise / (self.cfg.sigma ** 2)
        
        # MSE loss
        loss = F.mse_loss(score, target)
        
        return loss
    
    def endpoint_loss(self) -> torch.Tensor:
        """
        Compute loss to ensure drift reaches endpoints.
        
        Returns:
            loss: Scalar endpoint loss
        """
        # Forward: X_0 should reach X_1 at t=1
        t_end = torch.ones(self.X_0.shape[0], device=self.cfg.device)
        drift_forward_end = self.forward_drift(self.X_0, t_end)
        
        # Expected target from OT plan
        if self.P is not None:
            X_1_expected = self.P @ self.X_1
            loss_forward = F.mse_loss(self.X_0 + drift_forward_end, X_1_expected)
        else:
            loss_forward = torch.tensor(0.0, device=self.cfg.device)
        
        # Backward: X_1 should reach X_0 at t=0
        t_start = torch.zeros(self.X_1.shape[0], device=self.cfg.device)
        drift_backward_start = self.backward_drift(self.X_1, t_start)
        
        # Expected source from OT plan
        if self.P is not None:
            X_0_expected = self.P.T @ self.X_0
            loss_backward = F.mse_loss(self.X_1 + drift_backward_start, X_0_expected)
        else:
            loss_backward = torch.tensor(0.0, device=self.cfg.device)
        
        return loss_forward + loss_backward
    
    @torch.no_grad()
    def forward_integrate(
        self,
        x0: torch.Tensor,
        steps: int = 100,
        stochastic: bool = False
    ) -> torch.Tensor:
        """
        Integrate forward drift from t=0 to t=1 (e.g., aging).
        
        Args:
            x0: Initial states (B, D)
            steps: Number of integration steps
            stochastic: Whether to add stochastic noise
        
        Returns:
            trajectory: (B, steps+1, D)
        """
        B, D = x0.shape
        x = x0.clone()
        dt = 1.0 / steps
        
        if stochastic:
            sigma = (2.0 * self.cfg.beta * dt) ** 0.5
        
        trajectory = [x.clone()]
        
        for i in range(steps):
            t = torch.full((B,), i * dt, device=self.cfg.device)
            drift = self.forward_drift(x, t)
            
            if stochastic:
                x = x + drift * dt + torch.randn_like(x) * sigma
            else:
                x = x + drift * dt
            
            trajectory.append(x.clone())
        
        return torch.stack(trajectory, dim=1)
    
    @torch.no_grad()
    def backward_integrate(
        self,
        x1: torch.Tensor,
        steps: int = 100,
        stochastic: bool = False
    ) -> torch.Tensor:
        """
        Integrate backward drift from t=1 to t=0 (e.g., rejuvenation).
        
        Args:
            x1: Initial states at t=1 (B, D)
            steps: Number of integration steps
            stochastic: Whether to add stochastic noise
        
        Returns:
            trajectory: (B, steps+1, D)
        """
        B, D = x1.shape
        x = x1.clone()
        dt = 1.0 / steps
        
        if stochastic:
            sigma = (2.0 * self.cfg.beta * dt) ** 0.5
        
        trajectory = [x.clone()]
        
        for i in range(steps):
            t = torch.full((B,), 1.0 - i * dt, device=self.cfg.device)
            drift = self.backward_drift(x, t)
            
            if stochastic:
                x = x - drift * dt + torch.randn_like(x) * sigma
            else:
                x = x - drift * dt
            
            trajectory.append(x.clone())
        
        return torch.stack(trajectory, dim=1)
    
    def train_step(
        self,
        batch_size: int,
        update_ot: bool = False
    ) -> dict:
        """
        Single training step.
        
        Args:
            batch_size: Batch size for score matching
            update_ot: Whether to recompute OT plan
        
        Returns:
            losses: Dictionary of losses
        """
        if update_ot or self.P is None:
            self.compute_ot_plan()
        
        # Forward score matching
        loss_forward = self.forward_score_matching_loss(batch_size)
        
        # Backward score matching
        loss_backward = self.backward_score_matching_loss(batch_size)
        
        # Endpoint loss
        loss_endpoint = self.endpoint_loss()
        
        # Total loss
        loss_total = loss_forward + loss_backward + 0.1 * loss_endpoint
        
        return {
            'total': loss_total,
            'forward': loss_forward,
            'backward': loss_backward,
            'endpoint': loss_endpoint
        }

