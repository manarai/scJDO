"""
scjdo/models/schrodinger_bridge.py
=====================================
Schrödinger Bridge for optimal transport between two cell distributions
(e.g., young → old in aging studies).

Corrections applied
-------------------
1. Convergence criterion added to train_bridge():
   - Tracks OT cost per iteration.
   - Stops early when |ΔOT_cost| < tol for `patience` consecutive iterations.
   - Emits a warning if max_iterations reached without convergence.
2. Naming: all imports use ``scjdo`` (canonical package name).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SchrodingerBridgeConfig:
    dim: int = 64
    hidden: int = 256
    depth: int = 4
    beta: float = 0.1

    # OT regularisation (Sinkhorn ε)
    epsilon: float = 0.1
    sinkhorn_max_iter: int = 100
    sinkhorn_tol: float = 1e-6

    # Training
    lr: float = 3e-4
    n_score_steps: int = 500    # gradient steps per bridge iteration

    # Convergence
    convergence_tol: float = 1e-3   # |ΔCOST| threshold
    patience: int = 3               # consecutive iters below tol → stop
    max_iterations: int = 50        # hard upper limit


# ---------------------------------------------------------------------------
# Shared MLP backbone (same pattern as drift.py)
# ---------------------------------------------------------------------------


class SinusoidalTime(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        half = dim // 2
        freq = torch.exp(-torch.arange(half) * (np.log(10_000) / (half - 1)))
        self.register_buffer("freq", freq)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1, 1) * self.freq.unsqueeze(0)
        return torch.cat([t.sin(), t.cos()], dim=-1)


def _mlp(in_dim, out_dim, hidden, depth):
    layers = []
    prev = in_dim
    for _ in range(depth - 1):
        layers += [nn.Linear(prev, hidden), nn.SiLU()]
        prev = hidden
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class BridgeNet(nn.Module):
    """Drift network for one direction of the bridge."""

    def __init__(self, dim: int, hidden: int = 256, depth: int = 4):
        super().__init__()
        self.time_emb = SinusoidalTime(hidden)
        self.net = _mlp(dim + hidden, dim, hidden, depth)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = self.time_emb(t)
        if te.shape[0] == 1 and x.shape[0] > 1:
            te = te.expand(x.shape[0], -1)
        return self.net(torch.cat([x, te], dim=-1))


# ---------------------------------------------------------------------------
# Sinkhorn (log-domain for numerical stability)
# ---------------------------------------------------------------------------


def _sinkhorn_log(
    a: torch.Tensor,
    b: torch.Tensor,
    C: torch.Tensor,
    eps: float,
    max_iter: int,
    tol: float,
) -> tuple[torch.Tensor, float]:
    """
    Log-domain Sinkhorn.

    Returns
    -------
    P   : (N, M) coupling matrix
    cost: scalar OT cost (sum of P*C)
    """
    log_a = a.log()
    log_b = b.log()
    log_K = -C / eps

    u = torch.zeros_like(log_a)
    v = torch.zeros_like(log_b)

    for _ in range(max_iter):
        u_prev = u.clone()
        u = log_a - torch.logsumexp(log_K + v.unsqueeze(0), dim=1)
        v = log_b - torch.logsumexp(log_K + u.unsqueeze(1), dim=0)
        if (u - u_prev).abs().max().item() < tol:
            break

    log_P = log_K + u.unsqueeze(1) + v.unsqueeze(0)
    P = log_P.exp()
    cost = (P * C).sum().item()
    return P, cost


# ---------------------------------------------------------------------------
# Schrödinger Bridge
# ---------------------------------------------------------------------------


class SchrodingerBridge(nn.Module):
    """
    Schrödinger Bridge between source distribution ρ₀ and target ρ₁.

    Parameters
    ----------
    cfg : SchrodingerBridgeConfig
    X_0 : (N0, D)  Source samples (e.g., young cells).
    X_1 : (N1, D)  Target samples (e.g., old cells).
    """

    def __init__(
        self,
        cfg: SchrodingerBridgeConfig,
        X_0: torch.Tensor,
        X_1: torch.Tensor,
    ):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("X_0", X_0)
        self.register_buffer("X_1", X_1)

        self.forward_net = BridgeNet(cfg.dim, cfg.hidden, cfg.depth)
        self.backward_net = BridgeNet(cfg.dim, cfg.hidden, cfg.depth)

        # Will be populated during training
        self._ot_plan: Optional[torch.Tensor] = None
        self._convergence_history: list[float] = []

    # ------------------------------------------------------------------
    def _compute_ot_plan(self) -> tuple[torch.Tensor, float]:
        """Compute entropic OT plan via log-domain Sinkhorn."""
        X0, X1 = self.X_0, self.X_1
        N, M = X0.shape[0], X1.shape[0]

        # Cost matrix: pairwise squared Euclidean
        C = torch.cdist(X0, X1, p=2).pow(2)  # (N, M)

        a = torch.full((N,), 1.0 / N, device=X0.device, dtype=X0.dtype)
        b = torch.full((M,), 1.0 / M, device=X1.device, dtype=X1.dtype)

        with torch.no_grad():
            P, cost = _sinkhorn_log(
                a, b, C,
                eps=self.cfg.epsilon,
                max_iter=self.cfg.sinkhorn_max_iter,
                tol=self.cfg.sinkhorn_tol,
            )
        return P, cost

    # ------------------------------------------------------------------
    def _sample_bridge_pairs(
        self, P: torch.Tensor, n: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample (x0, x1) pairs proportional to the OT coupling P."""
        flat = P.reshape(-1)
        idx = torch.multinomial(flat, n, replacement=True)
        i0 = idx // P.shape[1]
        i1 = idx % P.shape[1]
        return self.X_0[i0], self.X_1[i1]

    # ------------------------------------------------------------------
    def _train_one_direction(
        self,
        net: BridgeNet,
        x_start: torch.Tensor,
        x_end: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        forward: bool,
    ) -> float:
        """Score-matching training for one drift direction."""
        net.train()
        total_loss = 0.0
        B = x_start.shape[0]

        for _ in range(self.cfg.n_score_steps):
            optimizer.zero_grad()
            t = torch.rand(B, device=x_start.device)

            # Linear interpolation of bridge path
            x_t = (1 - t.unsqueeze(-1)) * x_start + t.unsqueeze(-1) * x_end

            # Target: direction toward endpoint
            if forward:
                target = (x_end - x_t) / (1 - t.unsqueeze(-1) + 1e-6)
            else:
                target = (x_start - x_t) / (t.unsqueeze(-1) + 1e-6)

            pred = net(x_t, t)
            loss = (pred - target).pow(2).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        return total_loss / self.cfg.n_score_steps

    # ------------------------------------------------------------------
    def train_bridge(
        self,
        n_iterations: Optional[int] = None,
        batch_size: int = 512,
        verbose: bool = True,
    ) -> dict:
        """
        Iterative Schrödinger Bridge training with convergence criterion.

        Parameters
        ----------
        n_iterations : int, optional
            Maximum iterations. Defaults to cfg.max_iterations.
            Training may stop earlier if the OT cost converges.
        batch_size : int
            Pairs sampled per iteration.
        verbose : bool
            Print per-iteration progress.

        Returns
        -------
        history : dict with keys:
            'ot_costs'        — OT cost per iteration
            'forward_losses'  — forward score loss per iteration
            'backward_losses' — backward score loss per iteration
            'converged'       — bool: True if early-stop triggered
            'n_iters'         — actual number of iterations run
        """
        max_iter = n_iterations or self.cfg.max_iterations
        tol = self.cfg.convergence_tol
        patience = self.cfg.patience

        fwd_opt = Adam(self.forward_net.parameters(), lr=self.cfg.lr)
        bwd_opt = Adam(self.backward_net.parameters(), lr=self.cfg.lr)

        ot_costs: list[float] = []
        fwd_losses: list[float] = []
        bwd_losses: list[float] = []

        no_improve_count = 0
        converged = False

        iter_range = tqdm(range(max_iter), desc="Bridge iterations") if verbose else range(max_iter)

        for iteration in iter_range:
            # ── Step 1: Compute OT plan ───────────────────────────────
            P, ot_cost = self._compute_ot_plan()
            self._ot_plan = P
            ot_costs.append(ot_cost)

            # ── Convergence check ─────────────────────────────────────
            if len(ot_costs) >= 2:
                delta = abs(ot_costs[-1] - ot_costs[-2])
                if delta < tol:
                    no_improve_count += 1
                    if no_improve_count >= patience:
                        converged = True
                        if verbose:
                            print(
                                f"\n✓ Converged at iteration {iteration + 1} "
                                f"(|ΔCOST| = {delta:.2e} < tol={tol:.2e} "
                                f"for {patience} consecutive iters)."
                            )
                        break
                else:
                    no_improve_count = 0  # reset on improvement

            # ── Step 2: Sample pairs ──────────────────────────────────
            x0, x1 = self._sample_bridge_pairs(P, batch_size)

            # ── Step 3: Train forward drift ───────────────────────────
            fwd_loss = self._train_one_direction(
                self.forward_net, x0, x1, fwd_opt, forward=True
            )
            fwd_losses.append(fwd_loss)

            # ── Step 4: Train backward drift ──────────────────────────
            bwd_loss = self._train_one_direction(
                self.backward_net, x1, x0, bwd_opt, forward=False
            )
            bwd_losses.append(bwd_loss)

            if verbose:
                iter_range.set_postfix(  # type: ignore[union-attr]
                    OT=f"{ot_cost:.4f}",
                    fwd=f"{fwd_loss:.4f}",
                    bwd=f"{bwd_loss:.4f}",
                    no_imp=no_improve_count,
                )

        # ── Post-loop: warn if max iterations reached without convergence ──
        if not converged:
            warnings.warn(
                f"[SchrodingerBridge] Training reached the maximum of {max_iter} "
                f"iterations without satisfying the convergence criterion "
                f"(|ΔCOST| < {tol}). Consider increasing max_iterations, "
                f"loosening convergence_tol, or increasing epsilon (OT regularisation).",
                stacklevel=2,
                category=UserWarning,
            )

        self._convergence_history = ot_costs
        return {
            "ot_costs": ot_costs,
            "forward_losses": fwd_losses,
            "backward_losses": bwd_losses,
            "converged": converged,
            "n_iters": len(ot_costs),
        }

    # ------------------------------------------------------------------
    def _euler_integrate(
        self,
        net: BridgeNet,
        x0: torch.Tensor,
        steps: int = 100,
        t_start: float = 0.0,
        t_end: float = 1.0,
        stochastic: bool = False,
    ) -> torch.Tensor:
        """Euler-Maruyama integration of one drift direction."""
        net.eval()
        x = x0.clone()
        dt = (t_end - t_start) / steps
        trajectories = [x.unsqueeze(1)]

        with torch.no_grad():
            for step in range(steps):
                t_val = t_start + step * dt
                t = torch.full((x.shape[0],), t_val, device=x.device, dtype=x.dtype)
                drift = net(x, t)
                x = x + drift * dt
                if stochastic:
                    x = x + (2 * self.cfg.beta * abs(dt)) ** 0.5 * torch.randn_like(x)
                trajectories.append(x.unsqueeze(1))

        return torch.cat(trajectories, dim=1)  # (B, steps+1, D)

    def forward_integrate(
        self, x0: torch.Tensor, steps: int = 100, stochastic: bool = False
    ) -> torch.Tensor:
        """Simulate aging: ρ₀ → ρ₁."""
        return self._euler_integrate(self.forward_net, x0, steps, stochastic=stochastic)

    def backward_integrate(
        self, x1: torch.Tensor, steps: int = 100, stochastic: bool = False
    ) -> torch.Tensor:
        """Simulate rejuvenation: ρ₁ → ρ₀."""
        return self._euler_integrate(
            self.backward_net, x1, steps, t_start=1.0, t_end=0.0, stochastic=stochastic
        )

    # ------------------------------------------------------------------
    def jacobian(self, x: torch.Tensor, t: torch.Tensor, forward: bool = True) -> torch.Tensor:
        """Full Jacobian ∂f/∂x for forward or backward drift. Shape (B, D, D)."""
        net = self.forward_net if forward else self.backward_net
        x = x.detach().requires_grad_(True)
        D = x.shape[1]
        f = net(x, t)
        J = torch.zeros(x.shape[0], D, D, device=x.device, dtype=x.dtype)
        for i in range(D):
            grad = torch.autograd.grad(
                f[:, i].sum(), x,
                create_graph=False, retain_graph=(i < D - 1)
            )[0]
            J[:, i, :] = grad
        return J
