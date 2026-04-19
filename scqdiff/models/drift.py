"""
=======================
Hybrid drift field: score network + Neural ODE residual + RNA velocity prior.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Optional FAISS import (graceful fallback)
# ---------------------------------------------------------------------------
try:
    import faiss  # type: ignore

    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class DriftConfig:
    dim: int = 64
    hidden: int = 256
    depth: int = 4
    beta: float = 0.1

    # Velocity prior
    use_velocity_prior: bool = False
    vel_scale: float = 1.0
    vel_k: int = 32
    vel_tau: float = 1.0
    vel_time_mode: str = "mid"   # "mid" | "flat"
    vel_conf_power: float = 1.0

    # Loss weights
    alpha_control: float = 0.1
    alpha_fp: float = 0.01
    alpha_smooth: float = 0.001

    # Jacobian dimension safety threshold
    jacobian_dim_warn: int = 500


# ---------------------------------------------------------------------------
# MLP building blocks
# ---------------------------------------------------------------------------


class SinusoidalTime(nn.Module):
    """Encode scalar time t into a sinusoidal embedding."""

    def __init__(self, dim: int):
        super().__init__()
        half = dim // 2
        freq = torch.exp(-torch.arange(half) * (np.log(10_000) / (half - 1)))
        self.register_buffer("freq", freq)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) or scalar
        t = t.view(-1, 1) * self.freq.unsqueeze(0)  # (B, half)
        return torch.cat([t.sin(), t.cos()], dim=-1)  # (B, dim)


def _mlp(in_dim: int, out_dim: int, hidden: int, depth: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for _ in range(depth - 1):
        layers += [nn.Linear(prev, hidden), nn.SiLU()]
        prev = hidden
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class MLPScore(nn.Module):
    """Score network s_θ(x, t) ≈ ∇_x log ρ_t(x)."""

    def __init__(self, dim: int, hidden: int = 256, depth: int = 4):
        super().__init__()
        self.time_emb = SinusoidalTime(hidden)
        self.net = _mlp(dim + hidden, dim, hidden, depth)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = self.time_emb(t)
        if te.shape[0] == 1 and x.shape[0] > 1:
            te = te.expand(x.shape[0], -1)
        return self.net(torch.cat([x, te], dim=-1))


class ResidualNet(nn.Module):
    """Neural ODE residual correction r_θ(x, t)."""

    def __init__(self, dim: int, hidden: int = 128, depth: int = 3):
        super().__init__()
        self.time_emb = SinusoidalTime(hidden)
        self.net = _mlp(dim + hidden, dim, hidden, depth)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = self.time_emb(t)
        if te.shape[0] == 1 and x.shape[0] > 1:
            te = te.expand(x.shape[0], -1)
        return self.net(torch.cat([x, te], dim=-1))


# ---------------------------------------------------------------------------
# KNN Velocity prior  (FAISS-accelerated when available)
# ---------------------------------------------------------------------------


class KNNVelocity(nn.Module):
    """
    Soft k-NN velocity interpolation.

    Uses FAISS (if installed) for approximate nearest-neighbour search,
    falling back to exact sklearn/numpy search for smaller datasets or
    when FAISS is not available.

    Parameters
    ----------
    X_ref : torch.Tensor  (N, D)  Reference cell positions (PCA / embedding).
    V_ref : torch.Tensor  (N, D)  RNA velocity vectors at reference cells.
    k     : int           Number of neighbours.
    tau   : float         Softmax temperature.
    use_faiss : bool      Force FAISS on/off (default: auto-detect).
    """

    def __init__(
        self,
        X_ref: torch.Tensor,
        V_ref: torch.Tensor,
        k: int = 32,
        tau: float = 1.0,
        use_faiss: Optional[bool] = None,
    ):
        super().__init__()
        self.k = k
        self.tau = tau

        X_np = X_ref.detach().cpu().numpy().astype(np.float32)
        self.register_buffer("X_ref", X_ref)
        self.register_buffer("V_ref", V_ref)

        # Confidence: L2 norm of velocity (normalised to [0,1])
        conf = V_ref.norm(dim=-1)
        conf = conf / (conf.max() + 1e-8)
        self.register_buffer("conf_ref", conf)

        # Build index
        _use_faiss = _FAISS_AVAILABLE if use_faiss is None else use_faiss
        self._index = None

        if _use_faiss:
            d = X_np.shape[1]
            index = faiss.IndexFlatL2(d)
            index.add(X_np)
            self._index = index
            self._backend = "faiss"
        else:
            self._X_np = X_np
            self._backend = "numpy"

    # ------------------------------------------------------------------
    def _knn_distances(self, x: torch.Tensor):
        """Return (distances², indices) arrays for query x (B, D)."""
        x_np = x.detach().cpu().numpy().astype(np.float32)

        if self._backend == "faiss":
            D, I = self._index.search(x_np, self.k)  # (B, k)
        else:
            # Exact L2 via numpy broadcasting
            diff = self._X_np[None, :, :] - x_np[:, None, :]  # (B, N, D)
            D_all = (diff ** 2).sum(-1)                         # (B, N)
            I = np.argpartition(D_all, self.k, axis=1)[:, : self.k]
            D = np.take_along_axis(D_all, I, axis=1)

        return D, I  # both (B, k), float32 / int64

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : (B, D)

        Returns
        -------
        v_hat : (B, D)   Interpolated velocity.
        conf  : (B,)     Mean neighbour confidence.
        """
        D_np, I_np = self._knn_distances(x)
        D_t = torch.from_numpy(D_np).to(x)
        I_t = torch.from_numpy(I_np).to(x.device).long()

        # Soft-max weights
        w = torch.softmax(-D_t / self.tau, dim=-1)  # (B, k)

        # Gather velocities and confidences
        V_k = self.V_ref[I_t]       # (B, k, D)
        c_k = self.conf_ref[I_t]    # (B, k)

        v_hat = (w.unsqueeze(-1) * V_k).sum(1)   # (B, D)
        conf = (w * c_k).sum(1)                   # (B,)
        return v_hat, conf


# ---------------------------------------------------------------------------
# Main DriftField model
# ---------------------------------------------------------------------------


class DriftField(nn.Module):
    """
    Hybrid drift field:

        f(x,t) = β · score_θ(x,t) + residual_θ(x,t) + b(x,t)

    where b(x,t) is the RNA velocity prior (optional).

    Parameters
    ----------
    cfg    : DriftConfig
    X_ref  : Reference cell positions (for velocity prior).
    V_ref  : RNA velocity vectors (for velocity prior).
    """

    def __init__(
        self,
        cfg: DriftConfig,
        X_ref: Optional[torch.Tensor] = None,
        V_ref: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.cfg = cfg

        # ── Dimension safety warning ──────────────────────────────────────
        if cfg.dim > cfg.jacobian_dim_warn:
            warnings.warn(
                f"[DriftField] Input dimension ({cfg.dim}) exceeds the recommended "
                f"threshold ({cfg.jacobian_dim_warn}) for full Jacobian computation. "
                f"model.jacobian() will allocate a ({cfg.dim}, {cfg.dim}) matrix per "
                f"sample, which may exhaust memory. "
                f"Consider reducing dimensionality (e.g. PCA to 50–200 components) "
                f"before calling jacobian(), or use jacobian_approx() for random "
                f"projections.",
                stacklevel=2,
                category=ResourceWarning,
            )

        self.score = MLPScore(cfg.dim, cfg.hidden, cfg.depth)
        self.residual = ResidualNet(cfg.dim, cfg.hidden // 2, max(cfg.depth - 1, 2))

        self.vel: Optional[KNNVelocity] = None
        if cfg.use_velocity_prior and X_ref is not None and V_ref is not None:
            self.vel = KNNVelocity(X_ref, V_ref, k=cfg.vel_k, tau=cfg.vel_tau)

    # ------------------------------------------------------------------
    def _time_schedule(self, t: torch.Tensor) -> torch.Tensor:
        """g(t): time-dependent gate for velocity prior."""
        if self.cfg.vel_time_mode == "mid":
            return 4.0 * t * (1.0 - t)
        return torch.ones_like(t)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, D)
        t : (B,) or scalar — must be in [0, 1]

        Returns
        -------
        drift : (B, D)
        """
        if t.dim() == 0:
            t = t.expand(x.shape[0])

        u = self.cfg.beta * self.score(x, t) + self.residual(x, t)

        if self.vel is not None:
            v_hat, conf = self.vel(x)
            gate = conf.pow(self.cfg.vel_conf_power)          # (B,)
            g = self._time_schedule(t)                         # (B,)
            b = self.cfg.vel_scale * g * gate                  # (B,)
            u = u + b.unsqueeze(-1) * v_hat

        return u

    # ------------------------------------------------------------------
    def jacobian(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Full Jacobian ∂f/∂x via autograd.  Shape: (B, D, D).

        Warn if B*D*D tensor would be large (> 1 GB float32).
        """
        B, D = x.shape
        mem_gb = B * D * D * 4 / 1e9
        if mem_gb > 1.0:
            warnings.warn(
                f"[DriftField.jacobian] Estimated output size is {mem_gb:.1f} GB "
                f"(B={B}, D={D}). This may cause OOM. "
                f"Use a smaller batch or jacobian_approx() instead.",
                stacklevel=2,
                category=ResourceWarning,
            )

        x = x.detach().requires_grad_(True)
        f = self.forward(x, t)   # (B, D)
        J = torch.zeros(B, D, D, device=x.device, dtype=x.dtype)
        for i in range(D):
            grad = torch.autograd.grad(
                f[:, i].sum(), x, create_graph=False, retain_graph=(i < D - 1)
            )[0]
            J[:, i, :] = grad
        return J

    # ------------------------------------------------------------------
    def jacobian_approx(
        self, x: torch.Tensor, t: torch.Tensor, n_proj: int = 64
    ) -> torch.Tensor:
        """
        Memory-efficient approximate Jacobian via random projections.
        Returns (B, n_proj, D) — sufficient for eigenmode analysis.
        """
        B, D = x.shape
        x = x.detach().requires_grad_(True)
        f = self.forward(x, t)
        vecs = torch.randn(n_proj, D, device=x.device, dtype=x.dtype)
        J_approx = torch.zeros(B, n_proj, D, device=x.device, dtype=x.dtype)
        for i, v in enumerate(vecs):
            jvp = torch.autograd.grad(
                (f * v.unsqueeze(0)).sum(), x,
                create_graph=False, retain_graph=(i < n_proj - 1)
            )[0]
            J_approx[:, i, :] = jvp
        return J_approx
