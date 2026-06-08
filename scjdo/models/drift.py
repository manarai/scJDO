"""
Hybrid drift field with FiLM time-conditioning, spectral norm, and flexible velocity gate.

Architecture
------------
    f(x, t) = β · score_θ(x, t) + residual_θ(x, t) + v_prior(x, t)

Both score and residual use FiLM (Feature-wise Linear Modulation) at every
hidden layer, which is architecturally superior to simple time-concatenation:
time modulates *how* the network processes x, not just what it sees as input.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DriftConfig:
    dim: int = 64
    hidden: int = 256
    depth: int = 4
    beta: float = 0.1
    use_spectral_norm: bool = True      # spectral norm on output layer for Jacobian stability

    # Velocity prior
    use_velocity_prior: bool = False
    vel_scale: float = 2.0
    vel_k: int = 15
    vel_tau: float = 1.0
    vel_time_mode: str = "flat"         # "flat" | "mid" | "root" | "rise"
    vel_conf_power: float = 1.0

    # Loss weights (used by tl.fit_drift)
    alpha_control: float = 0.001

    # Safety
    jacobian_dim_warn: int = 500


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SinusoidalEmbed(nn.Module):
    """Sinusoidal time → learned projection."""

    def __init__(self, dim: int):
        super().__init__()
        half = dim // 2
        freq = torch.exp(-torch.arange(half) * (np.log(10_000) / max(half - 1, 1)))
        self.register_buffer("freq", freq)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.unsqueeze(0)
        t = t.view(-1, 1) * self.freq.unsqueeze(0)
        emb = torch.cat([t.sin(), t.cos()], dim=-1)
        return self.proj(emb)


class FiLMLayer(nn.Module):
    """Scale + shift hidden state by time embedding."""

    def __init__(self, hidden: int, emb_dim: int):
        super().__init__()
        self.gamma = nn.Linear(emb_dim, hidden)
        self.beta  = nn.Linear(emb_dim, hidden)

    def forward(self, h: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        return h * (1.0 + self.gamma(emb)) + self.beta(emb)


class FiLMNet(nn.Module):
    """
    MLP with FiLM conditioning at every hidden layer.

    Time modulates intermediate representations rather than being
    concatenated to the input — stronger inductive bias for temporal data.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: int,
        depth: int,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        self.emb    = SinusoidalEmbed(hidden)
        self.input  = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(depth - 1)])
        self.films  = nn.ModuleList([FiLMLayer(hidden, hidden) for _ in range(depth - 1)])
        self.act    = nn.SiLU()

        out = nn.Linear(hidden, out_dim)
        self.out = nn.utils.spectral_norm(out) if use_spectral_norm else out

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        elif t.shape[0] == 1 and x.shape[0] > 1:
            t = t.expand(x.shape[0])

        emb = self.emb(t)
        h   = self.act(self.input(x))
        for layer, film in zip(self.layers, self.films):
            h = self.act(film(layer(h), emb))
        return self.out(h)


# ---------------------------------------------------------------------------
# KNN Velocity prior
# ---------------------------------------------------------------------------

class KNNVelocity(nn.Module):
    """Soft k-NN velocity interpolation (FAISS-accelerated when available)."""

    def __init__(
        self,
        X_ref: torch.Tensor,
        V_ref: torch.Tensor,
        k: int = 15,
        tau: float = 1.0,
        use_faiss: Optional[bool] = None,
    ):
        super().__init__()
        self.k   = k
        self.tau = tau

        X_np = X_ref.detach().cpu().numpy().astype(np.float32)
        self.register_buffer("X_ref", X_ref)
        self.register_buffer("V_ref", V_ref)

        conf = V_ref.norm(dim=-1)
        conf = conf / (conf.max() + 1e-8)
        self.register_buffer("conf_ref", conf)

        _use_faiss = _FAISS_AVAILABLE if use_faiss is None else use_faiss
        self._index = None
        if _use_faiss:
            index = faiss.IndexFlatL2(X_np.shape[1])
            index.add(X_np)
            self._index = index
            self._backend = "faiss"
        else:
            self._X_np   = X_np
            self._backend = "numpy"

    def _knn(self, x: torch.Tensor):
        x_np = x.detach().cpu().numpy().astype(np.float32)
        if self._backend == "faiss":
            D, I = self._index.search(x_np, self.k)
        else:
            diff  = self._X_np[None] - x_np[:, None]
            D_all = (diff ** 2).sum(-1)
            I     = np.argpartition(D_all, self.k, axis=1)[:, :self.k]
            D     = np.take_along_axis(D_all, I, axis=1)
        return D, I

    def forward(self, x: torch.Tensor):
        D_np, I_np = self._knn(x)
        D_t = torch.from_numpy(D_np).to(x)
        I_t = torch.from_numpy(I_np).to(x.device).long()
        w   = torch.softmax(-D_t / self.tau, dim=-1)
        v_hat = (w.unsqueeze(-1) * self.V_ref[I_t]).sum(1)
        conf  = (w * self.conf_ref[I_t]).sum(1)
        return v_hat, conf


# ---------------------------------------------------------------------------
# DriftField
# ---------------------------------------------------------------------------

class DriftField(nn.Module):
    """
    Hybrid drift field:

        f(x, t) = β · score_θ(x, t) + residual_θ(x, t) + v_prior(x, t)

    Both networks use FiLM time-conditioning. Output layer optionally has
    spectral normalization for Jacobian stability.

    Parameters
    ----------
    cfg   : DriftConfig
    X_ref : (N, D) reference cell positions for velocity prior.
    V_ref : (N, D) velocity vectors (RNA velocity or pseudotime gradient).
    """

    def __init__(
        self,
        cfg: DriftConfig,
        X_ref: Optional[torch.Tensor] = None,
        V_ref: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.cfg = cfg

        if cfg.dim > cfg.jacobian_dim_warn:
            warnings.warn(
                f"[DriftField] dim={cfg.dim} > {cfg.jacobian_dim_warn}. "
                f"Full Jacobian will be ({cfg.dim},{cfg.dim}) per cell — "
                f"consider PCA to ≤200 dims or use jacobian_approx().",
                ResourceWarning, stacklevel=2,
            )

        sn = cfg.use_spectral_norm
        self.score    = FiLMNet(cfg.dim, cfg.dim, cfg.hidden, cfg.depth, use_spectral_norm=sn)
        self.residual = FiLMNet(cfg.dim, cfg.dim, cfg.hidden // 2, max(cfg.depth - 1, 2))

        self.vel: Optional[KNNVelocity] = None
        if cfg.use_velocity_prior and X_ref is not None and V_ref is not None:
            self.vel = KNNVelocity(X_ref, V_ref, k=cfg.vel_k, tau=cfg.vel_tau)

    # ------------------------------------------------------------------
    def _gate(self, t: torch.Tensor) -> torch.Tensor:
        """Time-dependent gate for velocity prior."""
        mode = self.cfg.vel_time_mode
        if mode == "mid":
            return 4.0 * t * (1.0 - t)
        elif mode == "root":
            return 1.0 - t          # strongest at root, fades at tips
        elif mode == "rise":
            return t                 # grows along trajectory
        else:                        # "flat" (default)
            return torch.ones_like(t)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.expand(x.shape[0])

        u = self.cfg.beta * self.score(x, t) + self.residual(x, t)

        if self.vel is not None:
            v_hat, conf = self.vel(x)
            gate = conf.pow(self.cfg.vel_conf_power) * self._gate(t)
            u    = u + (self.cfg.vel_scale * gate).unsqueeze(-1) * v_hat

        return u

    # ------------------------------------------------------------------
    def jacobian(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Full Jacobian ∂f/∂x via autograd. Shape: (B, D, D)."""
        B, D = x.shape
        if B * D * D * 4 / 1e9 > 1.0:
            warnings.warn(
                f"[DriftField.jacobian] Output ≈ {B*D*D*4/1e9:.1f} GB. "
                "Consider jacobian_approx() or smaller batch.",
                ResourceWarning, stacklevel=2,
            )
        x = x.detach().requires_grad_(True)
        f = self.forward(x, t)
        J = torch.zeros(B, D, D, device=x.device, dtype=x.dtype)
        for i in range(D):
            grad = torch.autograd.grad(
                f[:, i].sum(), x,
                create_graph=False, retain_graph=(i < D - 1)
            )[0]
            J[:, i, :] = grad
        return J

    # ------------------------------------------------------------------
    def jacobian_approx(self, x: torch.Tensor, t: torch.Tensor, n_proj: int = 64) -> torch.Tensor:
        """Approximate Jacobian via random projections. Shape: (B, n_proj, D)."""
        B, D = x.shape
        x    = x.detach().requires_grad_(True)
        f    = self.forward(x, t)
        vecs = torch.randn(n_proj, D, device=x.device, dtype=x.dtype)
        out  = torch.zeros(B, n_proj, D, device=x.device, dtype=x.dtype)
        for i, v in enumerate(vecs):
            jvp = torch.autograd.grad(
                (f * v.unsqueeze(0)).sum(), x,
                create_graph=False, retain_graph=(i < n_proj - 1)
            )[0]
            out[:, i, :] = jvp
        return out
