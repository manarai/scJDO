"""Step 2 — drift field over the latent positions.

Thin wrapper on v1's :class:`scjdo.models.drift.DriftField`. v2 does not
re-implement DSM; it reuses v1's training loop with a reduced-scope config.

Timescale
---------
The learned ``f(z)`` has *relative* timescale — no clock calibration unless
the caller supplies a per-cell clock scale (``clock_calibrated=True`` with a
``clock_scale`` vector). Phase B leaves ``clock_calibrated=False`` and v2
reports eigenvalues without time units downstream.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from scjdo.models.drift import DriftConfig, DriftField


@dataclass
class DriftV2Result:
    """A fitted drift field over the latent positions."""

    model: DriftField
    V: np.ndarray                 # (n_cells, n_latent) per-cell velocities f(z_c)
    clock_calibrated: bool
    clock_scale: Optional[np.ndarray]  # None in Phase B
    train_loss: float
    cfg: DriftConfig


def _dsm_step(model: DriftField, z: torch.Tensor, t: torch.Tensor,
              sigma: float) -> torch.Tensor:
    """One denoising-score-matching step: Gaussian perturbation then (−eps/σ)."""
    eps = torch.randn_like(z)
    z_pert = z + sigma * eps
    score_target = -eps / sigma
    pred = model(z_pert, t)
    return ((pred - score_target) ** 2).mean()


def fit_drift_v2(
    Z: np.ndarray,
    tau: np.ndarray,
    *,
    hidden: int = 128,
    depth: int = 3,
    beta: float = 0.1,
    n_epochs: int = 1500,
    batch_size: int = 256,
    lr: float = 2e-3,
    sigma: float = 0.1,
    use_spectral_norm: bool = True,
    use_velocity_prior: bool = False,
    seed: int = 42,
    device: str = "cpu",
) -> DriftV2Result:
    """Train a drift field ``f(z, tau)`` on latent positions and pseudotime.

    v2 does not use the kNN velocity prior by default (``use_velocity_prior=False``),
    since Phase B operates on simple synthetic benchmarks where the prior is a
    seed-dependent nuisance (see v1 Known limitations in README). The caller
    can turn it back on for real datasets.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    Z = np.asarray(Z, dtype=np.float32)
    tau = np.asarray(tau, dtype=np.float32).reshape(-1)
    n, dim = Z.shape

    cfg = DriftConfig(
        dim=dim, hidden=hidden, depth=depth, beta=beta,
        use_spectral_norm=use_spectral_norm,
        use_velocity_prior=use_velocity_prior,
    )
    model = DriftField(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    Z_t = torch.from_numpy(Z).to(device)
    tau_t = torch.from_numpy(tau).to(device)

    last_loss = float("nan")
    model.train()
    for epoch in range(n_epochs):
        idx = rng.integers(0, n, size=batch_size)
        zb = Z_t[idx]
        tb = tau_t[idx]
        opt.zero_grad()
        loss = _dsm_step(model, zb, tb, sigma)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu())

    model.eval()
    with torch.no_grad():
        V_t = model(Z_t, tau_t)
    V = V_t.detach().cpu().numpy().astype(np.float32)

    return DriftV2Result(
        model=model, V=V,
        clock_calibrated=False, clock_scale=None,
        train_loss=last_loss, cfg=cfg,
    )
