"""2-D toggle-switch pitchfork — the Phase B acceptance synthetic.

Ported from :mod:`reproducibility.code.toggle_truth` so v2 tests do not need
to reach into ``reproducibility/`` on sys.path. The arithmetic constants are
unchanged, so the two implementations give identical trajectories.

Dynamics
--------
Two-gene toggle switch, Hill exponent 4::

    dx/dt = alpha / (1 + y^4) − x
    dy/dt = alpha / (1 + x^4) − y

with ``alpha`` ramped linearly over pseudotime ``tau ∈ [0, 1]``::

    alpha(tau) = alpha_min + (alpha_max − alpha_min) * tau

The symmetric fixed point ``s(alpha)`` (root of ``s + s^5 = alpha``) loses
stability through a supercritical pitchfork at::

    alpha_crit = 3^(−1/4) * (1 + 1/3) ≈ 1.0131

At the default range ``(alpha_min, alpha_max) = (1.0, 4.0)`` this puts the
bifurcation at ``tau_crit ≈ 0.00437`` — the hardest localisation case.
"""
from __future__ import annotations

import numpy as np


# Default constants (match ``saddle_readouts.make_v3``).
ALPHA_MIN = 1.0
ALPHA_MAX = 4.0
SIGMA_SDE = 0.20
NOISE_SIGMA = 0.3
N_LATENT = 20
D_GENES = 200

ALPHA_CRIT = 3.0 ** (-0.25) * (1.0 + 1.0 / 3.0)
TAU_CRIT = float((ALPHA_CRIT - ALPHA_MIN) / (ALPHA_MAX - ALPHA_MIN))


def sym_fp(alpha: float, tol: float = 1e-12, max_iter: int = 200) -> float:
    """Positive root of ``s + s^5 = alpha`` via Newton on a strictly convex F."""
    if alpha <= 0.0:
        return 0.0
    s = min(alpha, alpha ** 0.2) if alpha < 1.0 else max(1.0, alpha ** 0.2)
    for _ in range(max_iter):
        F = s + s ** 5 - alpha
        Fp = 1.0 + 5.0 * s ** 4
        ds = F / Fp
        s -= ds
        if abs(ds) < tol:
            break
    return float(s)


def toggle_drift(z: np.ndarray, alpha: float) -> np.ndarray:
    x, y = z[..., 0], z[..., 1]
    fx = alpha / (1.0 + y ** 4) - x
    fy = alpha / (1.0 + x ** 4) - y
    return np.stack([fx, fy], axis=-1)


def toggle_jacobian_at(x: float, y: float, alpha: float) -> np.ndarray:
    a = -4.0 * alpha * y ** 3 / (1.0 + y ** 4) ** 2
    b = -4.0 * alpha * x ** 3 / (1.0 + x ** 4) ** 2
    return np.array([[-1.0, a], [b, -1.0]], dtype=np.float64)


def transverse_eigenvalue(alpha: float) -> float:
    s = sym_fp(alpha)
    c = 4.0 * alpha * s ** 3 / (1.0 + s ** 4) ** 2
    return -1.0 + c


def alpha_of_tau(tau, alpha_min: float = ALPHA_MIN, alpha_max: float = ALPHA_MAX):
    return alpha_min + (alpha_max - alpha_min) * tau


def tau_crit_at(alpha_min: float = ALPHA_MIN, alpha_max: float = ALPHA_MAX) -> float:
    return float((ALPHA_CRIT - alpha_min) / (alpha_max - alpha_min))


def simulate_v3(
    n_cells: int = 1500,
    seed: int = 42,
    T_total: float = 10.0,
    dt: float = 0.02,
    sigma_sde: float = SIGMA_SDE,
    alpha_min: float = ALPHA_MIN,
    alpha_max: float = ALPHA_MAX,
) -> tuple[np.ndarray, np.ndarray]:
    """Snapshot simulator matching ``saddle_readouts.make_v3``.

    Returns ``(z, tau)`` with shapes ``(n_cells, 2)`` and ``(n_cells,)``.
    ``tau`` is each cell's observation time scaled into ``[0, 1]``.
    """
    rng = np.random.default_rng(seed)
    n_steps = int(np.ceil(T_total / dt))
    target_step = np.sort(rng.integers(1, n_steps + 1, size=n_cells))
    t_per_cell = (target_step * dt) / T_total
    z = np.full((n_cells, 2), sym_fp(alpha_min), dtype=np.float32)
    z += 0.02 * rng.standard_normal(z.shape).astype(np.float32)
    z_out = np.zeros_like(z)
    sqrt_dt = float(np.sqrt(dt))
    idx = 0
    for step in range(1, n_steps + 1):
        a_t = alpha_min + (alpha_max - alpha_min) * (step * dt) / T_total
        z = (
            z
            + toggle_drift(z, a_t).astype(np.float32) * dt
            + sigma_sde * sqrt_dt * rng.standard_normal(z.shape).astype(np.float32)
        )
        while idx < n_cells and target_step[idx] == step:
            z_out[idx] = z[idx]
            idx += 1
    return z_out.astype(np.float32), t_per_cell.astype(np.float32)


def analytic_drift(z: np.ndarray, tau: np.ndarray,
                   alpha_min: float = ALPHA_MIN,
                   alpha_max: float = ALPHA_MAX) -> np.ndarray:
    """Ground-truth drift at each (z_i, tau_i). Shape (N, 2)."""
    alpha = alpha_min + (alpha_max - alpha_min) * tau  # (N,)
    x, y = z[:, 0], z[:, 1]
    fx = alpha / (1.0 + y ** 4) - x
    fy = alpha / (1.0 + x ** 4) - y
    return np.stack([fx, fy], axis=-1).astype(np.float32)


def analytic_jacobian_at_cells(z: np.ndarray, tau: np.ndarray,
                                alpha_min: float = ALPHA_MIN,
                                alpha_max: float = ALPHA_MAX) -> np.ndarray:
    """Analytic Jacobian ∂f/∂z at each cell. Shape (N, 2, 2).

    Note: in this 2-D toggle model ``f`` has no explicit ``z``-dependent
    diagonal contribution beyond ``−x``/``−y`` and no ``alpha`` dependence in
    the diagonal, so ``J[i, 0, 0] = J[i, 1, 1] = −1``. The off-diagonals
    carry the Hill-nonlinearity coupling.
    """
    alpha = alpha_min + (alpha_max - alpha_min) * tau
    x, y = z[:, 0], z[:, 1]
    off_xy = -4.0 * alpha * y ** 3 / (1.0 + y ** 4) ** 2
    off_yx = -4.0 * alpha * x ** 3 / (1.0 + x ** 4) ** 2
    J = np.zeros((z.shape[0], 2, 2), dtype=np.float32)
    J[:, 0, 0] = -1.0
    J[:, 1, 1] = -1.0
    J[:, 0, 1] = off_xy
    J[:, 1, 0] = off_yx
    return J
