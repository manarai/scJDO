"""
Ground-truth toggle-switch machinery used across T2–T9.

Everything is pure NumPy / analytic, no training. Reproduces the constants
used in ``analysis/supp_saddle_localization/saddle_readouts.py`` so results
compare directly.

Conventions
-----------
- z ∈ R^2 with columns (x, y).
- The symmetric fixed point is s(α) with s + s^5 = α, computed by fixed-point
  iteration in ``sym_fp``.
- Antisymmetric coordinate:  w = (x - y) / sqrt(2)          (transverse mode)
- Symmetric coordinate:      s = (x + y) / sqrt(2)          (drift-tangent)
- Reflection σ: (x, y) ↔ (y, x), which in (s, w) is (s, -w).
- α is ramped linearly with pseudotime τ:  α(τ) = α_min + (α_max − α_min) τ.

Analytic transverse eigenvalue
------------------------------
At the symmetric fixed point (x, y) = (s, s):
    J = [[-1, -c], [-c, -1]],   c = 4 α s^3 / (1 + s^4)^2
Eigen-decomposition of that 2x2:
    (1, 1)/sqrt(2)  →  λ_sym  = -1 - c
    (1,-1)/sqrt(2)  →  λ_anti = -1 + c    ← the bifurcating (transverse) mode
The pitchfork is at c = 1, i.e. 3 s^4 = 1, i.e. α_crit = 3^(-1/4)·(1 + 1/3).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Constants — match analysis/supp_saddle_localization/saddle_readouts.py
# ---------------------------------------------------------------------------
ALPHA_MIN = 1.0
ALPHA_MAX = 4.0
SIGMA_SDE = 0.20
NOISE_SIGMA = 0.3   # gene-space observation noise
N_LATENT = 20
D_GENES = 200

ALPHA_CRIT = 3.0 ** (-0.25) * (1.0 + 1.0 / 3.0)         # ≈ 1.0131
TAU_CRIT = float((ALPHA_CRIT - ALPHA_MIN) / (ALPHA_MAX - ALPHA_MIN))  # ≈ 0.00437


# ---------------------------------------------------------------------------
# Symmetric fixed point and its stability
# ---------------------------------------------------------------------------

def sym_fp(alpha: float, tol: float = 1e-12, max_iter: int = 200) -> float:
    """Real positive root of s + s^5 = alpha.

    The naive fixed-point iteration x_{n+1} = alpha/(1+x_n^4) DIVERGES for
    alpha > alpha_crit because the symmetric fixed point is unstable past the
    pitchfork — that iteration converges only when |ds/dx (α/(1+x^4))| < 1,
    which fails on exactly the interesting regime. We use Newton's method on
    F(s) = s + s^5 - alpha instead, which converges globally on s > 0 for
    alpha > 0 because F is strictly increasing and convex on that ray.
    """
    if alpha <= 0.0:
        return 0.0
    # Warm start with the α → 0 (s ≈ α) and α → ∞ (s ≈ α^{1/5}) regimes.
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
    """f(z; α) for the symmetric 2-D toggle switch."""
    x, y = z[..., 0], z[..., 1]
    fx = alpha / (1.0 + y ** 4) - x
    fy = alpha / (1.0 + x ** 4) - y
    return np.stack([fx, fy], axis=-1)


def toggle_jacobian_at(x: float, y: float, alpha: float) -> np.ndarray:
    """Analytic Jacobian of ``toggle_drift`` at (x, y; α). 2x2."""
    # d(fx)/dx = -1,   d(fx)/dy = -4 α y^3 / (1 + y^4)^2
    # d(fy)/dx = -4 α x^3 / (1 + x^4)^2,   d(fy)/dy = -1
    a = -4.0 * alpha * y ** 3 / (1.0 + y ** 4) ** 2
    b = -4.0 * alpha * x ** 3 / (1.0 + x ** 4) ** 2
    return np.array([[-1.0, a], [b, -1.0]], dtype=np.float64)


def transverse_eigenvalue(alpha: float) -> float:
    """λ_⊥ = −1 + c at the symmetric fixed point s(α)."""
    s = sym_fp(alpha)
    c = 4.0 * alpha * s ** 3 / (1.0 + s ** 4) ** 2
    return -1.0 + c


def symmetric_eigenvalue(alpha: float) -> float:
    """λ_∥ = −1 − c at the symmetric fixed point s(α)."""
    s = sym_fp(alpha)
    c = 4.0 * alpha * s ** 3 / (1.0 + s ** 4) ** 2
    return -1.0 - c


def alpha_of_tau(tau, alpha_min: float = ALPHA_MIN, alpha_max: float = ALPHA_MAX):
    return alpha_min + (alpha_max - alpha_min) * tau


def tau_of_alpha(alpha, alpha_min: float = ALPHA_MIN, alpha_max: float = ALPHA_MAX):
    return (alpha - alpha_min) / (alpha_max - alpha_min)


# ---------------------------------------------------------------------------
# Simulator (vectorized) — kept here so tests use exactly one implementation
# ---------------------------------------------------------------------------

def simulate_v3(
    n_cells: int = 1500,
    seed: int = 42,
    T_total: float = 10.0,
    dt: float = 0.02,
    sigma_sde: float = SIGMA_SDE,
    alpha_min: float = ALPHA_MIN,
    alpha_max: float = ALPHA_MAX,
):
    """
    Vectorized simulator matching ``saddle_readouts.make_v3``.

    Returns (z, tau) where z has shape (n_cells, 2) and tau ∈ [0, 1].
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


# ---------------------------------------------------------------------------
# Observation map (used to build lifted-truth drift in latent coordinates)
# ---------------------------------------------------------------------------

def build_observation_map(seed: int = 42, d_genes: int = D_GENES):
    """Return the 2xD gene projection W used in the manuscript pipeline."""
    rng_proj = np.random.default_rng(seed + 1)
    W = rng_proj.standard_normal((2, d_genes)).astype(np.float32)
    W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-8
    return W


def latent_axes(z_true: np.ndarray, W: np.ndarray, noise_sigma: float = NOISE_SIGMA,
                seed: int = 42, n_latent: int = N_LATENT):
    """
    Fit PCA on the noisy gene-space observation and return the reduction.

    Returns
    -------
    Z_lat : (N, n_latent) latent coordinates
    A     : (n_latent, 2) matrix such that Z_lat ≈ z_true @ (W A) up to noise,
            i.e. the map from ground-truth 2-D space to the latent 20-D space is
                z_true (∈ R^2)  →  z_true @ W (∈ R^200)  →  ⋯ PCA ⋯  →  Z_lat (∈ R^20)
    pca   : the fitted sklearn PCA object (useful for lifting/pulling back)
    """
    from sklearn.decomposition import PCA

    rng_g = np.random.default_rng(seed)
    X_obs = z_true @ W
    X_obs = X_obs + noise_sigma * rng_g.standard_normal(X_obs.shape).astype(np.float32)
    pca = PCA(n_components=n_latent, random_state=seed)
    Z_lat = pca.fit_transform(X_obs).astype(np.float32)

    # The linear map from ground-truth 2-D to latent 20-D (ignoring the PCA
    # mean, which is a per-column offset and does not enter the Jacobian).
    # Z_lat ≈ (X_obs − μ) @ V^T = (z_true @ W − μ) @ pca.components_.T
    # so the Jacobian ∂z_true → z_lat is  A = W @ pca.components_.T   (2 × n_latent).
    A_2_to_lat = (W @ pca.components_.T).astype(np.float32)  # (2, n_latent)
    return Z_lat, A_2_to_lat, pca
