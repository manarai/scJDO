"""Step 7 — operator space.

Per-cell spectral and non-normality descriptors of the Jacobian field.
Invariant tier (sign-agnostic, basis-free): eigenvalues sorted by Re,
counts of stable / unstable / near-zero / complex-pair modes, spectral gap,
Henrici index, reactivity, transient gain, eigenvector condition number.
Coordinate-dependent tier (basis-aware): leading eigenvectors.

The non-normality helpers reuse the mathematical conventions of
:mod:`scjdo.archetypes.koopman._matrix_geometry` but operate on one
``L × L`` Jacobian at a time rather than on windowed Koopman operators.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DescriptorResult:
    """Per-cell operator-space coordinates."""

    eigvals: np.ndarray           # (n_cells, L) complex, sorted desc by Re
    leading_eigvec: np.ndarray    # (n_cells, L) complex, eigenvector of eigvals[:, 0]
    lambda_max_real: np.ndarray   # (n_cells,) float — Re(leading)
    n_unstable: np.ndarray        # (n_cells,) int   — Re > +near_zero
    n_stable: np.ndarray          # (n_cells,) int   — Re < −near_zero
    n_near_zero: np.ndarray       # (n_cells,) int
    n_complex_pairs: np.ndarray   # (n_cells,) int
    spectral_gap: np.ndarray      # (n_cells,) float — Re(λ_0) − Re(λ_1)
    henrici: np.ndarray           # (n_cells,) float — departure-from-normality (sqrt-form)
    reactivity: np.ndarray        # (n_cells,) float — max Re eig of (J + J^T)/2
    transient_gain: np.ndarray    # (n_cells,) float — ‖exp(J t*)‖ at optimum t*
    eigvec_cond: np.ndarray       # (n_cells,) float — cond(V) for J = V Λ V^-1
    recovered: np.ndarray         # (n_cells,) bool


def _descriptors_single(J: np.ndarray, near_zero: float, geom_t_max: float) -> dict:
    L = J.shape[0]
    try:
        eigvals, eigvecs = np.linalg.eig(J)
    except np.linalg.LinAlgError:
        return {}

    order = np.argsort(-np.real(eigvals))
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    leading = eigvecs[:, 0]

    re = np.real(eigvals)
    im = np.imag(eigvals)
    unstable = int(np.sum(re > near_zero))
    stable = int(np.sum(re < -near_zero))
    near_zero_mask = np.abs(re) <= near_zero
    nz = int(near_zero_mask.sum())
    # Approximate complex-conjugate pair count: half the number with non-zero imag.
    n_pairs = int(np.sum(np.abs(im) > 1e-6)) // 2
    gap = float(re[0] - re[1]) if L >= 2 else 0.0

    # Henrici: sqrt(||J||_F^2 − Σ|λ_i|^2); 0 for normal J.
    frob2 = float((J.conj() * J).sum().real) if np.iscomplexobj(J) else float((J ** 2).sum())
    hen = float(np.sqrt(max(frob2 - float(np.sum(np.abs(eigvals) ** 2)), 0.0)))

    # Reactivity: largest real eigenvalue of (J + J^T) / 2.
    sym = 0.5 * (J + J.T)
    try:
        reactivity = float(np.linalg.eigvalsh(sym).max())
    except np.linalg.LinAlgError:
        reactivity = float("nan")

    # Transient gain: max over t in a short grid of ‖exp(J t)‖_2.
    ts = np.linspace(0.0, geom_t_max, 16, dtype=np.float32)
    try:
        from scipy.linalg import expm
        gains = [float(np.linalg.norm(expm(J * float(t)), 2)) for t in ts]
        trans_gain = max(gains)
    except Exception:
        trans_gain = float("nan")

    # Eigvec conditioning: cond(V) where J = V Λ V^{-1}.
    try:
        cond = float(np.linalg.cond(eigvecs))
    except np.linalg.LinAlgError:
        cond = float("nan")

    return dict(
        eigvals=eigvals.astype(np.complex64),
        leading=leading.astype(np.complex64),
        lambda_max_real=float(re[0]),
        n_unstable=unstable, n_stable=stable, n_near_zero=nz,
        n_complex_pairs=n_pairs, spectral_gap=gap,
        henrici=hen, reactivity=reactivity,
        transient_gain=trans_gain, eigvec_cond=cond,
    )


def operator_descriptors(
    J_field: np.ndarray,
    recovered: np.ndarray,
    *,
    near_zero: float = 1e-3,
    geom_t_max: float = 2.0,
) -> DescriptorResult:
    """Compute per-cell descriptors over the operator field."""
    n, L, L2 = J_field.shape
    assert L == L2

    eigvals = np.full((n, L), np.nan, dtype=np.complex64)
    leading = np.full((n, L), np.nan, dtype=np.complex64)
    lm = np.full(n, np.nan, dtype=np.float32)
    nu = np.zeros(n, dtype=np.int32)
    ns = np.zeros(n, dtype=np.int32)
    nz = np.zeros(n, dtype=np.int32)
    np_pairs = np.zeros(n, dtype=np.int32)
    gap = np.full(n, np.nan, dtype=np.float32)
    hen = np.full(n, np.nan, dtype=np.float32)
    react = np.full(n, np.nan, dtype=np.float32)
    gain = np.full(n, np.nan, dtype=np.float32)
    cond = np.full(n, np.nan, dtype=np.float32)

    for c in range(n):
        if not recovered[c] or np.any(~np.isfinite(J_field[c])):
            continue
        d = _descriptors_single(J_field[c], near_zero=near_zero, geom_t_max=geom_t_max)
        if not d:
            continue
        eigvals[c] = d["eigvals"]
        leading[c] = d["leading"]
        lm[c] = d["lambda_max_real"]
        nu[c] = d["n_unstable"]
        ns[c] = d["n_stable"]
        nz[c] = d["n_near_zero"]
        np_pairs[c] = d["n_complex_pairs"]
        gap[c] = d["spectral_gap"]
        hen[c] = d["henrici"]
        react[c] = d["reactivity"]
        gain[c] = d["transient_gain"]
        cond[c] = d["eigvec_cond"]

    return DescriptorResult(
        eigvals=eigvals, leading_eigvec=leading,
        lambda_max_real=lm,
        n_unstable=nu, n_stable=ns, n_near_zero=nz,
        n_complex_pairs=np_pairs, spectral_gap=gap,
        henrici=hen, reactivity=react, transient_gain=gain,
        eigvec_cond=cond, recovered=recovered.copy(),
    )
