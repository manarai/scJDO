"""Step 8 — archetypes over the operator field.

Semi-NMF on stacked per-cell Jacobians, exactly v1's objective but with the
"time" axis replaced by cells. Fisher weighting is deferred to Phase C (unit
weights here). ``K`` is selected by held-out reconstruction likelihood over
a candidate grid.

Archetypes are **named** from their eigenvalues:

- ``holder``    — all Re(λ) < 0, no near-zero mode (deeply stable).
- ``damped``    — all Re(λ) < 0 with a dominant complex-conjugate pair.
- ``drift``     — exactly one near-zero mode, rest stable.
- ``fork``      — one positive real mode (unstable, no oscillation).
- ``growing``   — a complex-conjugate pair with Re(λ) > 0 (growing oscillator).

A ``sensitive`` flag fires when the Henrici index exceeds a quantile cutoff
of the archetype distribution, flagging non-normal (transient-amplifying)
operators.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from scjdo.archetypes.decompose import _seminmf


@dataclass
class ArchetypeResult:
    """Fitted archetype operators and per-cell weights."""

    patterns: np.ndarray          # (K, L, L) archetype Jacobians
    weights: np.ndarray           # (n_cells, K) >= 0, rescaled
    reconstruction_error: float
    K: int
    names: list[str]              # one per archetype
    sensitive: list[bool]
    K_sweep: Optional[list[dict]] = None


def _name_archetype(A: np.ndarray, near_zero: float = 1e-3) -> tuple[str, dict]:
    """Classify an archetype operator by its eigenvalue signature."""
    try:
        eig = np.linalg.eigvals(A)
    except np.linalg.LinAlgError:
        return "unknown", {}
    re = np.real(eig)
    im = np.imag(eig)
    unstable = int(np.sum(re > near_zero))
    stable = int(np.sum(re < -near_zero))
    nz = int(np.sum(np.abs(re) <= near_zero))
    has_osc = bool(np.any(np.abs(im) > 1e-6))

    if unstable == 0 and nz == 0 and has_osc:
        name = "damped"
    elif unstable == 0 and nz == 0:
        name = "holder"
    elif unstable == 0 and nz >= 1:
        name = "drift"
    elif unstable >= 1 and has_osc:
        name = "growing"
    elif unstable >= 1:
        name = "fork"
    else:
        name = "unknown"
    return name, dict(unstable=unstable, stable=stable, near_zero=nz, oscillatory=has_osc)


def _heldout_fro(J_rows: np.ndarray, K: int, hold_frac: float, seed: int) -> float:
    """Hold out a fraction of cells, fit on the rest, score reconstruction."""
    n = J_rows.shape[0]
    rng = np.random.default_rng(seed)
    hold = rng.choice(n, size=max(1, int(hold_frac * n)), replace=False)
    train = np.setdiff1d(np.arange(n), hold)
    W_tr, H, _ = _seminmf(J_rows[train].astype(np.float32), rank=K, seed=seed)
    # Score the held-out cells by projecting with NNLS (same solver as training).
    from scipy.optimize import nnls
    A = H.T.astype(np.float64)
    err = 0.0
    for c in hold:
        w, _ = nnls(A, J_rows[c].astype(np.float64))
        err += float(np.linalg.norm(J_rows[c] - w @ H))
    return -err  # higher = better


def fit_archetypes(
    J_field: np.ndarray,
    recovered: np.ndarray,
    *,
    K: int | str = "auto",
    K_sweep: Sequence[int] = (2, 3, 4, 5, 6),
    weights: Optional[np.ndarray] = None,
    n_restarts: int = 3,
    seed: int = 42,
    verbose: bool = False,
) -> ArchetypeResult:
    """Semi-NMF archetype decomposition over the per-cell Jacobian field.

    Parameters
    ----------
    J_field
        ``(n_cells, L, L)`` Jacobian field from :func:`build_operator_field`.
    recovered
        ``(n_cells,)`` bool mask of recovered cells; unrecovered cells are
        excluded from the fit.
    K
        Number of archetypes, or ``'auto'`` to sweep ``K_sweep``.
    weights
        Optional per-cell weights (Fisher-weighted in Phase C). Must be
        non-negative; omitted → uniform weights. Weights are embedded via
        row-scaling in the semi-NMF objective (``diag(sqrt(w)) @ M``).
    """
    J_field = np.asarray(J_field, dtype=np.float32)
    n, L, _ = J_field.shape
    mask = recovered & np.all(np.isfinite(J_field).reshape(n, -1), axis=1)
    if mask.sum() < max(K_sweep) + 2:
        raise ValueError(f"Not enough recovered cells ({int(mask.sum())}) to fit archetypes.")

    J_rows = J_field[mask].reshape(mask.sum(), L * L).astype(np.float32)

    if weights is None:
        w = np.ones(mask.sum(), dtype=np.float32)
    else:
        w = np.asarray(weights, dtype=np.float32)[mask]
        if np.any(w < 0):
            raise ValueError("weights must be non-negative.")
    sw = np.sqrt(w + 1e-8)[:, None]
    M = J_rows * sw                                   # row-scaled design

    sweep = None
    if K == "auto":
        sweep = []
        for K_try in K_sweep:
            if K_try >= mask.sum():
                continue
            ll = _heldout_fro(M, K_try, hold_frac=0.2, seed=seed)
            sweep.append({"K": int(K_try), "heldout_score": ll})
            if verbose:
                print(f"[v2.archetypes] K={K_try} heldout_score={ll:.3f}")
        best = max(sweep, key=lambda r: r["heldout_score"])
        K_sel = int(best["K"])
    else:
        K_sel = int(K)

    best_W, best_H, best_err = None, None, np.inf
    for s in range(n_restarts):
        W, H, err = _seminmf(M, rank=K_sel, seed=seed + s)
        if err < best_err:
            best_W, best_H, best_err = W, H, err

    W_full = np.zeros((n, K_sel), dtype=np.float32)
    W_full[mask] = (best_W / (sw + 1e-8)).astype(np.float32)
    patterns = best_H.reshape(K_sel, L, L).astype(np.float32)

    names: list[str] = []
    sensitive: list[bool] = []
    henricis: list[float] = []
    for k in range(K_sel):
        A = patterns[k]
        name, _info = _name_archetype(A)
        names.append(name)
        frob2 = float((A ** 2).sum())
        eig = np.linalg.eigvals(A)
        hen = float(np.sqrt(max(frob2 - float(np.sum(np.abs(eig) ** 2)), 0.0)))
        henricis.append(hen)
    cutoff = float(np.quantile(henricis, 0.75)) if henricis else 0.0
    for h in henricis:
        sensitive.append(h >= cutoff and h > 0.0)

    return ArchetypeResult(
        patterns=patterns, weights=W_full,
        reconstruction_error=float(best_err),
        K=K_sel, names=names, sensitive=sensitive,
        K_sweep=sweep,
    )
