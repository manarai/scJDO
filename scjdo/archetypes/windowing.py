"""
Kernel-weighted temporal windowing for local Jacobian estimation.

Replaces fixed pseudotime bins with a Gaussian-kernel grid:

    J_bar(tau; h) = sum_i w_i(tau) * J_i / sum_i w_i(tau)
    w_i(tau)      = exp( -(tau - tau_i)^2 / (2 h^2) )

The bandwidth h is selected automatically by maximising

    S(h) = R(h) * C(h) * L(h)

with R = bootstrap reproducibility (mean pairwise Pearson correlation of
lambda_max curves), C = peak contrast against the median, L = peak
amplitude / FWHM, subject to an effective-sample-size floor
n_eff(tau) = (sum_i w_i)^2 / sum_i w_i^2 >= n_eff_min.

The grid serves as a continuous temporal resolution and should not be
interpreted as independent observations; downstream summaries should
report bootstrap uncertainty alongside the point estimate.
"""
from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Core estimators
# ---------------------------------------------------------------------------

def max_real_eig_curve(J_curve: np.ndarray) -> np.ndarray:
    """Per-time max real eigenvalue. J_curve: (T, D, D) -> (T,)."""
    return np.array(
        [np.real(np.linalg.eigvals(J)).max() for J in J_curve],
        dtype=np.float32,
    )


def kernel_jacobian_tensor(
    J_per_cell: np.ndarray,
    t_per_cell: np.ndarray,
    grid: np.ndarray,
    h,
    cell_weights: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gaussian-kernel-weighted local operator on a pseudotime grid.

    Parameters
    ----------
    J_per_cell : (N, D, D) per-cell Jacobians.
    t_per_cell : (N,) pseudotime in [0, 1].
    grid       : (T,) evaluation pseudotimes.
    h          : scalar OR (T,) bandwidth (adaptive per grid point).
    cell_weights : optional (N,) multiplicative weights (e.g. branch probabilities).

    Returns
    -------
    Jbar  : (T, D, D)
    n_eff : (T,)  effective sample size at each grid point.
    w_n   : (T, N) row-normalised weights (kept for diagnostics).
    """
    grid = np.asarray(grid, dtype=np.float32)
    if np.isscalar(h):
        h_arr = np.full_like(grid, float(h))
    else:
        h_arr = np.asarray(h, dtype=np.float32)
        if h_arr.shape != grid.shape:
            raise ValueError(f"adaptive h must match grid shape, got {h_arr.shape} vs {grid.shape}")
    diffs = grid[:, None] - t_per_cell[None, :]
    w = np.exp(-0.5 * (diffs / h_arr[:, None]) ** 2)
    if cell_weights is not None:
        w = w * cell_weights[None, :]
    w_sum = w.sum(1)
    w_sq = (w ** 2).sum(1)
    n_eff = (w_sum ** 2) / np.clip(w_sq, 1e-12, None)
    w_n = w / np.clip(w_sum[:, None], 1e-12, None)
    Jflat = J_per_cell.reshape(J_per_cell.shape[0], -1).astype(np.float32)
    Jbar = (w_n @ Jflat).reshape(len(grid), *J_per_cell.shape[1:])
    return Jbar.astype(np.float32), n_eff.astype(np.float32), w_n.astype(np.float32)


def knn_adaptive_h(t_per_cell: np.ndarray, grid: np.ndarray, k: int = 80,
                   floor: float = 0.005) -> np.ndarray:
    """h(tau) = distance to k-th nearest cell in pseudotime, lower-bounded by ``floor``."""
    grid = np.asarray(grid, dtype=np.float32)
    h_arr = np.empty_like(grid)
    for i, tau in enumerate(grid):
        d = np.abs(t_per_cell - tau)
        h_arr[i] = max(float(np.partition(d, k)[k]), float(floor))
    return h_arr


# ---------------------------------------------------------------------------
# Bandwidth scoring
# ---------------------------------------------------------------------------

def _fwhm(curve: np.ndarray, grid: np.ndarray) -> float:
    pk = int(np.argmax(curve))
    if curve[pk] <= 0:
        return float("inf")
    thr = 0.5 * curve[pk]
    L = pk
    while L > 0 and curve[L] > thr:
        L -= 1
    R = pk
    while R < len(curve) - 1 and curve[R] > thr:
        R += 1
    return float(grid[R] - grid[L])


def score_bandwidth(
    J_per_cell: np.ndarray,
    t_per_cell: np.ndarray,
    grid: np.ndarray,
    h,
    n_boot: int = 20,
    seed: int = 0,
    cell_weights: Optional[np.ndarray] = None,
) -> dict:
    """Compute C, R, L, S, n_eff for a single bandwidth choice."""
    Jbar, n_eff, _ = kernel_jacobian_tensor(J_per_cell, t_per_cell, grid, h, cell_weights)
    lam = max_real_eig_curve(Jbar)
    C = float(lam.max() - np.median(lam))
    w = _fwhm(lam, grid)
    L = 0.0 if (not np.isfinite(w) or w <= 0) else float(lam.max() / max(w, 1e-6))

    rng = np.random.default_rng(seed)
    N = J_per_cell.shape[0]
    curves = []
    for _ in range(n_boot):
        idx = rng.integers(0, N, N)
        cw = None if cell_weights is None else cell_weights[idx]
        Jb, _, _ = kernel_jacobian_tensor(J_per_cell[idx], t_per_cell[idx], grid, h, cw)
        curves.append(max_real_eig_curve(Jb))
    curves = np.asarray(curves)
    Rs = []
    for i in range(len(curves)):
        for j in range(i + 1, len(curves)):
            r = np.corrcoef(curves[i], curves[j])[0, 1]
            if np.isfinite(r):
                Rs.append(r)
    R = float(np.mean(Rs)) if Rs else 0.0

    return dict(C=C, R=R, L=L, S=R * C * L,
                lam=lam, n_eff=n_eff, boots=curves)


def select_bandwidth(
    J_per_cell: np.ndarray,
    t_per_cell: np.ndarray,
    grid: np.ndarray,
    bandwidth_grid: Sequence[float] = (0.01, 0.02, 0.03, 0.05, 0.08, 0.10),
    n_eff_min: float = 30.0,
    n_boot: int = 20,
    seed: int = 0,
    cell_weights: Optional[np.ndarray] = None,
    interior: tuple[float, float] = (0.10, 0.90),
    verbose: bool = True,
) -> tuple[dict, list[dict]]:
    """Sweep bandwidth, return (chosen, full sweep table).

    `chosen` is the dict from `score_bandwidth` for the maximiser of S(h)
    subject to min(n_eff[interior_mask]) >= n_eff_min. Falls back to the
    overall S-maximiser with a warning if no candidate meets the floor.
    """
    grid = np.asarray(grid, dtype=np.float32)
    lo, hi = interior
    interior_mask = (grid >= lo) & (grid <= hi)
    table: list[dict] = []
    for h in bandwidth_grid:
        sc = score_bandwidth(J_per_cell, t_per_cell, grid, float(h),
                             n_boot=n_boot, seed=seed, cell_weights=cell_weights)
        sc["h"] = float(h)
        sc["n_eff_interior_min"] = float(sc["n_eff"][interior_mask].min())
        table.append(sc)
        if verbose:
            print(f"  h={h:.3f}  R={sc['R']:.3f}  C={sc['C']:.3f}  "
                  f"L={sc['L']:.3f}  S={sc['S']:.4f}  "
                  f"n_eff_min(interior)={sc['n_eff_interior_min']:.1f}")
    valid = [t for t in table if t["n_eff_interior_min"] >= n_eff_min]
    if not valid:
        chosen = max(table, key=lambda t: t["S"])
        warnings.warn(
            f"No bandwidth in {list(bandwidth_grid)} met n_eff_min>={n_eff_min}. "
            f"Selected h={chosen['h']} despite low effective sample size; "
            "increase bandwidth_grid or relax n_eff_min.",
            UserWarning, stacklevel=2,
        )
    else:
        chosen = max(valid, key=lambda t: t["S"])
    return chosen, table


# ---------------------------------------------------------------------------
# End-to-end builder
# ---------------------------------------------------------------------------

def build_temporal_operator(
    J_per_cell: np.ndarray,
    t_per_cell: np.ndarray,
    *,
    grid_size: int = 200,
    grid_lo: float = 0.02,
    grid_hi: float = 0.98,
    bandwidth="auto",
    bandwidth_grid: Sequence[float] = (0.01, 0.02, 0.03, 0.05, 0.08, 0.10),
    n_eff_min: float = 30.0,
    n_boot: int = 20,
    adaptive: bool = False,
    knn_k: int = 80,
    seed: int = 0,
    cell_weights: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> dict:
    """Build a (T, D, D) temporal operator via kernel-weighted aggregation.

    Returns
    -------
    dict with keys:
        J_tensor   : (T, D, D)
        grid       : (T,) pseudotime centers
        bandwidth  : scalar OR (T,) bandwidth used
        n_eff      : (T,)
        lam        : (T,) max-real-eigenvalue curve
        score      : {'R','C','L','S'} when bandwidth='auto'; else None
        sweep      : list of per-bandwidth dicts (auto only)
        mode       : 'auto' | 'manual' | 'adaptive_knn'
        boots      : (n_boot, T) lambda_max bootstrap curves at the chosen
                     bandwidth (or None for manual/adaptive)
    """
    grid = np.linspace(grid_lo, grid_hi, int(grid_size), dtype=np.float32)
    mode = "adaptive_knn" if adaptive else ("auto" if bandwidth == "auto" else "manual")

    if adaptive:
        h = knn_adaptive_h(t_per_cell, grid, k=knn_k)
        Jbar, n_eff, _ = kernel_jacobian_tensor(J_per_cell, t_per_cell, grid, h, cell_weights)
        lam = max_real_eig_curve(Jbar)
        score = None
        sweep = None
        boots = None
        bw_used: object = h

    elif bandwidth == "auto":
        if verbose:
            print(f"[kernel-windowing] selecting bandwidth from {list(bandwidth_grid)}  "
                  f"(n_boot={n_boot}, n_eff_min={n_eff_min})")
        chosen, sweep = select_bandwidth(
            J_per_cell, t_per_cell, grid,
            bandwidth_grid=bandwidth_grid,
            n_eff_min=n_eff_min, n_boot=n_boot, seed=seed,
            cell_weights=cell_weights, verbose=verbose,
        )
        bw_used = float(chosen["h"])
        Jbar, n_eff, _ = kernel_jacobian_tensor(J_per_cell, t_per_cell, grid, bw_used, cell_weights)
        lam = chosen["lam"]
        score = {k: chosen[k] for k in ("R", "C", "L", "S")}
        boots = chosen["boots"]
        if verbose:
            print(f"[kernel-windowing] selected h*={bw_used:.4f}  "
                  f"R={score['R']:.3f}  C={score['C']:.3f}  L={score['L']:.3f}  S={score['S']:.4f}")

    else:
        bw_used = float(bandwidth)
        Jbar, n_eff, _ = kernel_jacobian_tensor(J_per_cell, t_per_cell, grid, bw_used, cell_weights)
        lam = max_real_eig_curve(Jbar)
        score = None
        sweep = None
        boots = None

    return dict(
        J_tensor=Jbar,
        grid=grid,
        bandwidth=bw_used,
        n_eff=n_eff,
        lam=lam,
        score=score,
        sweep=sweep,
        boots=boots,
        mode=mode,
    )
