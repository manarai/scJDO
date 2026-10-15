"""Step 5 — neighbourhood Jacobian field.

For each cell c, estimate ``J_c`` by regressing per-cell rate differences on
per-cell level differences over c's kNN in latent space::

    v_{c'} − v_c   ≈   J_c · (z_{c'} − z_c)       for c' ∈ N(c)

with Gaussian weights on neighbour distance. A smoothness penalty ties
``J_c`` to its neighbours' Jacobians (one Jacobi pass by default); this is
the operator-field analogue of v1's kernel-windowing smoothing but defined
over the latent-space graph instead of over pseudotime.

Phase B: ``v_c = f(z_c)`` comes from v1's trained drift (``DriftV2Result.V``),
so levels and rates are both in latent coordinates and the Jacobians are
``(L, L)``. Phase C replaces the "rate" input with the autodiff'd NB rate
model so each Jacobian is a full-gene operator with the Fisher ledger.

k selection
-----------
Pass ``k='auto'`` to sweep ``{15, 25, 50, 100}`` on a held-out 20% of cells
and pick the ``k`` that maximises Gaussian likelihood of predicted vs.
observed drift differences on the held-out set.

No silent fallbacks
-------------------
If a cell has fewer than ``L + 1`` neighbours within ``radius_floor``
(2× median neighbour distance), its row of the field is left as ``NaN`` and
its ``recovered`` flag is set to ``False``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import numpy as np
from sklearn.neighbors import NearestNeighbors


KSpec = Literal["auto"] | int


@dataclass
class OperatorFieldResult:
    """Per-cell neighbourhood Jacobians over the latent-space graph."""

    J: np.ndarray                 # (n_cells, L, L) — NaN rows for unrecovered cells
    recovered: np.ndarray         # (n_cells,) bool
    k_selected: int
    neighbors: np.ndarray         # (n_cells, k) int32 — cell indices
    neighbor_dists: np.ndarray    # (n_cells, k) float32
    weights: np.ndarray           # (n_cells, k) float32 — Gaussian kernel weights
    residuals: np.ndarray         # (n_cells,) float32 — per-cell training residual
    k_sweep: Optional[list[dict]] = None  # held-out LL for each candidate k
    tied_lambda: float = 0.0


def _knn(Z: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """kNN including the query cell as neighbour 0."""
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
    nn.fit(Z)
    dists, idx = nn.kneighbors(Z, n_neighbors=k, return_distance=True)
    return idx.astype(np.int32), dists.astype(np.float32)


def _cellwise_lstsq(
    Z: np.ndarray,
    V: np.ndarray,
    knn_idx: np.ndarray,
    knn_dist: np.ndarray,
    bandwidth: float,
    ridge: float,
    min_eff_rank: int,
    radius_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Local-linear fit of ``J_c`` for every cell.

    Returns ``(J, recovered, weights, residuals)``.
    """
    n, L = Z.shape
    k = knn_idx.shape[1]
    J = np.full((n, L, L), np.nan, dtype=np.float32)
    recovered = np.zeros(n, dtype=bool)
    W_out = np.zeros((n, k), dtype=np.float32)
    residuals = np.full(n, np.nan, dtype=np.float32)

    for c in range(n):
        nbr = knn_idx[c]                      # (k,)
        d = knn_dist[c]                       # (k,)

        w = np.exp(-0.5 * (d / max(bandwidth, 1e-6)) ** 2).astype(np.float32)
        in_ball = d <= radius_floor
        w = w * in_ball.astype(np.float32)
        if in_ball.sum() < min_eff_rank:
            continue

        dZ = Z[nbr] - Z[c]                    # (k, L)
        dV = V[nbr] - V[c]                    # (k, L)

        sqrt_w = np.sqrt(w)[:, None]
        Zw = dZ * sqrt_w                      # (k, L)
        Vw = dV * sqrt_w                      # (k, L)

        # Row-wise OLS with ridge: J_c = (Vw.T @ Zw) @ pinv(Zw.T @ Zw + λI).
        ZtZ = Zw.T @ Zw + ridge * np.eye(L, dtype=np.float32)
        rhs = Vw.T @ Zw                       # (L, L)
        try:
            J_c = np.linalg.solve(ZtZ.T, rhs.T).T.astype(np.float32)
        except np.linalg.LinAlgError:
            continue

        J[c] = J_c
        recovered[c] = True
        W_out[c] = w
        pred = dZ @ J_c.T
        residuals[c] = float(np.sqrt((w[:, None] * (dV - pred) ** 2).sum() / (w.sum() + 1e-12)))

    return J, recovered, W_out, residuals


def _smooth_field(
    J: np.ndarray,
    recovered: np.ndarray,
    knn_idx: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    n_iters: int,
) -> np.ndarray:
    """One or more passes of weighted Jacobi smoothing on the operator field.

    ``J_c ← (1 − alpha) J_c + alpha · (Σ w_{cc'} J_{c'}) / Σ w_{cc'}``

    Only recovered cells participate; NaN rows are preserved. This is the
    discrete form of the smoothness penalty in the plan.
    """
    if alpha <= 0.0 or n_iters <= 0:
        return J
    J_out = J.copy()
    for _ in range(n_iters):
        J_new = J_out.copy()
        for c in np.flatnonzero(recovered):
            nbr = knn_idx[c]
            mask = recovered[nbr] & (nbr != c)
            if not mask.any():
                continue
            w = weights[c, mask]
            w_sum = float(w.sum())
            if w_sum <= 1e-12:
                continue
            mean_nbr = (w[:, None, None] * J_out[nbr[mask]]).sum(0) / w_sum
            J_new[c] = (1.0 - alpha) * J_out[c] + alpha * mean_nbr
        J_out = J_new
    return J_out


def _heldout_ll(
    Z: np.ndarray,
    V: np.ndarray,
    k: int,
    bandwidth: float,
    ridge: float,
    hold_frac: float,
    seed: int,
) -> float:
    """Hold out ``hold_frac`` cells, fit J at the rest's neighbours, score
    Gaussian log-likelihood of drift differences on the held-out cells."""
    rng = np.random.default_rng(seed)
    n = Z.shape[0]
    hold = rng.choice(n, size=max(1, int(hold_frac * n)), replace=False)
    train_mask = np.ones(n, dtype=bool)
    train_mask[hold] = False
    Z_train = Z[train_mask]
    V_train = V[train_mask]

    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z_train)
    dists, idx = nn.kneighbors(Z[hold], n_neighbors=k)
    dists = dists.astype(np.float32)
    idx = idx.astype(np.int32)

    ll = 0.0
    L = Z.shape[1]
    for i, c in enumerate(hold):
        d = dists[i]
        w = np.exp(-0.5 * (d / max(bandwidth, 1e-6)) ** 2).astype(np.float32)
        dZ = Z_train[idx[i]] - Z[c]
        dV = V_train[idx[i]] - V[c]
        sqrt_w = np.sqrt(w)[:, None]
        Zw = dZ * sqrt_w
        Vw = dV * sqrt_w
        ZtZ = Zw.T @ Zw + ridge * np.eye(L, dtype=np.float32)
        rhs = Vw.T @ Zw
        try:
            J_c = np.linalg.solve(ZtZ.T, rhs.T).T.astype(np.float32)
        except np.linalg.LinAlgError:
            continue
        pred = dZ @ J_c.T
        resid = dV - pred
        var = (w[:, None] * resid ** 2).sum() / (w.sum() * L + 1e-12)
        var = max(float(var), 1e-6)
        # Gaussian LL per observation per output dim (constant across k, used only for ranking).
        ll += -0.5 * ((w[:, None] * resid ** 2).sum() / var + np.log(var) * w.sum() * L)
    return float(ll)


def build_operator_field(
    Z: np.ndarray,
    V: np.ndarray,
    *,
    k: KSpec = 25,
    k_sweep: Sequence[int] = (15, 25, 50, 100),
    bandwidth: float | Literal["auto"] = "auto",
    ridge: float = 1e-3,
    smoothness_alpha: float = 0.3,
    smoothness_iters: int = 1,
    radius_floor_factor: float = 2.0,
    seed: int = 42,
    verbose: bool = False,
) -> OperatorFieldResult:
    """Compute the operator field from latent positions and per-cell drifts.

    Parameters
    ----------
    Z
        ``(n_cells, L)`` latent positions.
    V
        ``(n_cells, L)`` per-cell drift vectors.
    k
        Neighbourhood size, or ``'auto'`` to sweep ``k_sweep``.
    bandwidth
        Gaussian kernel bandwidth on neighbour distances. ``'auto'`` sets it
        to the median kNN distance at the chosen ``k``.
    ridge
        Ridge regularisation on the per-cell OLS.
    smoothness_alpha
        Jacobi-smoothing mix. ``0`` disables; ``1`` fully replaces each cell
        with its neighbour-average.
    smoothness_iters
        Number of smoothing passes.
    radius_floor_factor
        Cells with fewer than ``L + 1`` neighbours within
        ``radius_floor_factor × median_knn_distance`` are marked unrecovered.
    """
    Z = np.asarray(Z, dtype=np.float32)
    V = np.asarray(V, dtype=np.float32)
    n, L = Z.shape
    assert V.shape == (n, L), f"V shape {V.shape} does not match Z {Z.shape}"

    sweep = None
    if k == "auto":
        sweep = []
        for k_try in k_sweep:
            k_try = int(min(k_try, n - 1))
            med_dist = float(np.median(_knn(Z, k_try)[1][:, -1]))
            bw = med_dist if bandwidth == "auto" else float(bandwidth)
            ll = _heldout_ll(Z, V, k_try, bw, ridge, hold_frac=0.2, seed=seed)
            sweep.append({"k": k_try, "bandwidth": bw, "ll": ll})
            if verbose:
                print(f"[v2.field] k={k_try} bw={bw:.4f} heldout_ll={ll:.2f}")
        best = max(sweep, key=lambda r: r["ll"])
        k_sel = int(best["k"])
        bw_sel = float(best["bandwidth"])
    else:
        k_sel = int(min(int(k), n - 1))
        if bandwidth == "auto":
            med_dist = float(np.median(_knn(Z, k_sel)[1][:, -1]))
            bw_sel = med_dist
        else:
            bw_sel = float(bandwidth)

    knn_idx, knn_dist = _knn(Z, k_sel)
    med_r = float(np.median(knn_dist[:, -1]))
    radius_floor = radius_floor_factor * med_r

    J, recovered, W, residuals = _cellwise_lstsq(
        Z, V, knn_idx, knn_dist,
        bandwidth=bw_sel, ridge=ridge,
        min_eff_rank=L + 1, radius_floor=radius_floor,
    )

    if smoothness_alpha > 0.0 and smoothness_iters > 0:
        J = _smooth_field(J, recovered, knn_idx, W,
                          alpha=smoothness_alpha, n_iters=smoothness_iters)

    return OperatorFieldResult(
        J=J, recovered=recovered,
        k_selected=k_sel,
        neighbors=knn_idx, neighbor_dists=knn_dist,
        weights=W, residuals=residuals,
        k_sweep=sweep,
        tied_lambda=smoothness_alpha * smoothness_iters,
    )
