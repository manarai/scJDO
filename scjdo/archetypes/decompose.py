"""
Temporal Jacobian tensor decomposition via semi-NMF.

Semi-NMF: J(t) ≈ Σ_k a_k(t) · A_k
  - activations a_k(t) ≥ 0  (interpretable: how active is archetype k at time t)
  - patterns    A_k  signed  (Jacobian operators, can have negative entries)

This matches the manuscript's "constrained factorization with non-negativity
on activations". SVD is kept as a fallback via jacobian_modes_svd().
"""
from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Semi-NMF (primary)
# ---------------------------------------------------------------------------

def _seminmf(M: np.ndarray, rank: int, max_iter: int = 500,
             tol: float = 1e-4, seed: int = 0) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Semi-NMF via alternating least squares.
    M (T, D) ≈ W (T, K) @ H (K, D)  where W ≥ 0, H unconstrained.
    """
    rng = np.random.default_rng(seed)
    T, D = M.shape

    W = np.abs(rng.standard_normal((T, rank)).astype(np.float32)) + 1e-4
    W /= W.sum(axis=0, keepdims=True) + 1e-8

    prev_err = np.inf
    for _ in range(max_iter):
        # Update H (unconstrained least squares)
        WtW = W.T @ W + 1e-6 * np.eye(rank)
        H   = np.linalg.solve(WtW, W.T @ M)   # (K, D)

        # Update W (non-negative least squares, column-wise)
        HHt  = H @ H.T + 1e-6 * np.eye(rank)
        W_new = M @ H.T @ np.linalg.inv(HHt)
        W     = np.maximum(W_new, 0.0)

        err = float(np.linalg.norm(M - W @ H, "fro"))
        if abs(prev_err - err) / (prev_err + 1e-8) < tol:
            break
        prev_err = err

    return W, H, prev_err


def jacobian_modes(
    J_tensor: torch.Tensor,
    rank: int = 5,
    n_restarts: int = 5,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Decompose a temporal Jacobian tensor into recurrent operator archetypes.

    Parameters
    ----------
    J_tensor   : (T, D, D) — stacked Jacobians across pseudotime windows.
    rank       : Number of archetypes K.
    n_restarts : ALS restarts (best reconstruction kept).
    seed       : Base random seed.

    Returns
    -------
    patterns    : (K, D, D) — archetype Jacobian matrices (signed).
    activations : (T, K)    — non-negative temporal activation profiles.
    error       : float     — final Frobenius reconstruction error.
    """
    if J_tensor.dim() == 4:
        J_tensor = J_tensor.mean(dim=1)

    T, d1, d2 = J_tensor.shape
    M = J_tensor.reshape(T, d1 * d2).numpy().astype(np.float32)

    best_W, best_H, best_err = None, None, np.inf
    for s in range(n_restarts):
        W, H, err = _seminmf(M, rank=rank, seed=seed + s)
        if err < best_err:
            best_W, best_H, best_err = W, H, err

    patterns    = torch.tensor(best_H, dtype=torch.float32).reshape(rank, d1, d2)
    activations = torch.tensor(best_W, dtype=torch.float32)
    return patterns, activations, best_err


# ---------------------------------------------------------------------------
# SVD fallback (kept for backward compatibility)
# ---------------------------------------------------------------------------

def jacobian_modes_svd(
    J_tensor: torch.Tensor,
    rank: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SVD-based decomposition (activations can be negative). Legacy API."""
    if J_tensor.dim() == 4:
        J_tensor = J_tensor.mean(dim=1)
    T, d1, d2 = J_tensor.shape
    M          = J_tensor.reshape(T, d1 * d2)
    U, S, Vh   = torch.linalg.svd(M, full_matrices=False)
    patterns    = Vh[:rank].reshape(rank, d1, d2)
    activations = U[:, :rank]
    return patterns, activations, S[:rank]
