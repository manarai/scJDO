"""
scqdiff.validation.null_models
================================
Null-model controls for archetype decomposition.

Two null models are provided to address the reviewer concern that
coordination motifs (sequential handoffs, concurrent activation) might
be generic artifacts of low-rank temporal factorization rather than
biologically meaningful structure:

1. **Temporal shuffle null** — shuffles the time-axis of the Jacobian
   tensor before decomposition.  If the motifs are real, they should
   collapse under shuffling.

2. **Continuous-control null** — generates a synthetic Jacobian tensor
   that changes smoothly but contains *no* discrete operator regimes.
   If the decomposition artificially forces discrete archetypes, they
   should appear here too; if it does not, they should not.

Usage example
-------------
>>> from scqdiff.validation.null_models import (
...     temporal_shuffle_null,
...     continuous_control_null,
...     run_null_comparison,
... )
>>> results = run_null_comparison(J_tensor, rank=5, n_shuffles=100)
>>> print(results["summary"])
"""
from __future__ import annotations

import numpy as np
import torch

from scqdiff.archetypes.decompose import jacobian_modes

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _activation_correlation_stats(
    activations: torch.Tensor,
) -> dict[str, float]:
    """
    Compute sequential-handoff and concurrent-activation prevalence
    from a (T, K) activation matrix.

    Returns
    -------
    dict with keys:
        sequential_frac  : fraction of archetype pairs with r < -0.5
        concurrent_frac  : fraction of archetype pairs with r > +0.5
        mean_abs_corr    : mean |r| across all pairs
    """
    act_np = activations.numpy()  # (T, K)
    K = act_np.shape[1]
    if K < 2:
        return {"sequential_frac": 0.0, "concurrent_frac": 0.0, "mean_abs_corr": 0.0}

    corr_matrix = np.corrcoef(act_np.T)  # (K, K)
    upper_idx = np.triu_indices(K, k=1)
    pairwise = corr_matrix[upper_idx]

    n_pairs = len(pairwise)
    sequential_frac = float(np.sum(pairwise < -0.5) / n_pairs)
    concurrent_frac = float(np.sum(pairwise > +0.5) / n_pairs)
    mean_abs_corr = float(np.mean(np.abs(pairwise)))

    return {
        "sequential_frac": sequential_frac,
        "concurrent_frac": concurrent_frac,
        "mean_abs_corr": mean_abs_corr,
    }


# ---------------------------------------------------------------------------
# Null model 1: Temporal shuffle
# ---------------------------------------------------------------------------

def temporal_shuffle_null(
    J_tensor: torch.Tensor,
    rank: int = 5,
    n_shuffles: int = 100,
    seed: int = 0,
) -> dict:
    """
    Temporal shuffle null model for archetype decomposition.

    Randomly permutes the time axis of ``J_tensor`` and re-runs the
    SVD-based decomposition.  Coordination statistics (sequential-handoff
    and concurrent-activation fractions) are collected across ``n_shuffles``
    permutations and compared to the observed statistics on the original
    tensor.

    Parameters
    ----------
    J_tensor : Tensor, shape (T, D, D)
        Observed time-resolved Jacobian tensor.
    rank : int
        Number of archetype components.
    n_shuffles : int
        Number of independent shuffle replicates.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        observed   : dict of observed coordination statistics
        null_mean  : dict of mean null statistics across shuffles
        null_std   : dict of std of null statistics across shuffles
        null_all   : list of per-shuffle dicts
        p_sequential : empirical p-value for sequential_frac
        p_concurrent : empirical p-value for concurrent_frac
        summary    : human-readable string
    """
    rng = np.random.default_rng(seed)

    # Observed statistics
    _, act_obs, _ = jacobian_modes(J_tensor, rank=rank)
    observed = _activation_correlation_stats(act_obs)

    T = J_tensor.shape[0]
    null_all: list[dict] = []

    for _ in range(n_shuffles):
        perm = rng.permutation(T)
        J_shuffled = J_tensor[perm]
        _, act_shuf, _ = jacobian_modes(J_shuffled, rank=rank)
        null_all.append(_activation_correlation_stats(act_shuf))

    null_seq = np.array([d["sequential_frac"] for d in null_all])
    null_con = np.array([d["concurrent_frac"] for d in null_all])
    null_mac = np.array([d["mean_abs_corr"] for d in null_all])

    null_mean = {
        "sequential_frac": float(null_seq.mean()),
        "concurrent_frac": float(null_con.mean()),
        "mean_abs_corr": float(null_mac.mean()),
    }
    null_std = {
        "sequential_frac": float(null_seq.std()),
        "concurrent_frac": float(null_con.std()),
        "mean_abs_corr": float(null_mac.std()),
    }

    # Empirical p-values (one-sided: observed >= null)
    p_seq = float(np.mean(null_seq >= observed["sequential_frac"]))
    p_con = float(np.mean(null_con >= observed["concurrent_frac"]))

    summary = (
        f"Temporal shuffle null ({n_shuffles} replicates)\n"
        f"  Sequential handoff fraction:  observed={observed['sequential_frac']:.3f}  "
        f"null={null_mean['sequential_frac']:.3f}±{null_std['sequential_frac']:.3f}  "
        f"p={p_seq:.4f}\n"
        f"  Concurrent activation fraction: observed={observed['concurrent_frac']:.3f}  "
        f"null={null_mean['concurrent_frac']:.3f}±{null_std['concurrent_frac']:.3f}  "
        f"p={p_con:.4f}"
    )

    return {
        "observed": observed,
        "null_mean": null_mean,
        "null_std": null_std,
        "null_all": null_all,
        "p_sequential": p_seq,
        "p_concurrent": p_con,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Null model 2: Continuous-control (no discrete regimes)
# ---------------------------------------------------------------------------

def continuous_control_null(
    T: int,
    d: int,
    rank: int = 5,
    n_replicates: int = 20,
    seed: int = 0,
) -> dict:
    """
    Continuous-control null model.

    Generates synthetic Jacobian tensors whose entries change smoothly
    over pseudotime but contain *no* discrete operator regimes.  The
    generating process is a random walk in matrix space smoothed by a
    Gaussian kernel — there are no abrupt transitions or discrete
    archetype switches by construction.

    If scIDiff's decomposition artificially forces discrete archetypes,
    they will appear here.  If the motifs are real, they should *not*
    appear (or appear at much lower rates than in real data).

    Parameters
    ----------
    T : int
        Number of pseudotime windows (should match the real tensor).
    d : int
        Latent dimension (should match the real tensor).
    rank : int
        Number of archetype components.
    n_replicates : int
        Number of independent synthetic tensors to generate.
    seed : int
        Random seed.

    Returns
    -------
    dict with keys:
        null_mean  : dict of mean coordination statistics across replicates
        null_std   : dict of std of coordination statistics
        null_all   : list of per-replicate dicts
        summary    : human-readable string
    """
    rng = np.random.default_rng(seed)

    def _smooth_random_tensor(T: int, d: int, sigma: float = 5.0) -> torch.Tensor:
        """Random walk in matrix space, Gaussian-smoothed along time."""
        raw = rng.standard_normal((T, d, d)).astype(np.float32)
        # Gaussian smooth along time axis
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(raw, sigma=sigma, axis=0)
        return torch.from_numpy(smoothed)

    null_all: list[dict] = []
    for _ in range(n_replicates):
        J_null = _smooth_random_tensor(T, d)
        _, act, _ = jacobian_modes(J_null, rank=rank)
        null_all.append(_activation_correlation_stats(act))

    null_seq = np.array([d_["sequential_frac"] for d_ in null_all])
    null_con = np.array([d_["concurrent_frac"] for d_ in null_all])
    null_mac = np.array([d_["mean_abs_corr"] for d_ in null_all])

    null_mean = {
        "sequential_frac": float(null_seq.mean()),
        "concurrent_frac": float(null_con.mean()),
        "mean_abs_corr": float(null_mac.mean()),
    }
    null_std = {
        "sequential_frac": float(null_seq.std()),
        "concurrent_frac": float(null_con.std()),
        "mean_abs_corr": float(null_mac.std()),
    }

    summary = (
        f"Continuous-control null ({n_replicates} replicates, T={T}, d={d})\n"
        f"  Sequential handoff fraction:    {null_mean['sequential_frac']:.3f}±{null_std['sequential_frac']:.3f}\n"
        f"  Concurrent activation fraction: {null_mean['concurrent_frac']:.3f}±{null_std['concurrent_frac']:.3f}\n"
        f"  Mean |correlation|:             {null_mean['mean_abs_corr']:.3f}±{null_std['mean_abs_corr']:.3f}"
    )

    return {
        "null_mean": null_mean,
        "null_std": null_std,
        "null_all": null_all,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def run_null_comparison(
    J_tensor: torch.Tensor,
    rank: int = 5,
    n_shuffles: int = 100,
    n_continuous: int = 20,
    seed: int = 0,
) -> dict:
    """
    Run both null models and return a combined results dict.

    Parameters
    ----------
    J_tensor : Tensor, shape (T, D, D)
        Observed time-resolved Jacobian tensor.
    rank : int
        Number of archetype components.
    n_shuffles : int
        Number of temporal shuffle replicates.
    n_continuous : int
        Number of continuous-control replicates.
    seed : int
        Random seed.

    Returns
    -------
    dict with keys:
        shuffle   : results from temporal_shuffle_null
        continuous: results from continuous_control_null
        summary   : combined human-readable summary
    """
    T, d, _ = J_tensor.shape

    shuffle_results = temporal_shuffle_null(
        J_tensor, rank=rank, n_shuffles=n_shuffles, seed=seed
    )
    continuous_results = continuous_control_null(
        T=T, d=d, rank=rank, n_replicates=n_continuous, seed=seed
    )

    summary = (
        "=" * 60 + "\n"
        "NULL MODEL COMPARISON\n"
        "=" * 60 + "\n"
        + shuffle_results["summary"] + "\n\n"
        + continuous_results["summary"] + "\n"
        "=" * 60
    )

    return {
        "shuffle": shuffle_results,
        "continuous": continuous_results,
        "summary": summary,
    }
