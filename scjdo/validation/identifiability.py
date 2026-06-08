"""
scjdo.validation.identifiability
=====================================
Utilities for quantifying what is and is not identifiable in the
scJDO operator framework.

Background
----------
scJDO infers time-varying Jacobians as derivatives of a learned
drift field.  Because the drift field is estimated from static
snapshot data, the Jacobians are *not* uniquely identifiable in an
absolute sense: many different drift fields can produce similar
observed distributions but different Jacobians.

However, certain *relative* and *structural* properties of the
inferred operator space are empirically robust across model choices,
embeddings, and seeds.  This module provides tools to quantify those
properties and to test how sensitive conclusions are to model-class
variation (architecture, regularization, noise level).

Invariant properties (empirically):
    - Relative reuse of operator regimes (archetype activation ordering)
    - Temporal ordering of instability peaks
    - Low-rank structure of Jacobian evolution (variance explained by K)

Non-invariant properties (explicitly acknowledged):
    - Absolute eigenvalues
    - Exact Jacobian matrix entries
    - Gene-level causal interpretation
    - Conclusions under nonlinear reparameterizations of latent space

Usage example
-------------
>>> from scjdo.validation.identifiability import (
...     archetype_cosine_similarity,
...     instability_peak_overlap,
...     model_sensitivity_report,
... )
>>> sim = archetype_cosine_similarity(archetypes_run1, archetypes_run2)
>>> print(f"Median cosine similarity: {sim['median']:.3f}")
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Archetype identity across runs / model choices
# ---------------------------------------------------------------------------

def archetype_cosine_similarity(
    archetypes_a: torch.Tensor,
    archetypes_b: torch.Tensor,
) -> dict:
    """
    Compute cosine similarity between matched archetype pairs from two runs.

    Archetypes are matched greedily by maximum cosine similarity (Hungarian
    matching is not required because K is small and similarity is typically
    unambiguous).

    Parameters
    ----------
    archetypes_a : Tensor, shape (K, D, D)
        Archetype matrices from run A.
    archetypes_b : Tensor, shape (K, D, D)
        Archetype matrices from run B.

    Returns
    -------
    dict with keys:
        per_archetype : list[float]  cosine similarity for each matched pair
        median        : float        median across all pairs
        min           : float        minimum (worst-case pair)
        summary       : str
    """
    Ka = archetypes_a.shape[0]
    Kb = archetypes_b.shape[0]
    K = min(Ka, Kb)

    # Flatten to vectors
    flat_a = archetypes_a[:Ka].reshape(Ka, -1).float()
    flat_b = archetypes_b[:Kb].reshape(Kb, -1).float()

    # Normalise
    norm_a = flat_a / (flat_a.norm(dim=1, keepdim=True) + 1e-8)
    norm_b = flat_b / (flat_b.norm(dim=1, keepdim=True) + 1e-8)

    # Similarity matrix (Ka x Kb)
    sim_matrix = (norm_a @ norm_b.T).numpy()

    # Greedy matching
    matched: list[float] = []
    used_b: set[int] = set()
    for i in range(K):
        row = sim_matrix[i].copy()
        for j in used_b:
            row[j] = -np.inf
        best_j = int(np.argmax(row))
        matched.append(float(sim_matrix[i, best_j]))
        used_b.add(best_j)

    median_sim = float(np.median(matched))
    min_sim = float(np.min(matched))

    summary = (
        f"Archetype cosine similarity (K={K})\n"
        f"  Per-archetype: {[f'{v:.3f}' for v in matched]}\n"
        f"  Median: {median_sim:.3f}  Min: {min_sim:.3f}"
    )

    return {
        "per_archetype": matched,
        "median": median_sim,
        "min": min_sim,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Instability peak timing across runs
# ---------------------------------------------------------------------------

def instability_peak_overlap(
    instability_curves: list[np.ndarray],
    peak_window: float = 0.1,
) -> dict:
    """
    Measure how consistently instability peaks are located across runs.

    Parameters
    ----------
    instability_curves : list of 1-D arrays
        Each array is a normalised instability curve over pseudotime [0, 1].
        All arrays should have the same length T.
    peak_window : float
        Two peaks are considered "overlapping" if they are within this
        fraction of pseudotime of each other.

    Returns
    -------
    dict with keys:
        peak_locations : list[float]  pseudotime of peak per run
        peak_std       : float        std of peak locations
        overlap_frac   : float        fraction of run pairs whose peaks overlap
        summary        : str
    """
    peak_locs = []
    for curve in instability_curves:
        t = np.linspace(0, 1, len(curve))
        peak_locs.append(float(t[np.argmax(curve)]))

    peak_std = float(np.std(peak_locs))

    n = len(peak_locs)
    n_pairs = n * (n - 1) // 2
    overlap_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(peak_locs[i] - peak_locs[j]) <= peak_window:
                overlap_count += 1
    overlap_frac = overlap_count / max(n_pairs, 1)

    summary = (
        f"Instability peak timing ({len(instability_curves)} runs)\n"
        f"  Peak locations: {[f'{p:.3f}' for p in peak_locs]}\n"
        f"  Std of peak locations: {peak_std:.4f}\n"
        f"  Fraction of pairs within window={peak_window}: {overlap_frac:.3f}"
    )

    return {
        "peak_locations": peak_locs,
        "peak_std": peak_std,
        "overlap_frac": overlap_frac,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Model-sensitivity report
# ---------------------------------------------------------------------------

def model_sensitivity_report(
    results_by_config: dict[str, dict],
) -> str:
    """
    Produce a formatted sensitivity report across model configurations.

    Parameters
    ----------
    results_by_config : dict[str, dict]
        Mapping from configuration label (e.g., "depth=4, hidden=256") to
        a dict containing at minimum:
            - "archetypes"         : Tensor (K, D, D)
            - "instability_curve"  : np.ndarray (T,)
            - "auroc"              : float (optional)

    Returns
    -------
    str
        Formatted report comparing archetype similarity and instability
        peak timing across all configurations.
    """
    configs = list(results_by_config.keys())
    lines = [
        "Model sensitivity report",
        "=" * 60,
        "Invariant properties tested:",
        "  1. Archetype cosine similarity across configurations",
        "  2. Instability peak timing across configurations",
        "",
        "Non-invariant (explicitly not claimed):",
        "  - Absolute eigenvalues",
        "  - Exact Jacobian entries",
        "  - Gene-level causal interpretation",
        "=" * 60,
    ]

    # Archetype similarity: compare each config to the first
    ref_label = configs[0]
    ref_arch = results_by_config[ref_label]["archetypes"]
    lines.append(f"\nArchetype similarity vs reference ({ref_label}):")
    for label in configs[1:]:
        arch = results_by_config[label]["archetypes"]
        sim = archetype_cosine_similarity(ref_arch, arch)
        lines.append(f"  {label}: median cosine = {sim['median']:.3f}, min = {sim['min']:.3f}")

    # Instability peak timing
    curves = [results_by_config[c]["instability_curve"] for c in configs]
    peak_info = instability_peak_overlap(curves)
    lines.append("\nInstability peak timing:")
    for label, loc in zip(configs, peak_info["peak_locations"]):
        lines.append(f"  {label}: peak at pseudotime = {loc:.3f}")
    lines.append(f"  Std across configs: {peak_info['peak_std']:.4f}")
    lines.append(f"  Fraction of pairs within 0.10: {peak_info['overlap_frac']:.3f}")

    # AUROC if available
    aurocs = {
        c: results_by_config[c]["auroc"]
        for c in configs
        if "auroc" in results_by_config[c]
    }
    if aurocs:
        lines.append("\nAUROC across configurations:")
        for label, auroc in aurocs.items():
            lines.append(f"  {label}: {auroc:.4f}")
        vals = list(aurocs.values())
        lines.append(
            f"  Range: {min(vals):.4f} – {max(vals):.4f}  "
            f"(ΔAUROC = {max(vals) - min(vals):.4f})"
        )

    return "\n".join(lines)
