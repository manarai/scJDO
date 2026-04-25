"""
scqdiff.validation.robustness
================================
Gene-level robustness analysis across pseudotime methods.

Addresses reviewer concern (Review 2, point 4) that DTW-aligned
archetype-level correlations are insufficient evidence of robustness.
The real question is whether the *gene-level* conclusions — specifically,
which genes are ranked as "unstable" — remain stable across different
pseudotime algorithms (DPT, Palantir, Slingshot).

This module provides:

- ``gene_overlap_across_pseudotimes``: given a list of per-method
  unstable-gene rankings, computes pairwise Jaccard indices and rank
  correlations for top-K gene lists.

- ``pseudotime_sensitivity_report``: a convenience wrapper that produces
  a formatted summary table suitable for inclusion in the manuscript's
  robustness section.

Usage example
-------------
>>> from scqdiff.validation.robustness import gene_overlap_across_pseudotimes
>>> # gene_rankings: dict mapping method name -> array of gene names sorted
>>> #                by unstable-mode loading (highest first)
>>> results = gene_overlap_across_pseudotimes(
...     gene_rankings={"DPT": dpt_genes, "Palantir": pal_genes, "Slingshot": sl_genes},
...     top_k_values=[50, 100, 200],
... )
>>> print(results["summary"])
"""
from __future__ import annotations

from itertools import combinations
from typing import Optional

import numpy as np
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Core metric: Jaccard index
# ---------------------------------------------------------------------------

def jaccard(set_a: set, set_b: set) -> float:
    """Jaccard index between two sets.  Returns 0.0 if both are empty."""
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


# ---------------------------------------------------------------------------
# Gene-level overlap across pseudotime methods
# ---------------------------------------------------------------------------

def gene_overlap_across_pseudotimes(
    gene_rankings: dict[str, list[str]],
    top_k_values: list[int] | None = None,
    score_arrays: dict[str, np.ndarray] | None = None,
) -> dict:
    """
    Compute pairwise gene-level overlap across pseudotime methods.

    Parameters
    ----------
    gene_rankings : dict[str, list[str]]
        Mapping from pseudotime method name to a list of gene names
        sorted by unstable-mode loading (highest first).
    top_k_values : list[int], optional
        Top-K thresholds at which to compute Jaccard overlap.
        Defaults to [50, 100, 200].
    score_arrays : dict[str, np.ndarray], optional
        If provided, mapping from method name to a numeric score array
        (same order as ``gene_rankings``).  Used to compute Spearman
        rank correlation in addition to Jaccard.

    Returns
    -------
    dict with keys:
        jaccard_by_k   : dict[int, dict[str, float]]
                         Jaccard index for each top-K and each method pair.
        spearman_by_pair : dict[str, float] (only if score_arrays given)
                         Spearman r for each method pair over all genes.
        mean_jaccard   : dict[int, float]
                         Mean Jaccard across all pairs for each top-K.
        summary        : human-readable table string
    """
    if top_k_values is None:
        top_k_values = [50, 100, 200]

    methods = list(gene_rankings.keys())
    pairs = list(combinations(methods, 2))

    jaccard_by_k: dict[int, dict[str, float]] = {}
    for k in top_k_values:
        jaccard_by_k[k] = {}
        for m1, m2 in pairs:
            genes1 = set(gene_rankings[m1][:k])
            genes2 = set(gene_rankings[m2][:k])
            key = f"{m1} vs {m2}"
            jaccard_by_k[k][key] = jaccard(genes1, genes2)

    mean_jaccard = {
        k: float(np.mean(list(jaccard_by_k[k].values())))
        for k in top_k_values
    }

    # Spearman rank correlation (optional)
    spearman_by_pair: dict[str, float] = {}
    if score_arrays is not None:
        for m1, m2 in pairs:
            key = f"{m1} vs {m2}"
            r, _ = spearmanr(score_arrays[m1], score_arrays[m2])
            spearman_by_pair[key] = float(r)

    # Build summary table
    lines = ["Gene-level robustness across pseudotime methods", "=" * 60]
    header = f"{'Pair':<30}" + "".join(f"  Jaccard@{k:<5}" for k in top_k_values)
    if spearman_by_pair:
        header += "  Spearman r"
    lines.append(header)
    lines.append("-" * len(header))

    for m1, m2 in pairs:
        key = f"{m1} vs {m2}"
        row = f"{key:<30}"
        for k in top_k_values:
            row += f"  {jaccard_by_k[k][key]:.3f}      "
        if spearman_by_pair:
            row += f"  {spearman_by_pair[key]:.3f}"
        lines.append(row)

    lines.append("-" * len(header))
    mean_row = f"{'Mean':<30}"
    for k in top_k_values:
        mean_row += f"  {mean_jaccard[k]:.3f}      "
    lines.append(mean_row)

    summary = "\n".join(lines)

    return {
        "jaccard_by_k": jaccard_by_k,
        "spearman_by_pair": spearman_by_pair,
        "mean_jaccard": mean_jaccard,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Convenience: full pseudotime sensitivity report
# ---------------------------------------------------------------------------

def pseudotime_sensitivity_report(
    gene_rankings: dict[str, list[str]],
    top_k_values: list[int] | None = None,
    score_arrays: dict[str, np.ndarray] | None = None,
) -> str:
    """
    Return a formatted sensitivity report string for use in manuscript
    supplementary materials or robustness section.

    Parameters
    ----------
    gene_rankings : dict[str, list[str]]
        Per-method gene rankings (see ``gene_overlap_across_pseudotimes``).
    top_k_values : list[int], optional
        Top-K thresholds.  Defaults to [50, 100, 200].
    score_arrays : dict[str, np.ndarray], optional
        Per-method numeric score arrays for Spearman correlation.

    Returns
    -------
    str
        Formatted report.
    """
    results = gene_overlap_across_pseudotimes(
        gene_rankings=gene_rankings,
        top_k_values=top_k_values,
        score_arrays=score_arrays,
    )
    return results["summary"]
