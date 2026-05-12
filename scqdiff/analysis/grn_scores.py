"""
scqdiff/analysis/grn_scores.py
================================
High-level summary statistics for the HybridGRN extension.

These functions aggregate the per-bin or per-cell GRN operators into
interpretable scores that can be stored back into AnnData and used for
downstream biological analysis.

Functions
---------
regulator_centrality     : Out-degree and in-degree centrality per TF/gene.
branch_control_score     : Which TFs are most active near fate branch points.
temporal_activity_score  : How each TF's regulatory activity changes over time.
grn_to_anndata           : Store GRN results back into AnnData.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Regulator centrality
# ---------------------------------------------------------------------------

def regulator_centrality(
    Kx: torch.Tensor,
    tf_names: list[str],
    gene_names: list[str],
    aggregate: str = "mean",
) -> dict[str, torch.Tensor]:
    """Compute out-degree and in-degree centrality for TFs and genes.

    Parameters
    ----------
    Kx : (T, n_tf, G)
        Time-binned refined GRN operators.
    tf_names :
        TF gene names (length n_tf).
    gene_names :
        Gene names (length G).
    aggregate :
        How to aggregate over time bins: ``"mean"``, ``"max"``, or ``"sum"``.

    Returns
    -------
    dict with keys:
        ``tf_out_degree``   : (n_tf,) — mean absolute out-degree per TF.
        ``gene_in_degree``  : (G,)    — mean absolute in-degree per gene.
        ``tf_net_effect``   : (n_tf,) — mean signed net effect per TF.
        ``gene_net_input``  : (G,)    — mean signed net input per gene.
    """
    # Aggregate over time
    if aggregate == "mean":
        K_agg = Kx.mean(dim=0)   # (n_tf, G)
    elif aggregate == "max":
        K_agg = Kx.abs().max(dim=0).values
    elif aggregate == "sum":
        K_agg = Kx.sum(dim=0)
    else:
        raise ValueError(f"Unknown aggregate='{aggregate}'.")

    tf_out_degree = K_agg.abs().sum(dim=-1)    # (n_tf,)  — total regulatory output
    gene_in_degree = K_agg.abs().sum(dim=0)    # (G,)     — total regulatory input
    tf_net_effect = K_agg.sum(dim=-1)          # (n_tf,)  — net signed output
    gene_net_input = K_agg.sum(dim=0)          # (G,)     — net signed input

    return {
        "tf_out_degree": tf_out_degree,
        "gene_in_degree": gene_in_degree,
        "tf_net_effect": tf_net_effect,
        "gene_net_input": gene_net_input,
    }


# ---------------------------------------------------------------------------
# Branch control score
# ---------------------------------------------------------------------------

def branch_control_score(
    Kx: torch.Tensor,
    instability_profile: torch.Tensor,
    tf_names: list[str],
) -> torch.Tensor:
    """Identify TFs most active near fate branch points.

    Weights each TF's regulatory activity by the instability profile
    (e.g. max positive eigenvalue of J_z from scOpAtlas) to identify
    TFs that are most active precisely when the cell is most plastic.

    Parameters
    ----------
    Kx : (T, n_tf, G)
        Time-binned refined GRN operators.
    instability_profile : (T,)
        Instability score per time bin (e.g. λ_max⁺ from scOpAtlas).
        Higher values indicate more plastic / bifurcating states.
    tf_names :
        TF gene names.

    Returns
    -------
    branch_scores : (n_tf,)
        Weighted regulatory activity score per TF.
        TFs with high scores are most active near branch points.
    """
    # TF activity at each time bin: ||K_x[t, tf, :]||_2
    tf_activity = Kx.norm(dim=-1)   # (T, n_tf)

    # Weight by instability
    weights = instability_profile.float()
    weights = weights / (weights.sum() + 1e-12)   # normalise

    branch_scores = (tf_activity * weights.unsqueeze(-1)).sum(dim=0)   # (n_tf,)
    return branch_scores


# ---------------------------------------------------------------------------
# Temporal activity score
# ---------------------------------------------------------------------------

def temporal_activity_score(
    Kx: torch.Tensor,
    tf_names: list[str],
    smooth: bool = True,
    smooth_sigma: float = 1.5,
) -> torch.Tensor:
    """Compute how each TF's regulatory activity changes over pseudotime.

    Parameters
    ----------
    Kx : (T, n_tf, G)
    tf_names :
        TF gene names.
    smooth :
        If True, apply Gaussian smoothing over the time axis.
    smooth_sigma :
        Standard deviation for Gaussian smoothing (in bins).

    Returns
    -------
    activity : (T, n_tf)
        Regulatory activity (L2 norm of K_x row) per TF per time bin.
    """
    activity = Kx.norm(dim=-1)   # (T, n_tf)

    if smooth and activity.shape[0] > 3:
        activity = _gaussian_smooth_1d(activity, smooth_sigma)

    return activity


def _gaussian_smooth_1d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply 1D Gaussian smoothing along the first axis (time)."""
    T = x.shape[0]
    kernel_size = max(3, int(4 * sigma) | 1)  # ensure odd
    half = kernel_size // 2
    t = torch.arange(-half, half + 1, dtype=x.dtype, device=x.device)
    kernel = torch.exp(-0.5 * (t / sigma) ** 2)
    kernel = kernel / kernel.sum()

    # Pad and convolve
    x_t = x.T.unsqueeze(1)   # (n_tf, 1, T)
    kernel_2d = kernel.unsqueeze(0).unsqueeze(0)   # (1, 1, kernel_size)
    x_padded = torch.nn.functional.pad(x_t, (half, half), mode="reflect")
    smoothed = torch.nn.functional.conv1d(x_padded, kernel_2d)   # (n_tf, 1, T)
    return smoothed.squeeze(1).T   # (T, n_tf)


# ---------------------------------------------------------------------------
# Store results into AnnData
# ---------------------------------------------------------------------------

def grn_to_anndata(
    adata,
    Kx: torch.Tensor,
    tf_names: list[str],
    gene_names: list[str],
    archetypes=None,
    prefix: str = "scqdiff_grn",
) -> None:
    """Store GRN results back into an AnnData object.

    Stores:
        adata.uns[f'{prefix}_Kx']          : (T, n_tf, G) numpy array
        adata.uns[f'{prefix}_tf_names']     : list of TF names
        adata.uns[f'{prefix}_gene_names']   : list of gene names
        adata.uns[f'{prefix}_archetypes']   : archetype summary (if provided)
        adata.var[f'{prefix}_in_degree']    : gene in-degree (if gene_names match)
        adata.obs[f'{prefix}_tf_activity']  : per-cell TF activity (if possible)

    Parameters
    ----------
    adata :
        AnnData object.
    Kx : (T, n_tf, G)
        Refined GRN operators.
    tf_names :
        TF gene names.
    gene_names :
        Gene names.
    archetypes :
        Optional GRNArchetypeResult from grn_modes().
    prefix :
        Key prefix for stored results.
    """
    import numpy as np

    adata.uns[f"{prefix}_Kx"] = Kx.cpu().numpy()
    adata.uns[f"{prefix}_tf_names"] = list(tf_names)
    adata.uns[f"{prefix}_gene_names"] = list(gene_names)

    # Centrality scores
    centrality = regulator_centrality(Kx, tf_names, gene_names)
    adata.uns[f"{prefix}_tf_out_degree"] = centrality["tf_out_degree"].cpu().numpy()
    adata.uns[f"{prefix}_gene_in_degree"] = centrality["gene_in_degree"].cpu().numpy()

    # Store gene in-degree in adata.var if gene names match
    try:
        if hasattr(adata, "var") and list(adata.var_names) == list(gene_names):
            adata.var[f"{prefix}_in_degree"] = (
                centrality["gene_in_degree"].cpu().numpy()
            )
    except Exception:
        pass

    # Store archetypes
    if archetypes is not None:
        from scqdiff.grn.archetypes import archetype_summary
        adata.uns[f"{prefix}_archetypes"] = {
            "archetypes": archetypes.archetypes.cpu().numpy(),
            "activations": archetypes.activations.cpu().numpy(),
            "singular_values": archetypes.singular_values.cpu().numpy(),
            "variance_explained": archetypes.variance_explained.cpu().numpy(),
            "peak_times": archetypes.peak_times.cpu().numpy(),
        }
