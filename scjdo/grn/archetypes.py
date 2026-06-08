"""
scjdo/grn/archetypes.py
==========================
Archetype decomposition of the refined GRN operator sequence K_x.

Design departure from spec
--------------------------
The existing ``scjdo/archetypes/decompose.py`` runs SVD on the raw
latent Jacobian snapshots.  We keep that module unchanged for the
default latent workflow.

This module runs SVD on the *refined sparse GRN operators* K_x (shape
T × n_tf × G), which is downstream of the pullback and refinement steps.
Running archetypes here rather than on J_z has two advantages:

    1. The archetypes are directly interpretable as TF→gene regulatory
       modules (not abstract latent operator patterns).
    2. The sparsity and prior constraints in K_x make the SVD more
       stable and biologically meaningful.

We also add:
    - Mean-centering option (important for signed regulation)
    - Variance-explained reporting
    - Activation peak detection (which pseudotime bin activates each archetype)
    - A ``GRNArchetypeResult`` dataclass for clean downstream use

Signed regulation note
----------------------
The spec correctly warns against using NMF as the default because it
loses sign information.  We use truncated SVD (via torch.linalg.svd)
which preserves signs.  The sign convention is:

    K_x ≈ sum_k  c_k(t) · A_k

where A_k ∈ R^{n_tf × G} is the k-th archetype and c_k(t) is its
time-dependent activation.  Positive c_k means the archetype is
"active" at time t; negative c_k means it is "inverted" (repressor
programme).  Both are biologically meaningful.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class GRNArchetypeResult:
    """Output of ``grn_modes``.

    Attributes
    ----------
    archetypes : (rank, n_tf, G)
        Operator archetypes.  Each slice A_k is a TF→gene regulatory module.
    activations : (T, rank)
        Time-dependent activation coefficients c_k(t).
    singular_values : (rank,)
        Singular values (proportional to variance explained).
    variance_explained : (rank,)
        Fraction of total variance explained by each component.
    peak_times : (rank,)
        Pseudotime index at which each archetype's activation is maximal.
    tf_scores : (rank, n_tf)
        Per-TF importance score for each archetype (row norm of A_k).
    gene_scores : (rank, G)
        Per-gene importance score for each archetype (column norm of A_k).
    """
    archetypes: torch.Tensor          # (rank, n_tf, G)
    activations: torch.Tensor         # (T, rank)
    singular_values: torch.Tensor     # (rank,)
    variance_explained: torch.Tensor  # (rank,)
    peak_times: torch.Tensor          # (rank,)
    tf_scores: torch.Tensor           # (rank, n_tf)
    gene_scores: torch.Tensor         # (rank, G)


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def grn_modes(
    K_tensor: torch.Tensor,
    rank: int = 5,
    center: bool = True,
) -> GRNArchetypeResult:
    """Decompose a sequence of GRN operators into archetypes via truncated SVD.

    Parameters
    ----------
    K_tensor : (T, n_tf, G)
        Time-binned refined GRN operators from SparseGRNRefiner.
    rank :
        Number of archetypes to extract.
    center :
        If True, subtract the temporal mean before SVD.  This is strongly
        recommended for signed regulation: it separates the "baseline"
        regulatory programme from the time-varying components.

    Returns
    -------
    GRNArchetypeResult
        See dataclass definition above.

    Notes
    -----
    The decomposition is:

        K_tensor[t] ≈ K_mean + sum_{k=1}^{rank}  c_k(t) · A_k

    where K_mean is the temporal mean (if center=True, else 0),
    A_k = Vh[k].reshape(n_tf, G) are the archetypes, and
    c_k(t) = U[t, k] * S[k] are the activations.

    The reshaping follows the same convention as the existing
    ``jacobian_modes`` function in ``scjdo/archetypes/decompose.py``.
    """
    T, n_tf, G = K_tensor.shape
    rank = min(rank, T, n_tf * G)

    K = K_tensor.float()

    # ── Mean centering ────────────────────────────────────────────────
    if center:
        K_mean = K.mean(dim=0, keepdim=True)   # (1, n_tf, G)
        K_centered = K - K_mean
    else:
        K_centered = K

    # ── Reshape to (T, n_tf * G) for SVD ─────────────────────────────
    M = K_centered.reshape(T, n_tf * G)   # (T, n_tf * G)

    # ── Truncated SVD ─────────────────────────────────────────────────
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    # U: (T, min(T, n_tf*G)), S: (min(T, n_tf*G),), Vh: (min(T, n_tf*G), n_tf*G)

    U_r = U[:, :rank]                          # (T, rank)
    S_r = S[:rank]                             # (rank,)
    Vh_r = Vh[:rank, :]                        # (rank, n_tf * G)

    # ── Activations: c_k(t) = U[t, k] * S[k] ────────────────────────
    activations = U_r * S_r.unsqueeze(0)       # (T, rank)

    # ── Archetypes: A_k = Vh[k].reshape(n_tf, G) ─────────────────────
    archetypes = Vh_r.reshape(rank, n_tf, G)   # (rank, n_tf, G)

    # ── Variance explained ────────────────────────────────────────────
    total_var = (S ** 2).sum()
    variance_explained = (S_r ** 2) / (total_var + 1e-12)

    # ── Peak activation times ─────────────────────────────────────────
    peak_times = activations.abs().argmax(dim=0)   # (rank,)

    # ── TF and gene importance scores ────────────────────────────────
    # tf_scores[k, tf] = ||A_k[tf, :]||_2  (row norm)
    tf_scores = archetypes.norm(dim=-1)    # (rank, n_tf)
    # gene_scores[k, g] = ||A_k[:, g]||_2  (column norm)
    gene_scores = archetypes.norm(dim=-2)  # (rank, G)

    return GRNArchetypeResult(
        archetypes=archetypes,
        activations=activations,
        singular_values=S_r,
        variance_explained=variance_explained,
        peak_times=peak_times,
        tf_scores=tf_scores,
        gene_scores=gene_scores,
    )


# ---------------------------------------------------------------------------
# Convenience: top TFs and genes per archetype
# ---------------------------------------------------------------------------

def top_tfs_per_archetype(
    result: GRNArchetypeResult,
    tf_names: list[str],
    top_k: int = 10,
) -> list[list[str]]:
    """Return the top-k TFs for each archetype by importance score.

    Parameters
    ----------
    result :
        Output of ``grn_modes``.
    tf_names :
        TF gene names (length n_tf).
    top_k :
        Number of top TFs to return per archetype.

    Returns
    -------
    List of length ``rank``, each element a list of top-k TF names.
    """
    rank = result.tf_scores.shape[0]
    out = []
    for k in range(rank):
        scores = result.tf_scores[k]   # (n_tf,)
        top_idx = scores.argsort(descending=True)[:top_k]
        out.append([tf_names[i] for i in top_idx.tolist()])
    return out


def top_genes_per_archetype(
    result: GRNArchetypeResult,
    gene_names: list[str],
    top_k: int = 20,
) -> list[list[str]]:
    """Return the top-k target genes for each archetype by importance score."""
    rank = result.gene_scores.shape[0]
    out = []
    for k in range(rank):
        scores = result.gene_scores[k]   # (G,)
        top_idx = scores.argsort(descending=True)[:top_k]
        out.append([gene_names[i] for i in top_idx.tolist()])
    return out


def archetype_summary(
    result: GRNArchetypeResult,
    tf_names: list[str],
    gene_names: list[str],
    top_k_tf: int = 5,
    top_k_gene: int = 10,
) -> list[dict]:
    """Return a human-readable summary of each archetype.

    Returns
    -------
    List of dicts, one per archetype, with keys:
        rank, singular_value, variance_explained, peak_time,
        top_tfs, top_genes.
    """
    top_tfs = top_tfs_per_archetype(result, tf_names, top_k_tf)
    top_genes = top_genes_per_archetype(result, gene_names, top_k_gene)
    summaries = []
    for k in range(result.archetypes.shape[0]):
        summaries.append({
            "rank": k + 1,
            "singular_value": result.singular_values[k].item(),
            "variance_explained": result.variance_explained[k].item(),
            "peak_time_bin": result.peak_times[k].item(),
            "top_tfs": top_tfs[k],
            "top_genes": top_genes[k],
        })
    return summaries
