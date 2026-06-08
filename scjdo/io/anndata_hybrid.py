"""
scjdo/io/anndata_hybrid.py
==============================
AnnData I/O utilities for the HybridGRN extension.

This module extends the existing ``scjdo/io/anndata.py`` with gene-panel
and TF-aware data loading.  It is kept separate to avoid modifying the
existing stable API.

The key addition is ``tensors_from_anndata_hybrid``, which returns both
the latent representation (for the drift model) and the full gene expression
matrix (for the pullback and GRN extraction).
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import torch
import anndata as ad

from scjdo.io.anndata import tensors_from_anndata


# ---------------------------------------------------------------------------
# Hybrid data loader
# ---------------------------------------------------------------------------

def tensors_from_anndata_hybrid(
    adata: ad.AnnData,
    use_rep: str = "X_pca",
    gene_layer: str = "X",
    vel_layer: Optional[str] = "velocity",
    pseudotime_key: Optional[str] = "pseudotime",
    tf_names: Optional[list[str]] = None,
    gene_names: Optional[list[str]] = None,
    normalize_pseudotime: bool = False,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Load all tensors needed for the HybridGRN pipeline.

    This function returns both the latent representation (for the drift
    model) and the full gene expression matrix (for the pullback and
    GRN extraction).

    Parameters
    ----------
    adata :
        AnnData object.
    use_rep :
        Key in ``adata.obsm`` for the latent representation (default: X_pca).
        Must match the dimensionality of the trained DriftField.
    gene_layer :
        Layer key for gene expression.  Use ``"X"`` for ``adata.X``.
        The matrix should be log1p-normalised counts.
    vel_layer :
        Layer key for RNA velocity.  Used for the local dynamics loss.
    pseudotime_key :
        Column in ``adata.obs`` for pseudotime (must be in [0, 1]).
    tf_names :
        List of TF gene names to extract.  If None, all genes are returned
        and TF identification is left to the caller.
    gene_names :
        Subset of gene names to use.  If None, all genes in adata.var_names
        are used.
    normalize_pseudotime :
        If True, rescale pseudotime to [0, 1].
    device :
        PyTorch device string.

    Returns
    -------
    dict with keys:
        ``z_init``      : (N, D) latent representation tensor.
        ``x_gene``      : (N, G) gene expression tensor.
        ``velocity``    : (N, D_or_G) velocity tensor, or None.
        ``pseudotime``  : (N,) pseudotime tensor.
        ``tf_index``    : (n_tf,) LongTensor of TF gene indices, or None.
        ``gene_index``  : (G,) LongTensor of selected gene indices.
        ``gene_names``  : list of selected gene names.
        ``tf_names``    : list of TF gene names found.
    """
    # ── Latent representation, velocity, pseudotime ───────────────────
    z_init, velocity, pseudotime = tensors_from_anndata(
        adata,
        use_rep=use_rep,
        vel_layer=vel_layer,
        pseudotime_key=pseudotime_key,
        normalize_pseudotime=normalize_pseudotime,
        device=device,
    )

    # ── Gene expression ───────────────────────────────────────────────
    all_gene_names = list(adata.var_names)

    if gene_names is not None:
        # Subset to requested genes
        gene_set = set(gene_names)
        gene_index_list = [i for i, g in enumerate(all_gene_names) if g in gene_set]
        selected_gene_names = [all_gene_names[i] for i in gene_index_list]
        if len(selected_gene_names) == 0:
            raise ValueError(
                "None of the requested gene_names were found in adata.var_names."
            )
        if len(selected_gene_names) < len(gene_names):
            missing = set(gene_names) - set(selected_gene_names)
            warnings.warn(
                f"[tensors_from_anndata_hybrid] {len(missing)} requested genes "
                f"not found in adata.var_names: {list(missing)[:5]}...",
                stacklevel=2,
            )
    else:
        gene_index_list = list(range(len(all_gene_names)))
        selected_gene_names = all_gene_names

    gene_index = torch.tensor(gene_index_list, dtype=torch.long)

    # Extract gene expression
    if gene_layer == "X":
        raw = adata.X
    elif gene_layer in adata.layers:
        raw = adata.layers[gene_layer]
    else:
        raise KeyError(
            f"Gene layer '{gene_layer}' not found. "
            f"Available layers: {list(adata.layers.keys())}"
        )

    if hasattr(raw, "toarray"):
        raw = raw.toarray()
    x_gene_full = np.array(raw, dtype=np.float32)
    x_gene = torch.tensor(x_gene_full[:, gene_index_list], device=device)

    # ── TF identification ─────────────────────────────────────────────
    found_tf_names = None
    tf_index = None

    if tf_names is not None:
        tf_set = set(tf_names)
        tf_index_list = [
            i for i, g in enumerate(selected_gene_names) if g in tf_set
        ]
        found_tf_names = [selected_gene_names[i] for i in tf_index_list]

        if len(found_tf_names) == 0:
            warnings.warn(
                "[tensors_from_anndata_hybrid] No TF names found in the selected "
                "gene panel.  Check that tf_names uses the same symbol convention "
                "as adata.var_names.",
                stacklevel=2,
            )
        else:
            tf_index = torch.tensor(tf_index_list, dtype=torch.long)

    return {
        "z_init": z_init,
        "x_gene": x_gene,
        "velocity": velocity,
        "pseudotime": pseudotime,
        "tf_index": tf_index,
        "gene_index": gene_index,
        "gene_names": selected_gene_names,
        "tf_names": found_tf_names,
    }


# ---------------------------------------------------------------------------
# Pseudotime-bin mean computation
# ---------------------------------------------------------------------------

def compute_bin_means(
    x_gene: torch.Tensor,
    pseudotime: torch.Tensor,
    n_bins: int = 20,
    tf_index: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """Compute mean gene expression and expression changes per pseudotime bin.

    Used to provide ``x_tf_seq`` and ``dx_seq`` to the SparseGRNRefiner.

    Parameters
    ----------
    x_gene : (N, G)
    pseudotime : (N,)
    n_bins :
        Number of pseudotime bins.
    tf_index : (n_tf,) optional
        TF gene indices.  If provided, also returns ``x_tf_seq``.

    Returns
    -------
    dict with keys:
        ``x_mean``    : (T, G) mean expression per bin.
        ``dx_mean``   : (T, G) mean expression change per bin (x_{t+1} - x_t).
        ``x_tf_seq``  : (T, n_tf) mean TF expression per bin (if tf_index given).
        ``bin_edges`` : (T+1,) pseudotime bin edges.
    """
    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = torch.bucketize(pseudotime, bin_edges[1:-1])

    G = x_gene.shape[1]
    x_mean = torch.zeros(n_bins, G)
    bin_counts = torch.zeros(n_bins, dtype=torch.long)

    for b in range(n_bins):
        mask = (bin_ids == b)
        if mask.any():
            x_mean[b] = x_gene[mask].mean(0)
            bin_counts[b] = mask.sum()

    # Expression change: forward difference between bin means
    dx_mean = torch.zeros(n_bins, G)
    for b in range(n_bins - 1):
        if bin_counts[b] > 0 and bin_counts[b + 1] > 0:
            dx_mean[b] = x_mean[b + 1] - x_mean[b]
    # Last bin: backward difference
    if n_bins >= 2 and bin_counts[-1] > 0 and bin_counts[-2] > 0:
        dx_mean[-1] = x_mean[-1] - x_mean[-2]

    result = {
        "x_mean": x_mean,
        "dx_mean": dx_mean,
        "bin_edges": bin_edges,
    }

    if tf_index is not None:
        result["x_tf_seq"] = x_mean[:, tf_index]

    return result
