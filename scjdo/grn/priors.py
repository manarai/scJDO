"""
scjdo/grn/priors.py
======================
Utilities for constructing TF masks, sign masks, and optional
motif/ATAC-based prior networks for the HybridGRN extension.

All functions return plain PyTorch tensors so they integrate directly
with SparseGRNRefiner without any external dependencies beyond numpy.

Optional integrations (imported lazily):
    - pybiomart / mygene  : for TF gene list lookup
    - pyranges / pybedtools: for ATAC peak overlap
    - pandas              : for reading CellOracle / SCENIC output tables
"""
from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
import torch


# ---------------------------------------------------------------------------
# TF identification helpers
# ---------------------------------------------------------------------------

# A minimal curated list of well-known human/mouse TFs used as a fallback
# when no external database is available.  This is intentionally small;
# users should supply their own list for serious analyses.
_FALLBACK_TF_SYMBOLS = {
    # Pluripotency
    "POU5F1", "SOX2", "NANOG", "KLF4", "MYC",
    # Haematopoiesis
    "GATA1", "GATA2", "TAL1", "SPI1", "CEBPA", "CEBPB",
    "IRF1", "IRF4", "IRF8", "EBF1", "PAX5",
    # Epithelial / mesenchymal
    "SNAI1", "SNAI2", "TWIST1", "ZEB1", "ZEB2", "CDH1",
    # Generic regulators
    "TP53", "MYB", "FLI1", "RUNX1", "RUNX2", "RUNX3",
    "TCF7L2", "HNF4A", "FOXA1", "FOXA2",
    # Mouse orthologues (lowercase)
    "Pou5f1", "Sox2", "Nanog", "Klf4", "Myc",
    "Gata1", "Gata2", "Tal1", "Spi1", "Cebpa", "Cebpb",
    "Irf1", "Irf4", "Irf8", "Ebf1", "Pax5",
    "Snai1", "Snai2", "Twist1", "Zeb1", "Zeb2",
    "Tp53", "Myb", "Fli1", "Runx1", "Runx2", "Runx3",
    "Tcf7l2", "Hnf4a", "Foxa1", "Foxa2",
}


def identify_tfs(
    gene_names: list[str],
    tf_list: Optional[list[str]] = None,
    use_fallback: bool = True,
) -> tuple[list[str], torch.Tensor]:
    """Identify TF genes and return their indices.

    Parameters
    ----------
    gene_names :
        Full list of gene names in the dataset (length G).
    tf_list :
        User-supplied list of TF gene names.  If None, uses the built-in
        fallback list (suitable for quick tests; not comprehensive).
    use_fallback :
        If True and tf_list is None, use the built-in fallback list.

    Returns
    -------
    tf_names : list[str]
        TF gene names found in gene_names.
    tf_index : (n_tf,) LongTensor
        Indices of TF genes in gene_names.
    """
    if tf_list is None:
        if use_fallback:
            warnings.warn(
                "[scJDO GRN] No TF list provided; using built-in fallback "
                "list of common TFs.  For serious analyses, supply tf_list= "
                "from a comprehensive database (e.g. AnimalTFDB, Lambert 2018).",
                stacklevel=2,
            )
            tf_set = _FALLBACK_TF_SYMBOLS
        else:
            raise ValueError("tf_list must be provided when use_fallback=False.")
    else:
        tf_set = set(tf_list)

    gene_array = np.array(gene_names)
    tf_names = [g for g in gene_names if g in tf_set]
    tf_index = torch.tensor(
        [i for i, g in enumerate(gene_names) if g in tf_set],
        dtype=torch.long,
    )

    if len(tf_names) == 0:
        raise ValueError(
            "No TF genes found in gene_names.  "
            "Check that gene_names uses the same symbol convention as tf_list."
        )

    return tf_names, tf_index


# ---------------------------------------------------------------------------
# TF mask construction
# ---------------------------------------------------------------------------

def build_tf_mask(
    tf_names: list[str],
    gene_names: list[str],
    prior_network: Optional[Union[np.ndarray, torch.Tensor, "pd.DataFrame"]] = None,
    allow_autoregulation: bool = True,
) -> torch.Tensor:
    """Build a (n_tf, G) boolean mask of allowed TF→gene edges.

    Parameters
    ----------
    tf_names :
        TF gene names (subset of gene_names).
    gene_names :
        Full gene list (length G).
    prior_network :
        Optional prior network specifying allowed edges.  Can be:
        - (n_tf, G) numpy array or tensor (1 = allowed, 0 = forbidden)
        - pandas DataFrame with columns ['TF', 'target'] listing allowed edges
        If None, all TF→gene edges are allowed (fully permissive mask).
    allow_autoregulation :
        If True, TF→TF self-edges are allowed even if not in prior_network.

    Returns
    -------
    mask : (n_tf, G) bool tensor
    """
    n_tf = len(tf_names)
    G = len(gene_names)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    tf_to_row = {tf: i for i, tf in enumerate(tf_names)}

    if prior_network is None:
        # Fully permissive: all edges allowed
        mask = torch.ones(n_tf, G, dtype=torch.bool)
    else:
        try:
            import pandas as pd
            if isinstance(prior_network, pd.DataFrame):
                # DataFrame with 'TF' and 'target' columns
                mask = torch.zeros(n_tf, G, dtype=torch.bool)
                for _, row in prior_network.iterrows():
                    tf = row.get("TF") or row.get("source") or row.get("regulator")
                    tgt = row.get("target") or row.get("gene")
                    if tf in tf_to_row and tgt in gene_to_idx:
                        mask[tf_to_row[tf], gene_to_idx[tgt]] = True
                prior_network = None  # already processed
        except ImportError:
            pass

        if prior_network is not None:
            if isinstance(prior_network, np.ndarray):
                prior_network = torch.tensor(prior_network)
            mask = prior_network.bool()
            if mask.shape != (n_tf, G):
                raise ValueError(
                    f"prior_network shape {mask.shape} does not match "
                    f"(n_tf={n_tf}, G={G})."
                )

    # Allow TF→TF autoregulation
    if allow_autoregulation:
        for i, tf in enumerate(tf_names):
            if tf in gene_to_idx:
                mask[i, gene_to_idx[tf]] = True

    return mask


# ---------------------------------------------------------------------------
# Sign mask construction
# ---------------------------------------------------------------------------

def build_sign_mask(
    tf_names: list[str],
    gene_names: list[str],
    activator_tfs: Optional[list[str]] = None,
    repressor_tfs: Optional[list[str]] = None,
) -> torch.Tensor:
    """Build a (n_tf, G) sign mask: +1 activator, -1 repressor, 0 unknown.

    Parameters
    ----------
    tf_names :
        TF gene names.
    gene_names :
        Full gene list.
    activator_tfs :
        TFs known to be activators (all their edges get +1).
    repressor_tfs :
        TFs known to be repressors (all their edges get -1).

    Returns
    -------
    sign_mask : (n_tf, G) int8 tensor
    """
    n_tf = len(tf_names)
    G = len(gene_names)
    sign_mask = torch.zeros(n_tf, G, dtype=torch.int8)

    if activator_tfs:
        act_set = set(activator_tfs)
        for i, tf in enumerate(tf_names):
            if tf in act_set:
                sign_mask[i, :] = 1

    if repressor_tfs:
        rep_set = set(repressor_tfs)
        for i, tf in enumerate(tf_names):
            if tf in rep_set:
                sign_mask[i, :] = -1

    return sign_mask


# ---------------------------------------------------------------------------
# ATAC / motif prior (optional)
# ---------------------------------------------------------------------------

def build_atac_prior(
    tf_names: list[str],
    gene_names: list[str],
    peak_tf_matrix: Union[np.ndarray, torch.Tensor],
    peak_gene_matrix: Union[np.ndarray, torch.Tensor],
    min_peak_score: float = 0.1,
) -> torch.Tensor:
    """Build a (n_tf, G) prior mask from ATAC peak accessibility.

    Constructs a TF→gene prior by linking:
        TF motif scores in peaks  ×  peak-to-gene accessibility

    Parameters
    ----------
    tf_names :
        TF gene names (n_tf).
    gene_names :
        Gene names (G).
    peak_tf_matrix : (n_peaks, n_tf)
        TF motif scores per peak (e.g. from chromVAR or JASPAR scanning).
    peak_gene_matrix : (n_peaks, G)
        Peak-to-gene accessibility scores (e.g. from ArchR or Cicero).
    min_peak_score :
        Minimum score threshold for an edge to be considered active.

    Returns
    -------
    prior : (n_tf, G) float tensor
        Soft prior weights (not boolean).  Pass to SparseGRNRefiner via
        the ``tf_mask`` argument after thresholding if a hard mask is needed.
    """
    if isinstance(peak_tf_matrix, np.ndarray):
        peak_tf_matrix = torch.tensor(peak_tf_matrix, dtype=torch.float32)
    if isinstance(peak_gene_matrix, np.ndarray):
        peak_gene_matrix = torch.tensor(peak_gene_matrix, dtype=torch.float32)

    # prior[tf, g] = sum_peak  peak_tf[peak, tf] * peak_gene[peak, g]
    prior = peak_tf_matrix.T @ peak_gene_matrix   # (n_tf, G)

    # Normalise to [0, 1]
    max_val = prior.max()
    if max_val > 0:
        prior = prior / max_val

    return prior


# ---------------------------------------------------------------------------
# Load CellOracle / SCENIC output as prior
# ---------------------------------------------------------------------------

def load_celloracle_prior(
    path: str,
    tf_names: list[str],
    gene_names: list[str],
) -> torch.Tensor:
    """Load a CellOracle base GRN table as a (n_tf, G) boolean mask.

    Parameters
    ----------
    path :
        Path to CellOracle base GRN CSV (columns: 'source', 'target').
    tf_names :
        TF gene names.
    gene_names :
        Gene names.

    Returns
    -------
    mask : (n_tf, G) bool tensor
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required to load CellOracle priors.")

    df = pd.read_csv(path)
    # Normalise column names
    col_map = {}
    for col in df.columns:
        if col.lower() in ("source", "tf", "regulator"):
            col_map["TF"] = col
        elif col.lower() in ("target", "gene"):
            col_map["target"] = col
    if "TF" not in col_map or "target" not in col_map:
        raise ValueError(
            f"Could not identify TF/target columns in {path}. "
            f"Found: {list(df.columns)}"
        )
    df = df.rename(columns=col_map)
    return build_tf_mask(tf_names, gene_names, prior_network=df)
