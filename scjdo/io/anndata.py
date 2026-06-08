"""
======================
AnnData ↔ tensor utilities for the scjdo pipeline.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import torch
import anndata as ad


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def tensors_from_anndata(
    adata: ad.AnnData,
    use_rep: str = "X_pca",
    vel_layer: Optional[str] = "velocity",
    pseudotime_key: Optional[str] = "pseudotime",
    normalize_pseudotime: bool = False,
    device: str = "cpu",
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Extract expression (X), velocity (V), and pseudotime (T) tensors from
    an AnnData object.

    Parameters
    ----------
    adata               : AnnData object (cells × genes or cells × PCs).
    use_rep             : Key in ``adata.obsm`` for cell representation,
                          or ``"X"`` to use ``adata.X`` directly.
    vel_layer           : Layer key for RNA velocity; ``None`` to skip.
    pseudotime_key      : Column in ``adata.obs`` for pseudotime;
                          ``None`` to skip.
    normalize_pseudotime: If ``True``, linearly rescale pseudotime to [0, 1]
                          and emit a warning.  If ``False`` (default), raise
                          a ``ValueError`` when pseudotime is outside [0, 1].
    device              : PyTorch device string.

    Returns
    -------
    X : (N, D) cell representation tensor.
    V : (N, D) velocity tensor, or ``None``.
    T : (N,)   pseudotime tensor in [0, 1], or ``None``.

    Notes
    -----
    **Pseudotime must be in [0, 1].** The drift field uses g(t) = 4t(1-t)
    as a time-dependent gate, which is only meaningful when t ∈ [0, 1].
    Raw pseudotime values from DPT, Monocle, etc. are typically *not*
    normalised.  Normalise before calling this function or pass
    ``normalize_pseudotime=True``.

    Example
    -------
    >>> # Normalise manually (recommended — gives you control)
    >>> adata.obs['pseudotime_norm'] = (
    ...     adata.obs['dpt_pseudotime'] - adata.obs['dpt_pseudotime'].min()
    ... ) / (adata.obs['dpt_pseudotime'].max() - adata.obs['dpt_pseudotime'].min())
    >>> X, V, T = tensors_from_anndata(adata, pseudotime_key='pseudotime_norm')

    >>> # Or let the function normalise for you (emits a warning)
    >>> X, V, T = tensors_from_anndata(
    ...     adata, pseudotime_key='dpt_pseudotime', normalize_pseudotime=True
    ... )
    """
    # ── Cell representation ────────────────────────────────────────────────
    if use_rep == "X":
        raw = adata.X
    elif use_rep in adata.obsm:
        raw = adata.obsm[use_rep]
    else:
        raise KeyError(
            f"Representation '{use_rep}' not found in adata.obsm. "
            f"Available keys: {list(adata.obsm.keys())}"
        )

    # Handle scipy sparse
    if hasattr(raw, "toarray"):
        raw = raw.toarray()
    X = torch.tensor(np.array(raw, dtype=np.float32), device=device)

    # ── Velocity ───────────────────────────────────────────────────────────
    V: Optional[torch.Tensor] = None
    if vel_layer is not None:
        if vel_layer not in adata.layers:
            warnings.warn(
                f"[tensors_from_anndata] Velocity layer '{vel_layer}' not found "
                f"in adata.layers {list(adata.layers.keys())}. "
                f"Continuing without velocity prior.",
                stacklevel=2,
            )
        else:
            vel_raw = adata.layers[vel_layer]
            if hasattr(vel_raw, "toarray"):
                vel_raw = vel_raw.toarray()
            V_full = np.array(vel_raw, dtype=np.float32)

            # If velocity is in gene space but X is in embedding space,
            # project velocity (requires PCA loadings stored in adata.varm)
            if V_full.shape[1] != X.shape[1]:
                if "PCs" in adata.varm and use_rep == "X_pca":
                    PCs = adata.varm["PCs"][:, : X.shape[1]]  # (G, n_pcs)
                    V_full = V_full @ PCs                      # (N, n_pcs)
                    warnings.warn(
                        f"[tensors_from_anndata] Velocity projected from gene space "
                        f"({vel_raw.shape[1]}d) to PCA space ({X.shape[1]}d) via adata.varm['PCs'].",
                        stacklevel=2,
                    )
                else:
                    warnings.warn(
                        f"[tensors_from_anndata] Velocity shape {V_full.shape} does "
                        f"not match representation shape {tuple(X.shape)}. "
                        f"Dropping velocity prior.",
                        stacklevel=2,
                    )
                    V_full = None

            if V_full is not None:
                V = torch.tensor(V_full, device=device)

    # ── Pseudotime ─────────────────────────────────────────────────────────
    T: Optional[torch.Tensor] = None
    if pseudotime_key is not None:
        if pseudotime_key not in adata.obs.columns:
            raise KeyError(
                f"Pseudotime key '{pseudotime_key}' not found in adata.obs. "
                f"Available columns: {list(adata.obs.columns)}"
            )

        pt = adata.obs[pseudotime_key].values.astype(np.float32)
        pt_min, pt_max = float(pt.min()), float(pt.max())

        _in_unit_range = (pt_min >= 0.0) and (pt_max <= 1.0)

        if not _in_unit_range:
            if normalize_pseudotime:
                warnings.warn(
                    f"[tensors_from_anndata] Pseudotime '{pseudotime_key}' is NOT "
                    f"in [0, 1] (found range [{pt_min:.4f}, {pt_max:.4f}]). "
                    f"Automatically normalising to [0, 1] because "
                    f"normalize_pseudotime=True. "
                    f"\n⚠  The drift field gate g(t) = 4t(1-t) is only meaningful "
                    f"for t ∈ [0, 1]. Always normalise pseudotime before training.",
                    stacklevel=2,
                )
                pt = (pt - pt_min) / (pt_max - pt_min + 1e-8)
            else:
                raise ValueError(
                    f"\n[tensors_from_anndata] Pseudotime '{pseudotime_key}' is NOT "
                    f"in [0, 1] — found range [{pt_min:.4f}, {pt_max:.4f}].\n\n"
                    f"scjdo's drift field uses g(t) = 4t(1-t) as a time-dependent "
                    f"gate for the velocity prior, which is only meaningful when "
                    f"t ∈ [0, 1].  Raw pseudotime from DPT, Monocle, Palantir, etc. "
                    f"is typically NOT normalised.\n\n"
                    f"Fix options:\n"
                    f"  1. Normalise manually (recommended):\n"
                    f"       pt = adata.obs['{pseudotime_key}']\n"
                    f"       adata.obs['{pseudotime_key}_norm'] = "
                    f"(pt - pt.min()) / (pt.max() - pt.min())\n"
                    f"       Then pass pseudotime_key='{pseudotime_key}_norm'\n\n"
                    f"  2. Let scjdo normalise automatically:\n"
                    f"       tensors_from_anndata(..., normalize_pseudotime=True)\n"
                )

        T = torch.tensor(pt, device=device)

    return X, V, T


# ---------------------------------------------------------------------------
# Convenience: store results back into AnnData
# ---------------------------------------------------------------------------


def store_predictions(
    adata: ad.AnnData,
    drift: Optional[np.ndarray] = None,
    trajectories: Optional[np.ndarray] = None,
    regimes: Optional[np.ndarray] = None,
    prefix: str = "scjdo",
) -> ad.AnnData:
    """
    Store model outputs back into the AnnData object.

    Parameters
    ----------
    drift        : (N, D) drift vectors → stored in adata.obsm[f'{prefix}_drift'].
    trajectories : (N, T, D) trajectory array → stored in adata.uns[f'{prefix}_traj'].
    regimes      : (N,) string labels → stored in adata.obs[f'{prefix}_regime'].
    prefix       : Key prefix for all stored results.
    """
    if drift is not None:
        adata.obsm[f"{prefix}_drift"] = drift
    if trajectories is not None:
        adata.uns[f"{prefix}_trajectories"] = trajectories
    if regimes is not None:
        adata.obs[f"{prefix}_regime"] = regimes
    return adata
