"""Preprocessing: one-call trajectory setup."""
from __future__ import annotations

import warnings
import numpy as np


def prepare_trajectory(
    adata,
    *,
    groupby: str | None = None,
    root: str | None = None,
    n_hvg: int = 2000,
    n_pcs: int = 50,
    n_neighbors: int = 15,
    time_key: str = "pseudotime",
    compute_umap: bool = True,
    flavor: str = "seurat",
    copy: bool = False,
):
    """
    Normalize, select HVGs, PCA, neighbors, and compute DPT pseudotime.

    All results are stored in ``adata`` in-place (or a copy if ``copy=True``).
    Pseudotime is normalized to [0, 1] and stored in ``adata.obs[time_key]``.

    Parameters
    ----------
    adata       : AnnData (raw counts or log-normalized).
    groupby     : Column in ``adata.obs`` used to identify the root cluster.
    root        : Value in ``adata.obs[groupby]`` that is the trajectory root.
    n_hvg       : Number of highly variable genes to retain.
    n_pcs       : PCA components.
    n_neighbors : k for the kNN graph.
    time_key    : Key under which normalized pseudotime is stored.
    compute_umap: Whether to compute UMAP (useful for visualization).
    flavor      : HVG flavor passed to ``sc.pp.highly_variable_genes``.
    copy        : Return a copy instead of modifying in place.

    Returns
    -------
    AnnData if ``copy=True``, else None.

    Examples
    --------
    >>> import scanpy as sc
    >>> import scqdiff as sqd
    >>> adata = sc.datasets.paul15()
    >>> sqd.pp.prepare_trajectory(adata, groupby="paul15_clusters", root="7MEP")
    """
    import scanpy as sc

    if copy:
        adata = adata.copy()

    # ── Normalization ──────────────────────────────────────────────────────
    if adata.X.max() > 50:   # raw counts heuristic
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    # ── HVG + PCA ──────────────────────────────────────────────────────────
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor=flavor)
    adata._inplace_subset_var(adata.var.highly_variable)
    sc.tl.pca(adata, n_comps=n_pcs, svd_solver="arpack")

    # ── Neighbors + UMAP ───────────────────────────────────────────────────
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, metric="euclidean")
    if compute_umap:
        sc.tl.umap(adata)

    # ── DPT pseudotime ─────────────────────────────────────────────────────
    if root is not None and groupby is not None:
        if groupby not in adata.obs.columns:
            raise ValueError(f"groupby='{groupby}' not found in adata.obs")

        mask = (adata.obs[groupby] == root).values
        if mask.sum() == 0:
            raise ValueError(
                f"root='{root}' not found in adata.obs['{groupby}']. "
                f"Available values: {sorted(adata.obs[groupby].unique().tolist())}"
            )

        adata.uns["iroot"] = int(np.flatnonzero(mask)[0])
        sc.tl.dpt(adata, n_dcs=10)

        pt = adata.obs["dpt_pseudotime"].values.astype(np.float32)
        pt = np.where(np.isnan(pt), float(np.nanmedian(pt)), pt)
        pt = (pt - pt.min()) / (pt.max() - pt.min() + 1e-8)
        adata.obs[time_key] = pt
    elif root is not None or groupby is not None:
        warnings.warn(
            "Both `groupby` and `root` must be provided to compute pseudotime. "
            "Skipping DPT.",
            UserWarning, stacklevel=2,
        )

    return adata if copy else None
