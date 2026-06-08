"""
Preprocessing: one-call trajectory setup.

Supports three latent space backends (pca, scvi, harmony) and two
pseudotime methods (dpt, palantir).  All downstream scJDO tools
automatically use whatever representation was stored here.
"""
from __future__ import annotations

import warnings
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Latent space helpers
# ---------------------------------------------------------------------------

def _run_pca(adata, n_pcs: int):
    import scanpy as sc
    sc.tl.pca(adata, n_comps=n_pcs, svd_solver="arpack")
    adata.obsm["X_latent"] = adata.obsm["X_pca"].copy()
    print(f"[latent] PCA: {n_pcs} components")


def _run_scvi(adata, n_latent: int, n_epochs: int, batch_key: Optional[str]):
    try:
        import scvi
    except ImportError:
        raise ImportError(
            "scVI not installed. Run:  pip install scvi-tools\n"
            "Or use latent='pca' instead."
        )
    # scVI requires raw counts — store a raw layer if not present
    import scipy.sparse as sp
    if "counts" not in adata.layers:
        # Check if X looks like counts (integers, max > 50 already handled upstream)
        X = adata.X.toarray() if sp.issparse(adata.X) else adata.X
        adata.layers["counts"] = X.copy()
        warnings.warn(
            "[scVI] Storing adata.X as 'counts' layer. "
            "If your data is already log-normalized, pass raw counts first.",
            UserWarning, stacklevel=4,
        )

    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=batch_key)
    model = scvi.model.SCVI(adata, n_latent=n_latent, n_layers=2, n_hidden=128)
    model.train(max_epochs=n_epochs, plan_kwargs={"lr": 1e-3},
                early_stopping=True, early_stopping_patience=20)

    adata.obsm["X_scvi"]   = model.get_latent_representation()
    adata.obsm["X_latent"] = adata.obsm["X_scvi"].copy()
    adata.uns["scvi_model_params"] = {
        "n_latent": n_latent, "n_epochs": n_epochs, "batch_key": batch_key
    }
    print(f"[latent] scVI: {n_latent} latent dimensions")
    return model


# ---------------------------------------------------------------------------
# Pseudotime helpers
# ---------------------------------------------------------------------------

def _run_dpt(adata, root: str, groupby: str, time_key: str, n_neighbors: int, n_pcs: int):
    import scanpy as sc
    # Use latent space for neighbors if scVI was run
    use_rep = "X_latent" if "X_latent" in adata.obsm else "X_pca"
    n_dim   = adata.obsm[use_rep].shape[1]
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=use_rep,
                    n_pcs=min(n_dim, 50), metric="euclidean")

    mask = (adata.obs[groupby] == root).values
    if mask.sum() == 0:
        raise ValueError(
            f"root='{root}' not found in adata.obs['{groupby}']. "
            f"Available: {sorted(adata.obs[groupby].unique().tolist())}"
        )
    adata.uns["iroot"] = int(np.flatnonzero(mask)[0])
    sc.tl.dpt(adata, n_dcs=10)

    pt = adata.obs["dpt_pseudotime"].values.astype(np.float32)
    pt = np.where(np.isnan(pt), float(np.nanmedian(pt)), pt)
    pt = (pt - pt.min()) / (pt.max() - pt.min() + 1e-8)
    adata.obs[time_key] = pt
    print(f"[pseudotime] DPT: root={root}, range=[0, 1]")


def _run_palantir(
    adata,
    root: str,
    groupby: str,
    time_key: str,
    n_waypoints: int,
    n_components: int,
):
    """
    Run Palantir pseudotime and branch probabilities.

    Stores:
      adata.obs[time_key]              — normalized pseudotime [0, 1]
      adata.obs['palantir_entropy']    — differentiation entropy
      adata.obsm['palantir_branch_probs'] — (N, n_branches) branch probabilities
      adata.uns['palantir_branch_names']  — branch terminal state names
    """
    try:
        import palantir
    except ImportError:
        raise ImportError(
            "Palantir not installed. Run:  pip install palantir\n"
            "Or use pseudotime_method='dpt' instead."
        )

    # Find root cell — cell in root cluster closest to its cluster centroid
    mask        = (adata.obs[groupby] == root).values
    X_lat       = adata.obsm.get("X_latent", adata.obsm.get("X_pca"))
    centroid    = X_lat[mask].mean(0)
    dists       = np.linalg.norm(X_lat[mask] - centroid, axis=1)
    root_idx    = int(np.flatnonzero(mask)[np.argmin(dists)])
    root_cell   = adata.obs_names[root_idx]

    # MAGIC imputation (Palantir expects smoothed expression)
    try:
        import magic
        magic_op = magic.MAGIC(n_components=n_components, random_state=42)
        data_df  = palantir.utils.run_magic_imputation(adata, dm_res=None)
    except Exception:
        # Fallback: use diffusion components if MAGIC fails
        import pandas as pd
        data_df = pd.DataFrame(
            X_lat[:, :n_components],
            index=adata.obs_names,
        )

    # Run Palantir
    pr_res = palantir.core.run_palantir(
        data_df,
        root_cell,
        num_waypoints=n_waypoints,
        use_early_cell_as_start=True,
    )

    # Store pseudotime
    pt = pr_res.pseudotime.reindex(adata.obs_names).values.astype(np.float32)
    pt = np.where(np.isnan(pt), float(np.nanmedian(pt[~np.isnan(pt)])), pt)
    pt = (pt - pt.min()) / (pt.max() - pt.min() + 1e-8)
    adata.obs[time_key]             = pt
    adata.obs["palantir_entropy"]   = pr_res.entropy.reindex(adata.obs_names).values

    # Store branch probabilities
    bp = pr_res.branch_probs.reindex(adata.obs_names).fillna(0).values.astype(np.float32)
    adata.obsm["palantir_branch_probs"] = bp
    adata.uns["palantir_branch_names"]  = list(pr_res.branch_probs.columns)

    print(f"[pseudotime] Palantir: root={root_cell}, "
          f"branches={adata.uns['palantir_branch_names']}, "
          f"entropy mean={adata.obs['palantir_entropy'].mean():.3f}")


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def prepare_trajectory(
    adata,
    *,
    groupby: Optional[str] = None,
    root: Optional[str] = None,
    # Latent space
    latent: str = "pca",
    n_pcs: int = 50,
    n_latent: int = 20,
    n_scvi_epochs: int = 400,
    batch_key: Optional[str] = None,
    # Pseudotime
    pseudotime_method: str = "dpt",
    n_waypoints: int = 500,
    n_palantir_components: int = 10,
    # Graph
    n_hvg: int = 2000,
    n_neighbors: int = 15,
    time_key: str = "pseudotime",
    compute_umap: bool = True,
    flavor: str = "seurat",
    copy: bool = False,
):
    """
    Normalize, select HVGs, build latent space, and compute pseudotime.

    Parameters
    ----------
    latent : 'pca' (default) | 'scvi'
        Latent space backend.

        ``'pca'`` — fast, linear, sufficient for well-separated datasets.

        ``'scvi'`` — nonlinear probabilistic embedding; significantly better
        for data with technical noise, dropout, or subtle manifold structure.
        Requires ``pip install scvi-tools``. The scVI latent space is stored in
        ``adata.obsm['X_scvi']`` and used as ``rep`` in ``sjd.tl.fit_drift``.

    pseudotime_method : 'dpt' (default) | 'palantir'
        Pseudotime algorithm.

        ``'dpt'`` — fast, Scanpy-native diffusion pseudotime.

        ``'palantir'`` — better for branching systems; returns per-branch
        probabilities stored in ``adata.obsm['palantir_branch_probs']``.
        Use these with ``sjd.tl.fit_drift(branch_key='palantir_branch_probs')``
        to weight each window by branch membership.
        Requires ``pip install palantir``.

    n_latent : int
        scVI latent dimensions (ignored for pca).

    n_scvi_epochs : int
        scVI training epochs (default 400; increase to 800 for larger datasets).

    batch_key : str or None
        adata.obs column for batch correction in scVI.

    n_waypoints : int
        Palantir waypoints (default 500; increase for large datasets).

    Examples
    --------
    **PCA + DPT (default):**

    >>> sjd.pp.prepare_trajectory(adata, groupby='cell_type', root='HSC')

    **scVI + DPT (better geometry):**

    >>> sjd.pp.prepare_trajectory(adata, groupby='cell_type', root='HSC',
    ...                           latent='scvi', n_latent=20)

    **scVI + Palantir (best for branching systems):**

    >>> sjd.pp.prepare_trajectory(adata, groupby='cell_type', root='HSC',
    ...                           latent='scvi', pseudotime_method='palantir')
    >>> # Then in fit_drift, weight by branch:
    >>> sjd.tl.fit_drift(adata, branch_key='palantir_branch_probs')
    """
    import scanpy as sc

    if latent not in ("pca", "scvi"):
        raise ValueError(f"latent must be 'pca' or 'scvi', got '{latent}'")
    if pseudotime_method not in ("dpt", "palantir"):
        raise ValueError(f"pseudotime_method must be 'dpt' or 'palantir', "
                         f"got '{pseudotime_method}'")

    if copy:
        adata = adata.copy()

    # ── Normalization ──────────────────────────────────────────────────────
    import scipy.sparse as sp
    X_check = adata.X.toarray() if sp.issparse(adata.X) else adata.X
    if X_check.max() > 50:
        # Store raw counts before normalization (needed for scVI)
        adata.layers["counts"] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    # ── HVG ───────────────────────────────────────────────────────────────
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor=flavor)
    adata._inplace_subset_var(adata.var.highly_variable)

    # ── Latent space ───────────────────────────────────────────────────────
    # Always compute PCA (needed for UMAP fallback and DPT diffusion map)
    sc.tl.pca(adata, n_comps=n_pcs, svd_solver="arpack")

    if latent == "scvi":
        _run_scvi(adata, n_latent=n_latent, n_epochs=n_scvi_epochs,
                  batch_key=batch_key)
        # Use scVI for neighbors
        sc.pp.neighbors(adata, n_neighbors=n_neighbors,
                        use_rep="X_scvi", metric="euclidean")
    else:
        adata.obsm["X_latent"] = adata.obsm["X_pca"].copy()
        sc.pp.neighbors(adata, n_neighbors=n_neighbors,
                        n_pcs=n_pcs, metric="euclidean")

    if compute_umap:
        sc.tl.umap(adata)

    # ── Pseudotime ─────────────────────────────────────────────────────────
    if root is not None and groupby is not None:
        if groupby not in adata.obs.columns:
            raise ValueError(f"groupby='{groupby}' not found in adata.obs")

        if pseudotime_method == "palantir":
            _run_palantir(adata, root=root, groupby=groupby,
                          time_key=time_key, n_waypoints=n_waypoints,
                          n_components=n_palantir_components)
        else:
            _run_dpt(adata, root=root, groupby=groupby, time_key=time_key,
                     n_neighbors=n_neighbors, n_pcs=n_pcs)

    elif root is not None or groupby is not None:
        warnings.warn(
            "Both `groupby` and `root` must be provided to compute pseudotime. "
            "Skipping.", UserWarning, stacklevel=2,
        )

    # Store which latent/pseudotime was used (for fit_drift auto-detection)
    adata.uns["scjdo_prep"] = {
        "latent":            latent,
        "rep":               "X_scvi" if latent == "scvi" else "X_pca",
        "pseudotime_method": pseudotime_method,
        "time_key":          time_key,
    }

    return adata if copy else None
