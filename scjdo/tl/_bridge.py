"""
High-level Schrödinger Bridge training, Jacobian analysis, and archetype decomposition.
Results stored in adata.uns[key_added].
"""
from __future__ import annotations

from typing import Optional, Tuple
import warnings
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _select_populations(adata, time_key, src_quantile, tgt_quantile,
                         src_group, tgt_group, groupby):
    """Return (src_mask, tgt_mask) boolean arrays."""
    if src_group is not None and tgt_group is not None and groupby is not None:
        if groupby not in adata.obs.columns:
            raise KeyError(f"groupby='{groupby}' not in adata.obs")
        src_mask = (adata.obs[groupby] == src_group).values
        tgt_mask = (adata.obs[groupby] == tgt_group).values
    else:
        if time_key not in adata.obs.columns:
            raise KeyError(f"time_key='{time_key}' not in adata.obs. "
                           "Run sjd.pp.prepare_trajectory first.")
        pt       = adata.obs[time_key].values
        src_mask = pt <= np.quantile(pt, src_quantile)
        tgt_mask = pt >= np.quantile(pt, tgt_quantile)
    return src_mask, tgt_mask


def _compute_jacobian_tensor(bridge, fwd_traj, bwd_traj, t_vals, steps, n_pcs):
    """Compute Jacobian tensors along both bridge directions with sign consistency."""
    T       = len(t_vals)
    step_idx = (t_vals * steps).astype(int).clip(0, steps)

    max_eig_fwd = np.zeros(T)
    max_eig_bwd = np.zeros(T)
    evec_fwd    = np.zeros((T, n_pcs))
    evec_bwd    = np.zeros((T, n_pcs))
    J_fwd       = np.zeros((T, n_pcs, n_pcs))
    J_bwd       = np.zeros((T, n_pcs, n_pcs))

    prev_f = prev_b = None
    bridge.forward_net.eval()
    bridge.backward_net.eval()

    for i, (t_val, si) in enumerate(zip(t_vals, step_idx)):
        t_batch = torch.full((1,), float(t_val))

        # Forward
        x_f = fwd_traj[:, si, :].mean(0, keepdim=True)
        Jf  = bridge.jacobian(x_f, t_batch, forward=True)[0].numpy()
        J_fwd[i] = Jf
        ev_f, evec_f = np.linalg.eig(Jf)
        idx_f = int(np.argmax(np.real(ev_f)))
        v_f   = np.real(evec_f[:, idx_f])
        v_f  /= np.linalg.norm(v_f) + 1e-8
        if prev_f is not None and np.dot(v_f, prev_f) < 0:
            v_f = -v_f
        prev_f = v_f
        max_eig_fwd[i] = float(np.real(ev_f[idx_f]))
        evec_fwd[i]    = v_f

        # Backward
        x_b = bwd_traj[:, si, :].mean(0, keepdim=True)
        Jb  = bridge.jacobian(x_b, t_batch, forward=False)[0].numpy()
        J_bwd[i] = Jb
        ev_b, evec_b = np.linalg.eig(Jb)
        idx_b = int(np.argmax(np.real(ev_b)))
        v_b   = np.real(evec_b[:, idx_b])
        v_b  /= np.linalg.norm(v_b) + 1e-8
        if prev_b is not None and np.dot(v_b, prev_b) < 0:
            v_b = -v_b
        prev_b = v_b
        max_eig_bwd[i] = float(np.real(ev_b[idx_b]))
        evec_bwd[i]    = v_b

    return (gaussian_filter1d(J_fwd, sigma=1.5, axis=0),
            gaussian_filter1d(J_bwd, sigma=1.5, axis=0),
            max_eig_fwd, max_eig_bwd, evec_fwd, evec_bwd)


def _arch_genes(pat_np, act_np, evec, max_eig, t_vals,
                pca_load, gene_names, n_genes, sens_thresh, act_thresh):
    """Gene extraction per archetype for one bridge direction."""
    import pandas as pd
    K    = pat_np.shape[0]
    sens = max_eig > sens_thresh
    rows = []
    for k in range(K):
        mask = (act_np[:, k] >= act_thresh) & sens
        if mask.sum() == 0:
            mask = sens
        if mask.sum() == 0:
            continue
        mean_sc = np.abs(pca_load @ evec[mask].T).mean(axis=1)
        top_idx = np.argsort(mean_sc)[::-1][:n_genes]
        peak_t  = float(t_vals[np.argmax(act_np[:, k])])
        mean_s  = float(max_eig[mask].mean())
        for rank, gi in enumerate(top_idx, 1):
            rows.append({
                "archetype":        f"A{k+1}",
                "peak_t":           round(peak_t, 3),
                "mean_sensitivity": round(mean_s, 4),
                "gene":             gene_names[gi],
                "instability_score":round(float(mean_sc[gi]), 4),
                "rank":             rank,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_bridge(
    adata,
    *,
    time_key: str = "pseudotime",
    rep: str = "X_pca",
    # Population selection
    src_quantile: float = 0.20,
    tgt_quantile: float = 0.80,
    src_group: Optional[str] = None,
    tgt_group: Optional[str] = None,
    groupby: Optional[str] = None,
    # Bridge architecture
    hidden: int = 256,
    depth: int = 4,
    epsilon: float = 0.5,
    n_score_steps: int = 500,
    max_iterations: int = 30,
    # Analysis
    n_archetypes: int = 4,
    t_steps: int = 30,
    n_traj: int = 80,
    steps: int = 100,
    n_genes: int = 20,
    sens_thresh: float = 0.05,
    act_thresh: float = 0.5,
    seed: int = 42,
    device: Optional[str] = None,
    key_added: str = "scjdo_bridge",
    verbose: bool = True,
):
    """
    Full Schrödinger Bridge pipeline: train, simulate, compute instability
    archetypes, and extract top instability genes for both forward and backward
    directions.

    All results are stored in ``adata.uns[key_added]``.

    Population selection
    --------------------
    By pseudotime quantile (default)::

        sjd.tl.fit_bridge(adata, src_quantile=0.20, tgt_quantile=0.80)

    By cell-type label::

        sjd.tl.fit_bridge(adata, groupby='cell_type',
                          src_group='progenitor', tgt_group='committed')

    Parameters
    ----------
    src_quantile / tgt_quantile : Bottom / top pseudotime fraction to use as
        source / target populations.
    src_group / tgt_group / groupby : Alternative: define populations by
        cluster label rather than pseudotime.
    n_archetypes : Number of operator archetypes per bridge direction.
    t_steps      : Bridge time steps for Jacobian analysis.
    n_genes      : Top instability genes to extract per archetype.

    Returns
    -------
    bridge : Trained ``SchrodingerBridge`` object (also stored in
             ``adata.uns[key_added]['_bridge']``).
    """
    from scjdo.models.schrodinger_bridge import SchrodingerBridge, SchrodingerBridgeConfig
    from scjdo.archetypes.decompose import jacobian_modes

    torch.manual_seed(seed)
    np.random.seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if rep not in adata.obsm:
        raise KeyError(f"'{rep}' not in adata.obsm. Run sjd.pp.prepare_trajectory first.")

    X_np       = adata.obsm[rep].astype("float32")
    n_pcs      = X_np.shape[1]
    X_pca      = torch.tensor(X_np)
    pca_load   = adata.varm.get("PCs", None)
    gene_names = list(adata.var_names) if pca_load is not None else []
    if pca_load is not None:
        pca_load = pca_load.astype("float32")

    # ── Source / target selection ──────────────────────────────────────────
    src_mask, tgt_mask = _select_populations(
        adata, time_key, src_quantile, tgt_quantile, src_group, tgt_group, groupby
    )
    X_src = X_pca[src_mask]
    X_tgt = X_pca[tgt_mask]

    if verbose:
        print(f"Source: {src_mask.sum()} cells | Target: {tgt_mask.sum()} cells")

    # ── Train bridge ───────────────────────────────────────────────────────
    cfg = SchrodingerBridgeConfig(
        dim=n_pcs, hidden=hidden, depth=depth,
        epsilon=epsilon, n_score_steps=n_score_steps,
        max_iterations=max_iterations, patience=3,
    )
    bridge  = SchrodingerBridge(cfg, X_src, X_tgt)
    history = bridge.train_bridge(verbose=verbose)

    if verbose:
        print(f"Bridge converged: {history['converged']}  "
              f"Iterations: {history['n_iters']}")

    # ── Simulate trajectories ─────────────────────────────────────────────
    torch.manual_seed(seed)
    idx_src  = torch.randperm(len(X_src))[:n_traj]
    idx_tgt  = torch.randperm(len(X_tgt))[:n_traj]
    fwd_traj = bridge.forward_integrate(X_src[idx_src], steps=steps, stochastic=False)
    bwd_traj = bridge.backward_integrate(X_tgt[idx_tgt], steps=steps, stochastic=False)
    # Store trajectories in PC1/PC2 space for visualization
    fwd_traj_2d = fwd_traj[:, :, :2].detach().numpy()   # (n_traj, steps+1, 2)
    bwd_traj_2d = bwd_traj[:, :, :2].detach().numpy()

    # ── Jacobian tensors ──────────────────────────────────────────────────
    if verbose:
        print(f"Computing Jacobians ({t_steps} time steps × 2 directions)...")

    t_vals = np.linspace(0.03, 0.97, t_steps)
    (J_fwd, J_bwd, max_eig_fwd, max_eig_bwd,
     evec_fwd, evec_bwd) = _compute_jacobian_tensor(
        bridge, fwd_traj, bwd_traj, t_vals, steps, n_pcs
    )

    # ── Archetype decomposition ───────────────────────────────────────────
    if verbose:
        print("Running semi-NMF archetype decomposition...")

    pat_fwd, act_fwd, err_fwd = jacobian_modes(
        torch.from_numpy(J_fwd), rank=n_archetypes, n_restarts=5, seed=seed)
    pat_bwd, act_bwd, err_bwd = jacobian_modes(
        torch.from_numpy(J_bwd), rank=n_archetypes, n_restarts=5, seed=seed)

    act_fwd_np = act_fwd.numpy().copy()
    act_bwd_np = act_bwd.numpy().copy()
    for k in range(n_archetypes):
        act_fwd_np[:, k] /= (act_fwd_np[:, k].max() + 1e-8)
        act_bwd_np[:, k] /= (act_bwd_np[:, k].max() + 1e-8)

    # ── Gene extraction ───────────────────────────────────────────────────
    df_fwd = df_bwd = None
    if pca_load is not None:
        df_fwd = _arch_genes(pat_fwd.numpy(), act_fwd_np, evec_fwd, max_eig_fwd,
                              t_vals, pca_load, gene_names, n_genes, sens_thresh, act_thresh)
        df_bwd = _arch_genes(pat_bwd.numpy(), act_bwd_np, evec_bwd, max_eig_bwd,
                              t_vals, pca_load, gene_names, n_genes, sens_thresh, act_thresh)

    # ── Store results ─────────────────────────────────────────────────────
    adata.uns[key_added] = {
        "src_mask":      src_mask,
        "tgt_mask":      tgt_mask,
        "t_vals":        t_vals,
        "J_fwd":         J_fwd,
        "J_bwd":         J_bwd,
        "max_eig_fwd":   max_eig_fwd,
        "max_eig_bwd":   max_eig_bwd,
        "evec_fwd":      evec_fwd,
        "evec_bwd":      evec_bwd,
        "pat_fwd":       pat_fwd.numpy(),
        "pat_bwd":       pat_bwd.numpy(),
        "act_fwd":       act_fwd_np,
        "act_bwd":       act_bwd_np,
        "history":       history,
        "gene_names":    gene_names,
        "df_fwd":        df_fwd.to_dict("records") if df_fwd is not None else [],
        "df_bwd":        df_bwd.to_dict("records") if df_bwd is not None else [],
        "fwd_traj_2d":   fwd_traj_2d,   # (n_traj, steps+1, 2) for PCA trajectory plot
        "bwd_traj_2d":   bwd_traj_2d,
        "_bridge":       bridge,   # live object for further analysis
        "params": {
            "src_quantile": src_quantile, "tgt_quantile": tgt_quantile,
            "src_group": src_group, "tgt_group": tgt_group, "groupby": groupby,
            "n_archetypes": n_archetypes, "t_steps": t_steps,
            "epsilon": epsilon, "hidden": hidden, "depth": depth,
            "n_genes": n_genes, "seed": seed, "rep": rep,
        },
    }

    if verbose:
        print(f"Done. Results in adata.uns['{key_added}']")
        if df_fwd is not None:
            fwd_top = df_fwd[df_fwd["rank"] == 1]["gene"].tolist()
            bwd_top = df_bwd[df_bwd["rank"] == 1]["gene"].tolist()
            print(f"  Top forward gene per archetype:  {fwd_top}")
            print(f"  Top backward gene per archetype: {bwd_top}")

    return bridge


def get_bridge_instability_genes(
    adata,
    key: str = "scjdo_bridge",
    n_genes: Optional[int] = None,
    top_archetypes: Optional[int] = None,
) -> Tuple["pd.DataFrame", "pd.DataFrame"]:
    """
    Return ranked instability gene tables for forward and backward bridge directions.

    Parameters
    ----------
    n_genes       : Limit to top N genes per archetype (None = all stored).
    top_archetypes: Return only the K most sensitive archetypes per direction.

    Returns
    -------
    df_fwd, df_bwd : pandas DataFrames with columns
        archetype, peak_t, mean_sensitivity, gene, instability_score, rank
    """
    import pandas as pd

    if key not in adata.uns:
        raise KeyError(f"'{key}' not in adata.uns. Run sjd.tl.fit_bridge first.")

    res = adata.uns[key]

    def _to_df(records, n_genes, top_archetypes):
        df = pd.DataFrame(records)
        if df.empty:
            return df
        if n_genes:
            df = df[df["rank"] <= n_genes]
        if top_archetypes:
            top_arch = (df.groupby("archetype")["mean_sensitivity"]
                        .mean().nlargest(top_archetypes).index.tolist())
            df = df[df["archetype"].isin(top_arch)]
        return df.reset_index(drop=True)

    df_fwd = _to_df(res["df_fwd"], n_genes, top_archetypes)
    df_bwd = _to_df(res["df_bwd"], n_genes, top_archetypes)
    return df_fwd, df_bwd


# ---------------------------------------------------------------------------
# Per-target wrapper — the bridge analog of fit_drift_branches.
# Encapsulates the per-perturbation loop used in Figure5.
# ---------------------------------------------------------------------------

def fit_bridge_branches(
    adata,
    *,
    groupby: str,
    src_group: str,
    tgt_groups,
    time_key: str = "bridge_t",
    rep: str = "X_pca",
    key_prefix: str = "scjdo_bridge",
    auto_time_key: bool = True,
    verbose: bool = True,
    **fit_bridge_kwargs,
) -> dict:
    """
    Run :func:`fit_bridge` separately for each target group against a shared source.

    The natural API for snapshot perturb-seq / paired-condition data: a single
    control population (e.g. NT sgRNAs) and several target populations (e.g.
    different gene knockdowns). Each (src, tgt) pair gets its own forward +
    backward bridge, Jacobian tensor, archetype decomposition, and instability
    gene table.

    Parameters
    ----------
    adata : AnnData
        Must contain ``adata.obs[groupby]`` with the group labels and
        ``adata.obsm[rep]`` with the latent coordinates.
    groupby : str
        Column in ``adata.obs`` containing the group labels (e.g. ``'target'``
        with values ``Non-Targeting``, ``PVT1``, ``MALAT1`` …).
    src_group : str
        The control / source group label (e.g. ``'Non-Targeting'``).
    tgt_groups : list of str
        Target group labels — one bridge per entry.
    time_key : str
        Column in ``adata.obs`` providing the bridge time axis.
        :func:`fit_bridge` interpolates on this axis; the standard convention
        for paired populations is ``0`` for source, ``1`` for target. If
        ``auto_time_key=True`` and the column is missing, it is created from
        ``(adata.obs[groupby] != src_group).astype('float32')``.
    rep : str
        ``adata.obsm`` key for the latent representation (e.g. ``'X_fa'``).
    key_prefix : str
        Each bridge is stored under ``adata.uns[f'{key_prefix}_{name}']``.
        Default produces keys like ``'scjdo_bridge_PVT1'``.
    auto_time_key : bool
        If True (default), build ``adata.obs[time_key]`` from ``groupby`` when
        absent. Set to False to require the column already exist.
    verbose : bool
        One-line summary per target.
    **fit_bridge_kwargs
        Forwarded to :func:`fit_bridge` (``epsilon``, ``max_iterations``,
        ``n_score_steps``, ``n_archetypes``, ``n_genes``, ``seed``, …).

    Returns
    -------
    dict
        ``{tgt_group: fit_bridge_return_value}``. Each value is the trained
        Bridge object; full results are in ``adata.uns[f'{key_prefix}_{name}']``.

    Examples
    --------
    >>> bridges = sjd.tl.fit_bridge_branches(
    ...     adata, groupby='target',
    ...     src_group='Non-Targeting',
    ...     tgt_groups=['PVT1', 'MALAT1', 'PSMA3-AS1'],
    ...     rep='X_fa', n_archetypes=5, n_genes=50, seed=42)
    >>> sjd.tl.infer_regulators_branches(
    ...     adata, bridges, direction='both',
    ...     organism='human', min_targets=1, n_top=15)
    """
    if groupby not in adata.obs.columns:
        raise KeyError(f"groupby='{groupby}' not in adata.obs.columns")

    if time_key not in adata.obs.columns:
        if not auto_time_key:
            raise KeyError(
                f"time_key='{time_key}' not in adata.obs and "
                f"auto_time_key=False. Either create it manually or pass "
                f"auto_time_key=True to derive it from `groupby`."
            )
        adata.obs[time_key] = (
            adata.obs[groupby] != src_group
        ).astype("float32")
        if verbose:
            print(f"[bridge] auto-created adata.obs['{time_key}']: "
                  f"0={src_group}, 1=other")

    src_n = int((adata.obs[groupby] == src_group).sum())
    if src_n == 0:
        raise ValueError(f"src_group='{src_group}' has 0 cells in "
                         f"adata.obs['{groupby}']")
    if verbose:
        print(f"[bridge] source '{src_group}': {src_n} cells")

    out = {}
    for tgt in tgt_groups:
        tgt_n = int((adata.obs[groupby] == tgt).sum())
        if tgt_n == 0:
            warnings.warn(
                f"tgt_group='{tgt}' has 0 cells in adata.obs['{groupby}']; "
                f"skipping.",
                UserWarning, stacklevel=2,
            )
            continue
        key_added = f"{key_prefix}_{tgt}"
        if verbose:
            print(f"[bridge] {src_group} → {tgt}  (n_tgt={tgt_n}) ...")
        try:
            bridge = fit_bridge(
                adata,
                rep=rep,
                time_key=time_key,
                groupby=groupby,
                src_group=src_group,
                tgt_group=tgt,
                key_added=key_added,
                verbose=False,
                **fit_bridge_kwargs,
            )
            out[tgt] = bridge
            if verbose:
                res = adata.uns[key_added]
                import numpy as _np
                pfwd = int(_np.argmax(res["max_eig_fwd"]))
                print(f"           fwd peak_t={res['t_vals'][pfwd]:.2f}  "
                      f"max Re(λ)={res['max_eig_fwd'][pfwd]:+.3f}")
        except Exception as e:
            warnings.warn(
                f"[{tgt}] fit_bridge raised {type(e).__name__}: {e}",
                UserWarning, stacklevel=2,
            )

    return out
