"""
High-level drift field training and Jacobian tensor decomposition.
All results are stored in adata.uns[key_added].
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from tqdm.auto import trange


# ---------------------------------------------------------------------------
# Pseudotime velocity helper
# ---------------------------------------------------------------------------

def _pseudotime_velocity(X_pca: np.ndarray, pseudotime: np.ndarray, k: int = 15) -> np.ndarray:
    """
    Compute a directional velocity from DPT pseudotime.
    For each cell: weighted average displacement toward higher-pseudotime neighbours.
    Gives the drift field a directional signal when RNA velocity is unavailable.
    """
    from sklearn.neighbors import NearestNeighbors
    N = X_pca.shape[0]
    V = np.zeros_like(X_pca, dtype=np.float32)
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X_pca)
    _, idx = nbrs.kneighbors(X_pca)
    for i in range(N):
        nbr = idx[i, 1:]
        dt  = pseudotime[nbr] - pseudotime[i]
        fwd = dt > 0
        if not fwd.any():
            continue
        w    = dt[fwd]
        disp = X_pca[nbr[fwd]] - X_pca[i]
        V[i] = (w[:, None] * disp).sum(0) / w.sum()
    mag = np.linalg.norm(V, axis=1).mean() + 1e-8
    return V / mag   # unit-mean magnitude


# ---------------------------------------------------------------------------
# Pseudotime window construction
# ---------------------------------------------------------------------------

def _build_windows(
    pseudotime: np.ndarray,
    n_windows: int = 100,
    overlap: float = 0.80,
    min_cells: int = 5,
) -> list[tuple[float, float, np.ndarray]]:
    """Build overlapping pseudotime windows with guaranteed minimum cell count."""
    width  = 1.0 / (1.0 + (n_windows - 1) * (1.0 - overlap))
    step   = width * (1.0 - overlap)
    windows = []
    for i in range(n_windows):
        lo  = i * step
        hi  = lo + width
        idx = np.where((pseudotime >= lo) & (pseudotime <= hi))[0]
        # Expand window if too sparse
        if len(idx) < min_cells:
            dists = np.abs(pseudotime - (lo + hi) / 2)
            idx   = np.argsort(dists)[:min_cells]
        windows.append((lo, hi, idx))
    return windows


# ---------------------------------------------------------------------------
# Gene-space projection
# ---------------------------------------------------------------------------

def _gene_scores(patterns: torch.Tensor, pca_loadings: np.ndarray, sign_flip: list[float]):
    """Project archetype patterns to gene space and return ranked gene scores."""
    K = patterns.shape[0]
    results = {}
    for k in range(K):
        A_k     = patterns[k].numpy() * sign_flip[k]
        eigvals, eigvecs = np.linalg.eig(A_k)
        max_idx = int(np.argmax(np.real(eigvals)))
        v_max   = np.real(eigvecs[:, max_idx])
        scores  = pca_loadings @ v_max        # (n_genes,)
        results[k] = scores
    return results


# ---------------------------------------------------------------------------
# Main API function
# ---------------------------------------------------------------------------

def fit_drift(
    adata,
    *,
    time_key: str = "pseudotime",
    rep: str = "X_pca",
    # Model
    hidden: int = 256,
    depth: int = 4,
    beta: float = 0.1,
    use_spectral_norm: bool = True,
    # Velocity prior
    vel_scale: float = 2.0,
    vel_time_mode: str = "flat",
    # Training
    n_epochs: int = 5000,
    batch_size: int = 512,
    lr: float = 2e-4,
    weight_decay: float = 1e-4,
    sigma: float = 0.1,
    use_local_sigma: bool = False,
    seed: int = 42,
    device: Optional[str] = None,
    # Jacobian tensor
    n_windows: int = 100,
    overlap: float = 0.80,
    smooth_sigma: float = 1.5,
    # Archetype decomposition
    n_archetypes: int = 5,
    n_restarts: int = 5,
    # Output
    key_added: str = "scqdiff",
    verbose: bool = True,
):
    """
    Train a drift field and compute archetype decomposition.

    Automatically derives a pseudotime-gradient velocity prior when no RNA
    velocity is available, so the drift field follows the developmental path.

    Results are stored in ``adata.uns[key_added]``:

    ========================  ================================================
    Key                       Content
    ========================  ================================================
    ``J_tensor``              Jacobian tensor (n_windows, D, D), numpy
    ``t_centers``             Window pseudotime centers (n_windows,), numpy
    ``patterns``              Archetype patterns (K, D, D), numpy
    ``activations``           Temporal activations (n_windows, K), numpy
    ``max_real_eig``          Max real eigenvalue per window (n_windows,)
    ``gene_scores``           Gene loadings per archetype {k: (n_genes,)}
    ``corr_mat``              Pairwise activation correlations (K, K)
    ``params``                Training configuration dict
    ========================  ================================================

    Parameters
    ----------
    adata         : AnnData after ``sqd.pp.prepare_trajectory``.
    time_key      : Column in ``adata.obs`` with normalized pseudotime [0,1].
    rep           : Key in ``adata.obsm`` for the latent representation.
    n_epochs      : Training iterations.
    n_archetypes  : Number of operator archetypes (K).
    vel_scale     : Strength of the pseudotime-gradient velocity prior.
    vel_time_mode : Gate shape — 'flat' (constant), 'root', 'rise', 'mid'.
    key_added     : Where to store results in ``adata.uns``.

    Examples
    --------
    >>> sqd.tl.fit_drift(adata, time_key="pseudotime", n_archetypes=5, n_epochs=5000)
    >>> adata.uns["scqdiff"].keys()
    dict_keys(['J_tensor', 't_centers', 'patterns', 'activations', ...])
    """
    from scqdiff.models.drift import DriftField, DriftConfig
    from scqdiff.losses import denoising_score_matching, control_energy, local_sigma
    from scqdiff.archetypes.decompose import jacobian_modes

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Extract tensors ────────────────────────────────────────────────────
    if rep not in adata.obsm:
        raise KeyError(f"Representation '{rep}' not in adata.obsm. Run sqd.pp.prepare_trajectory first.")
    if time_key not in adata.obs.columns:
        raise KeyError(f"time_key='{time_key}' not in adata.obs. Run sqd.pp.prepare_trajectory first.")

    X_np = adata.obsm[rep].astype(np.float32)
    T_np = adata.obs[time_key].values.astype(np.float32)
    N, D = X_np.shape

    X = torch.tensor(X_np, device=device)
    T = torch.tensor(T_np, device=device)

    # ── Pseudotime-gradient velocity (auto) ───────────────────────────────
    if verbose:
        print("Computing pseudotime-gradient velocity prior...")
    V_np     = _pseudotime_velocity(X_np, T_np, k=15)
    V_tensor = torch.tensor(V_np, device=device)

    # ── Build model ────────────────────────────────────────────────────────
    cfg = DriftConfig(
        dim=D, hidden=hidden, depth=depth, beta=beta,
        use_spectral_norm=use_spectral_norm,
        use_velocity_prior=True,
        vel_scale=vel_scale,
        vel_k=15,
        vel_time_mode=vel_time_mode,
    )
    model = DriftField(cfg, X_ref=X, V_ref=V_tensor).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    if verbose:
        print(f"DriftField: {sum(p.numel() for p in model.parameters()):,} parameters | "
              f"device={device} | epochs={n_epochs}")

    # Precompute local sigma if requested
    sigma_per_cell = None
    if use_local_sigma:
        sigma_per_cell = local_sigma(X).cpu()

    # ── Training ───────────────────────────────────────────────────────────
    losses = []
    model.train()
    for step in trange(n_epochs, desc="Training drift field", disable=not verbose):
        idx    = torch.randint(0, N, (batch_size,), device=device)
        xb, tb = X[idx], T[idx]

        if sigma_per_cell is not None:
            sig = sigma_per_cell[idx.cpu()].to(device)
        else:
            sig = sigma

        loss = denoising_score_matching(model, xb, tb, sigma=sig)
        loss = loss + cfg.alpha_control * control_energy(model(xb, tb))

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())

    model.eval()
    if verbose:
        print(f"Training complete. Final loss: {losses[-1]:.4f}")

    # ── Jacobian tensor ────────────────────────────────────────────────────
    if verbose:
        print("Computing Jacobian tensor...")

    windows   = _build_windows(T_np, n_windows=n_windows, overlap=overlap)
    J_tensor  = torch.zeros(n_windows, D, D)
    t_centers = np.zeros(n_windows, dtype=np.float32)

    iter_j = trange(n_windows, desc="Jacobians", leave=False) if verbose else range(n_windows)
    for i in iter_j:
        lo, hi, idx_w = windows[i]
        t_centers[i] = (lo + hi) / 2.0
        xb = X[idx_w].to(device)
        tb = T[idx_w].to(device)
        with torch.enable_grad():
            Jb = model.jacobian(xb, tb)   # (n_cells, D, D) — average operators
        J_tensor[i] = Jb.detach().cpu().mean(dim=0)

    # Gaussian smoothing along time axis
    J_np = gaussian_filter1d(J_tensor.numpy(), sigma=smooth_sigma, axis=0)

    if verbose:
        print("Running archetype decomposition (semi-NMF)...")

    # ── Archetype decomposition ────────────────────────────────────────────
    J_smooth    = torch.from_numpy(J_np)
    patterns, activations, recon_err = jacobian_modes(
        J_smooth, rank=n_archetypes, n_restarts=n_restarts, seed=seed
    )
    activations_np = activations.numpy()   # (T, K) — already ≥ 0

    # Variance explained (R²)
    total_sq = float((J_smooth.reshape(n_windows, -1) ** 2).sum())
    recon_sq = float(np.sum((J_np.reshape(n_windows, -1) -
                             (activations_np @ patterns.numpy().reshape(n_archetypes, -1))) ** 2))
    r2 = max(0.0, 1.0 - recon_sq / (total_sq + 1e-8))

    # Max real eigenvalue per window (local sensitivity)
    max_eig = np.array([
        float(np.real(np.linalg.eigvals(J_np[i])).max())
        for i in range(n_windows)
    ])

    # Normalize activations for downstream plotting
    act_norm = activations_np.copy()
    for k in range(n_archetypes):
        a = act_norm[:, k]
        act_norm[:, k] = a / (a.max() + 1e-8)

    # Sign convention: activations are already ≥ 0 from semi-NMF
    sign_flip = [1.0] * n_archetypes

    # Pairwise temporal correlations
    corr_mat = np.corrcoef(act_norm.T)   # (K, K)

    # ── Gene-space projection (per archetype) ─────────────────────────────
    gene_scores_dict = {}
    top_genes_dict   = {}
    pca_loadings     = None
    gene_names       = []
    if "PCs" in adata.varm:
        pca_loadings = adata.varm["PCs"].astype(np.float32)   # (n_genes, n_pcs)
        gene_names   = list(adata.var_names)
        gene_scores_dict = _gene_scores(patterns, pca_loadings, sign_flip)
        for k, gs in gene_scores_dict.items():
            ranked = np.argsort(np.abs(gs))[::-1]
            top_genes_dict[k] = [gene_names[j] for j in ranked[:50]]
            gene_scores_dict[k] = gs

    # ── Per-window instability gene scores ─────────────────────────────────
    # For each sensitive window (Re(λ_max) > 0), project the eigenvector
    # associated with max Re(λ) to gene space. Sign consistency is enforced
    # by aligning each window's eigenvector to the previous one.
    instability_scores  = np.zeros((n_windows, len(gene_names)), dtype=np.float32)
    instability_evecs   = np.zeros((n_windows, D), dtype=np.float32)
    prev_v = None
    for i in range(n_windows):
        if max_eig[i] <= 0 or pca_loadings is None:
            prev_v = None
            continue
        eigvals, eigvecs = np.linalg.eig(J_np[i])
        max_idx  = int(np.argmax(np.real(eigvals)))
        v        = np.real(eigvecs[:, max_idx])
        v        = v / (np.linalg.norm(v) + 1e-8)
        # Flip sign to be consistent with previous sensitive window
        if prev_v is not None and np.dot(v, prev_v) < 0:
            v = -v
        prev_v = v
        instability_evecs[i]  = v
        instability_scores[i] = pca_loadings @ v          # (n_genes,)

    # Global instability ranking: mean |score| across sensitive windows
    sensitive_mask = max_eig > 0
    if sensitive_mask.sum() > 0 and len(gene_names) > 0:
        mean_instab = np.abs(instability_scores[sensitive_mask]).mean(0)
        top_instab_idx   = np.argsort(mean_instab)[::-1]
        top_instab_genes = [gene_names[j] for j in top_instab_idx[:50]]
        top_instab_scores = mean_instab[top_instab_idx[:50]]
    else:
        top_instab_genes  = []
        top_instab_scores = np.array([])

    # Per-archetype instability genes: which genes drive instability
    # specifically when archetype k is dominant?
    arch_instab_genes = {}
    arch_instab_scores = {}
    for k in range(n_archetypes):
        # Windows where this archetype is in top quartile of activation
        thresh    = np.quantile(act_norm[:, k], 0.75)
        arch_mask = (act_norm[:, k] >= thresh) & sensitive_mask
        if arch_mask.sum() > 0 and len(gene_names) > 0:
            arch_mean = np.abs(instability_scores[arch_mask]).mean(0)
            ranked    = np.argsort(arch_mean)[::-1]
            arch_instab_genes[str(k)]  = [gene_names[j] for j in ranked[:50]]
            arch_instab_scores[str(k)] = arch_mean[ranked[:50]]
        else:
            arch_instab_genes[str(k)]  = []
            arch_instab_scores[str(k)] = np.array([])

    # ── Store results ──────────────────────────────────────────────────────
    adata.uns[key_added] = {
        "J_tensor":    J_np,
        "t_centers":   t_centers,
        "patterns":    patterns.numpy(),
        "activations": activations_np,
        "act_norm":    act_norm,
        "max_real_eig": max_eig,
        "corr_mat":    corr_mat,
        "gene_scores":  {str(k): v for k, v in gene_scores_dict.items()},
        "top_genes":    {str(k): v for k, v in top_genes_dict.items()},
        # instability-specific outputs
        "instability_scores":      instability_scores,    # (n_windows, n_genes)
        "top_instability_genes":   top_instab_genes,      # list[str], global ranking
        "top_instability_scores":  top_instab_scores,     # mean |score| per gene
        "arch_instability_genes":  arch_instab_genes,     # per-archetype ranking
        "arch_instability_scores": arch_instab_scores,
        "gene_names":  gene_names,
        "losses":      losses,
        "r2":          r2,
        "recon_err":   recon_err,
        "params": {
            "n_archetypes": n_archetypes, "n_windows": n_windows,
            "n_epochs": n_epochs, "hidden": hidden, "depth": depth,
            "beta": beta, "vel_scale": vel_scale, "vel_time_mode": vel_time_mode,
            "lr": lr, "sigma": sigma, "seed": seed, "rep": rep, "time_key": time_key,
        },
    }

    # Store model drift and pseudotime velocity for visualization
    with torch.no_grad():
        drift_all = model(X, T).cpu().numpy()
    adata.obsm["X_drift"]          = drift_all   # full model drift (50D)
    adata.obsm["X_velocity_pseudo"] = V_np        # pseudotime-gradient velocity (50D)

    if verbose:
        print(f"Done. R²={r2:.3f} | Archetypes stored in adata.uns['{key_added}']")

    return model


# ---------------------------------------------------------------------------
# Gene extraction utility
# ---------------------------------------------------------------------------

def get_instability_genes(
    adata,
    key: str = "scqdiff",
    n_genes: int = 20,
    activation_threshold: float = 0.5,
    sensitivity_threshold: float = 0.05,
    min_sensitive_fraction: float = 0.0,
    top_archetypes: Optional[int] = None,
) -> "pd.DataFrame":
    """
    Extract the top genes driving instability from each archetype,
    ranked by how much the archetype is associated with locally sensitive
    (positive-eigenvalue) pseudotime windows.

    An archetype is characterised as "instable" when it is strongly active
    (activation > ``activation_threshold``) during windows where the
    maximum real eigenvalue exceeds ``sensitivity_threshold``.

    Parameters
    ----------
    n_genes               : Top N genes to return per archetype.
    activation_threshold  : Minimum normalised activation to consider a
                            window "active" for that archetype.
    sensitivity_threshold : Minimum Re(λ_max) to consider a window sensitive.
    min_sensitive_fraction: Minimum fraction of active windows that must be
                            sensitive to include the archetype (0 = all).
    top_archetypes        : If set, return only the K most instable archetypes.

    Returns
    -------
    pandas.DataFrame with columns:
        archetype, mean_sensitivity, sensitive_fraction,
        peak_pseudotime, gene, instability_score, rank
    """
    import pandas as pd

    res         = adata.uns[key]
    eig         = res["max_real_eig"]          # (n_windows,)
    act_norm    = res["act_norm"]              # (n_windows, K)
    t_np        = res["t_centers"]
    gene_names  = res["gene_names"]
    instab_sc   = res["instability_scores"]    # (n_windows, n_genes)
    arch_genes  = res["arch_instability_genes"]
    arch_scores = res["arch_instability_scores"]

    if not gene_names:
        raise ValueError("No gene names. Re-run sqd.tl.fit_drift on adata with PCA loadings.")

    K = act_norm.shape[1]
    sens_mask = eig > sensitivity_threshold

    # ── Score each archetype by mean sensitivity during active windows ──────
    arch_stats = []
    for k in range(K):
        active_mask = act_norm[:, k] > activation_threshold
        n_active    = int(active_mask.sum())
        if n_active == 0:
            continue
        both_mask       = active_mask & sens_mask
        mean_sens       = float(eig[active_mask].mean())
        sens_frac       = float(both_mask.sum() / n_active)
        peak_pt         = float(t_np[np.argmax(act_norm[:, k])])

        if sens_frac < min_sensitive_fraction:
            continue

        arch_stats.append({
            "k":                  k,
            "archetype":          f"A{k+1}",
            "mean_sensitivity":   round(mean_sens, 4),
            "sensitive_fraction": round(sens_frac, 3),
            "peak_pseudotime":    round(peak_pt, 3),
        })

    if not arch_stats:
        raise ValueError("No archetypes passed the sensitivity filter. "
                         "Lower min_sensitive_fraction or sensitivity_threshold.")

    # Sort by mean sensitivity (most instable first)
    arch_stats.sort(key=lambda x: x["mean_sensitivity"], reverse=True)

    if top_archetypes is not None:
        arch_stats = arch_stats[:top_archetypes]

    # ── Build gene table ───────────────────────────────────────────────────
    rows = []
    for stat in arch_stats:
        k        = stat["k"]
        genes_k  = arch_genes.get(str(k), [])[:n_genes]
        scores_k = arch_scores.get(str(k), np.array([]))[:n_genes]

        for rank, (gene, score) in enumerate(zip(genes_k, scores_k), start=1):
            rows.append({
                "archetype":          stat["archetype"],
                "mean_sensitivity":   stat["mean_sensitivity"],
                "sensitive_fraction": stat["sensitive_fraction"],
                "peak_pseudotime":    stat["peak_pseudotime"],
                "gene":               gene,
                "instability_score":  round(float(score), 4),
                "rank":               rank,
            })

    df = pd.DataFrame(rows, columns=[
        "archetype", "mean_sensitivity", "sensitive_fraction",
        "peak_pseudotime", "gene", "instability_score", "rank",
    ])
    return df
