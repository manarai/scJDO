"""
FOLLOWUP4 (#1) — Cosine-similarity discriminator.

Per cell:
   f_train(z_lat_i, τ_i)        — trained neural drift, (1, D_lat)
   f_lifted(z_lat_i, τ_i)       — analytic 2-D drift at cell's actual (x_i, y_i, α_i),
                                   lifted to D_lat via  f_2 @ A
   cos_i = <f_train_i, f_lifted_i> / (||f_train_i|| · ||f_lifted_i||)
   mag_i = ||f_train_i|| / ||f_lifted_i||

Aggregate onto a τ-grid via the same Gaussian kernel used everywhere.
This produces a three-way discriminator over τ:

  1. **cos → 1**   trained drift is direction-correct
  2. **mag ratio** magnitude relative to analytic ground truth
  3. **|f_train − f_lifted|**   raw disagreement

Where in τ does the trained drift DIVERGE from truth? Predictions per
hypothesis:
  - E (bias in Jacobian aggregation, drift itself is close to truth):
        cos ≈ 1 everywhere; the failure is in the aggregation, not the field.
  - D (density-dominance in generalization):
        cos → 1 for high-density τ, drops for low-density τ (near saddle).
  - Something else: some other τ-signature.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path("/Users/terooatt/Downloads/scJDO")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "reproducibility" / "code"))

import toggle_truth as gt
from FOLLOWUP2_trained_probe import _build_adata


def _sanitize(x):
    if isinstance(x, dict): return {k: _sanitize(v) for k, v in x.items()}
    if isinstance(x, list): return [_sanitize(v) for v in x]
    if isinstance(x, np.floating): return float(x)
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, np.ndarray): return x.tolist()
    return x


def cosine_probe(seed=42, n_cells=1500, n_epochs=800):
    """Fit fresh drift, compute per-cell (cos, mag ratio) between trained and analytic."""
    from scjdo.tl import fit_drift
    from sklearn.decomposition import PCA

    z2, tau = gt.simulate_v3(n_cells=n_cells, seed=seed)
    W = gt.build_observation_map(seed=seed)
    rng_g = np.random.default_rng(seed)
    X_obs = z2 @ W + gt.NOISE_SIGMA * rng_g.standard_normal(z2.shape[0] * W.shape[1]).reshape(z2.shape[0], W.shape[1]).astype(np.float32)
    pca = PCA(n_components=20, random_state=seed)
    Z_lat = pca.fit_transform(X_obs).astype(np.float32)
    mu = pca.mean_.astype(np.float32)   # (200,)
    V = pca.components_.astype(np.float32)   # (20, 200)

    import anndata as ad
    adata = ad.AnnData(X=X_obs)
    adata.obs_names = [f"c{i:05d}" for i in range(adata.n_obs)]
    adata.obsm["X_pca"] = Z_lat
    adata.obs["pseudotime"] = tau

    t0 = time.time()
    model = fit_drift(adata, rep="X_pca", time_key="pseudotime",
                       n_epochs=n_epochs, n_archetypes=4, n_eff_min=20.0,
                       n_boot=10, grid_size=200, seed=seed, verbose=False)
    print(f"  seed={seed} fit_time={time.time() - t0:.1f}s   R²={float(adata.uns['scjdo']['r2']):.4f}   "
          f"bw_auto={adata.uns['scjdo'].get('bandwidth')}")

    device = next(model.parameters()).device
    z_t = torch.tensor(Z_lat, dtype=torch.float32, device=device)
    t_t = torch.tensor(tau, dtype=torch.float32, device=device)
    with torch.no_grad():
        f_train = model(z_t, t_t).detach().cpu().numpy()   # (N, 20)

    alphas = gt.alpha_of_tau(tau)
    f2 = gt.toggle_drift(z2.astype(np.float64), alphas)   # (N, 2)

    # Lift the 2-D drift into the SAME latent space used by the trained model.
    # Encoder in row form: Z_lat = (X_obs - μ) @ V.T. So a small change dz2 in
    # 2-D produces dX = dz2 @ W, then dZ_lat = dz2 @ W @ V.T. So the lifted
    # drift vector is  f_lifted = f2 @ (W @ V.T).
    A_lift = (W @ V.T).astype(np.float64)   # (2, 20)
    f_lifted = f2 @ A_lift   # (N, 20)

    # Per-cell cosine similarity + magnitude ratio
    nu = np.linalg.norm(f_train, axis=1) + 1e-12
    nv = np.linalg.norm(f_lifted, axis=1) + 1e-12
    cos = (f_train * f_lifted).sum(1) / (nu * nv)
    mag_ratio = nu / nv
    dot = (f_train * f_lifted).sum(1)
    diff = np.linalg.norm(f_train - f_lifted, axis=1)

    return {
        "seed": seed, "tau": tau,
        "cos": cos, "mag_ratio": mag_ratio,
        "f_train_norm": nu, "f_lifted_norm": nv,
        "diff_norm": diff,
        "r2": float(adata.uns["scjdo"]["r2"]),
    }


def kernel_curve(scalar, tau, grid, bandwidth=0.02, n_eff_min=15.0):
    out = np.full(grid.size, np.nan)
    for k, tc in enumerate(grid):
        w = np.exp(-0.5 * ((tau - tc) / bandwidth) ** 2)
        W = w.sum()
        if W < 1e-9: continue
        if W ** 2 / (w * w).sum() < n_eff_min: continue
        out[k] = float((w * scalar).sum() / W)
    return out


def main():
    out_dir = Path("/Users/terooatt/Downloads/scJDO/reproducibility")

    grid = np.linspace(0.0, 1.0, 400)
    results = {"tau_crit": gt.TAU_CRIT, "grid": grid.tolist(), "seeds": []}
    print("Cosine discriminator across seeds:")
    for seed in [42, 0, 1, 2, 3]:
        r = cosine_probe(seed=seed)
        cos_curve = kernel_curve(r["cos"], r["tau"], grid)
        mag_curve = kernel_curve(np.log(np.maximum(r["mag_ratio"], 1e-6)),
                                  r["tau"], grid)  # log to compress
        diff_curve = kernel_curve(r["diff_norm"], r["tau"], grid)

        # Report values at four τ landmarks
        landmarks = {
            "tau=0.02 (pre-transition tail, close to τ_crit)": 0.02,
            "tau=τ_crit + 0.05 (early post-transition)": gt.TAU_CRIT + 0.05,
            "tau=0.3 (committed, moderate)": 0.3,
            "tau=0.8 (committed, deep)": 0.8,
        }
        for label, tv in landmarks.items():
            idx = int(np.round(tv / (grid[1] - grid[0])))
            idx = min(max(idx, 0), len(grid) - 1)
            print(f"    seed={seed}  {label:52s}  cos={cos_curve[idx]:+.3f}  "
                  f"log(mag)={mag_curve[idx]:+.2f}  |diff|={diff_curve[idx]:.3f}")

        results["seeds"].append({
            "seed": seed, "r2": r["r2"],
            "cos_curve": cos_curve, "mag_log_curve": mag_curve,
            "diff_curve": diff_curve,
            "cos_mean_all_tau": float(np.nanmean(cos_curve)),
            "cos_min_all_tau": float(np.nanmin(cos_curve)),
            "cos_at_taucrit": float(cos_curve[int(np.round(gt.TAU_CRIT / (grid[1] - grid[0])))]),
        })

    (out_dir / "data" / "FOLLOWUP4_cosine.json").write_text(
        json.dumps(_sanitize(results), indent=2)
    )
    print(f"\nSaved: {out_dir}/data/FOLLOWUP4_cosine.json")

    # Also save curves for plotting
    np.savez_compressed(
        out_dir / "data" / "FOLLOWUP4_cosine.npz",
        grid=grid,
        cos_curves=np.stack([s["cos_curve"] for s in results["seeds"]]),
        mag_log_curves=np.stack([s["mag_log_curve"] for s in results["seeds"]]),
        diff_curves=np.stack([s["diff_curve"] for s in results["seeds"]]),
        seeds=np.array([s["seed"] for s in results["seeds"]]),
        tau_crit=gt.TAU_CRIT,
    )

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for s in results["seeds"]:
        ax[0].plot(grid, s["cos_curve"], label=f"seed={s['seed']}", alpha=0.7)
        ax[1].plot(grid, s["mag_log_curve"], label=f"seed={s['seed']}", alpha=0.7)
        ax[2].plot(grid, s["diff_curve"], label=f"seed={s['seed']}", alpha=0.7)
    for a in ax:
        a.axvline(gt.TAU_CRIT, color="red", lw=0.8, ls="--")
    ax[0].axhline(1.0, color="grey", lw=0.5, ls=":")
    ax[1].axhline(0.0, color="grey", lw=0.5, ls=":")
    ax[0].set_xlabel(r"$\tau$"); ax[0].set_ylabel("cosine similarity")
    ax[0].set_title("cos( f_train, f_lifted_analytic )")
    ax[0].set_ylim(-1.05, 1.05); ax[0].legend(fontsize=7)
    ax[1].set_xlabel(r"$\tau$"); ax[1].set_ylabel(r"log( ||f_train|| / ||f_lifted|| )")
    ax[1].set_title("magnitude ratio (log scale)")
    ax[2].set_xlabel(r"$\tau$"); ax[2].set_ylabel(r"||f_train − f_lifted||")
    ax[2].set_title("raw disagreement")
    plt.tight_layout()
    fig.savefig(out_dir / "figures" / "FOLLOWUP4_cosine.pdf")
    fig.savefig(out_dir / "figures" / "FOLLOWUP4_cosine.png", dpi=150)
    print(f"       {out_dir}/figures/FOLLOWUP4_cosine.pdf")


if __name__ == "__main__":
    main()
