"""
FOLLOWUP2 (5) — single-seed trained-model probe of the fixes.

Fits ONE scJDO drift on the baseline synthetic dataset (edge saddle,
n_cells=1500, n_epochs=800, seed=42 — same as the manuscript). Then
scores the aggregated λ curve under

  A. manuscript-default:  h = 0.04,  argmax on [0.05, 0.95].
  B. E-fix:               h = 0.01,  robust crossing (min_run = 5).

Answers the single question: does the robust detector + adapted
bandwidth alone recover τ_crit on a trained drift, or does the neural
drift's density-dominance override even the improved estimator?
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
from scjdo.tl import fit_drift
import anndata as ad
from sklearn.decomposition import PCA
from T2_lifted_truth import kernel_matrix_curve
from crossing_robust import robust_crossing


def _build_adata(seed=42, n_cells=1500, d_genes=200, n_latent=20, noise_sigma=0.3):
    z, tau = gt.simulate_v3(n_cells=n_cells, seed=seed)
    W = gt.build_observation_map(seed=seed, d_genes=d_genes)
    rng_g = np.random.default_rng(seed)
    X_obs = z @ W
    X_obs = X_obs + noise_sigma * rng_g.standard_normal(X_obs.shape).astype(np.float32)
    pca = PCA(n_components=n_latent, random_state=seed)
    Z_lat = pca.fit_transform(X_obs).astype(np.float32)
    a = ad.AnnData(X=X_obs)
    a.obs_names = [f"c{i:05d}" for i in range(a.n_obs)]
    a.obsm["X_pca"] = Z_lat
    a.obs["pseudotime"] = tau
    return a, Z_lat, tau


def _per_cell_jacobians(model, Z_lat, tau):
    device = next(model.parameters()).device
    z = torch.tensor(Z_lat, dtype=torch.float32, device=device)
    t = torch.tensor(tau, dtype=torch.float32, device=device)
    B, D = z.shape
    Js = np.zeros((B, D, D), dtype=np.float64)
    for i in range(B):
        xi = z[i:i + 1].detach().requires_grad_(True)
        ti = t[i:i + 1]
        u = model(xi, ti)
        rows = []
        for j in range(D):
            g = torch.autograd.grad(u[0, j], xi, retain_graph=True, create_graph=False)[0][0]
            rows.append(g.detach().cpu().numpy())
        Js[i] = np.stack(rows, axis=0)
    return Js


def _argmax_interior(curve, grid, low=0.05, high=0.95):
    m = (grid >= low) & (grid <= high) & ~np.isnan(curve)
    if not m.any():
        return float("nan")
    return float(grid[np.where(m)[0][np.argmax(curve[m])]])


def main():
    out_dir = Path("/Users/terooatt/Downloads/scJDO/reproducibility")
    tau_crit = gt.TAU_CRIT
    print(f"τ_crit = {tau_crit:.5f}")

    print("Building AnnData …")
    adata, Z_lat, tau = _build_adata(seed=42)
    print(f"cells={adata.n_obs}  latent={Z_lat.shape[1]}")

    print("Fitting drift  (~5-10 min) …")
    t0 = time.time()
    model = fit_drift(
        adata, rep="X_pca", time_key="pseudotime",
        n_epochs=800, n_archetypes=4, n_eff_min=20.0, n_boot=10,
        grid_size=200, seed=42, verbose=False,
    )
    r2 = float(adata.uns["scjdo"]["r2"])
    print(f"  done in {time.time() - t0:.1f}s   R² = {r2:.3f}")

    print("Computing per-cell Jacobians …")
    J_pc = _per_cell_jacobians(model, Z_lat, tau)

    # Baseline aggregation (manuscript defaults from adata.uns)
    grid_ms = np.asarray(adata.uns["scjdo"]["t_centers"])
    lam_ms = np.asarray(adata.uns["scjdo"]["max_real_eig"])
    bw_ms = adata.uns["scjdo"].get("bandwidth", 0.04)
    tau_argmax_ms = _argmax_interior(lam_ms, grid_ms)
    tau_cross_robust_ms, code_ms = robust_crossing(lam_ms, grid_ms, threshold=0.0, min_run=5)
    print(f"\nManuscript defaults  (h={bw_ms}, argmax on interior):")
    print(f"  argmax(Re λ):  τ̂ = {tau_argmax_ms:.4f}  offset = {abs(tau_argmax_ms - tau_crit):.4f}")
    print(f"  robust cross: τ̂ = {tau_cross_robust_ms:.4f} ({code_ms})  "
          f"offset = {abs(tau_cross_robust_ms - tau_crit):.4f}")

    # Fix: h = 0.01, robust crossing on 400-point grid
    grid = np.linspace(0.0, 1.0, 400)
    Jbar, _ = kernel_matrix_curve(J_pc, tau, grid, bandwidth=0.01, n_eff_min=15.0)
    lam = np.array([float(np.real(np.linalg.eigvals(J)).max())
                     if not np.isnan(J).any() else np.nan for J in Jbar])
    tau_cross_fix, code_fix = robust_crossing(lam, grid, threshold=0.0, min_run=5)
    tau_argmax_fix = _argmax_interior(lam, grid)
    print(f"\nE-fix (h=0.01, robust cross, min_run=5):")
    print(f"  argmax(Re λ):  τ̂ = {tau_argmax_fix:.4f}  offset = {abs(tau_argmax_fix - tau_crit):.4f}")
    print(f"  robust cross: τ̂ = {tau_cross_fix:.4f} ({code_fix})  "
          f"offset = {abs(tau_cross_fix - tau_crit):.4f}")

    # Also with an even smaller bandwidth
    Jbar2, _ = kernel_matrix_curve(J_pc, tau, grid, bandwidth=0.005, n_eff_min=10.0)
    lam2 = np.array([float(np.real(np.linalg.eigvals(J)).max())
                      if not np.isnan(J).any() else np.nan for J in Jbar2])
    tau_cross_fix2, code_fix2 = robust_crossing(lam2, grid, threshold=0.0, min_run=5)
    print(f"\nE-fix++ (h=0.005, robust cross, min_run=5):")
    print(f"  robust cross: τ̂ = {tau_cross_fix2:.4f} ({code_fix2})  "
          f"offset = {abs(tau_cross_fix2 - tau_crit):.4f}")

    out = {
        "tau_crit": tau_crit,
        "grid_step": float(grid[1] - grid[0]),
        "r2": r2,
        "manuscript": {
            "bandwidth": float(bw_ms) if bw_ms is not None else None,
            "tau_argmax": tau_argmax_ms,
            "argmax_offset": abs(tau_argmax_ms - tau_crit),
            "tau_robust_crossing": tau_cross_robust_ms, "robust_code": code_ms,
        },
        "e_fix_h001": {
            "bandwidth": 0.01,
            "tau_argmax": tau_argmax_fix,
            "argmax_offset": abs(tau_argmax_fix - tau_crit),
            "tau_robust_crossing": tau_cross_fix, "robust_code": code_fix,
            "robust_offset": abs(tau_cross_fix - tau_crit)
                if np.isfinite(tau_cross_fix) else float("nan"),
        },
        "e_fix_h0005": {
            "bandwidth": 0.005,
            "tau_robust_crossing": tau_cross_fix2, "robust_code": code_fix2,
            "robust_offset": abs(tau_cross_fix2 - tau_crit)
                if np.isfinite(tau_cross_fix2) else float("nan"),
        },
    }
    (out_dir / "data" / "FOLLOWUP2_trained_probe.json").write_text(
        json.dumps(out, indent=2, default=lambda x: float(x))
    )
    # Save curves
    np.savez_compressed(
        out_dir / "data" / "FOLLOWUP2_trained_probe.npz",
        grid_ms=grid_ms, lam_ms=lam_ms, grid=grid, lam=lam, lam2=lam2,
        tau_crit=tau_crit,
    )
    print(f"\nSaved: {out_dir}/data/FOLLOWUP2_trained_probe.json")


if __name__ == "__main__":
    main()
