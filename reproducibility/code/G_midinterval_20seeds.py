"""
Mid-interval bimodal-argmax question at ≥ 20 seeds.

Independent of G-5..G-9. Reports the full seed-level distribution of R1
argmax on the mid-domain saddle (τ_crit = 0.5 by construction), so the
bimodal-outcome property surfaced in FOLLOWUP6 §5b can be characterised
directly rather than summarised by a median.
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


ALPHA_MIN_MID = 0.023114247535456656
ALPHA_MAX_MID = 2.0031142475354566


def _make_mid_adata(seed, n_cells=1500):
    import anndata as ad
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(seed)
    dt = 0.02; T_total = 10.0
    n_steps = int(np.ceil(T_total / dt))
    target_step = np.sort(rng.integers(1, n_steps + 1, size=n_cells))
    t_per_cell = (target_step * dt) / T_total
    z = np.full((n_cells, 2), gt.sym_fp(ALPHA_MIN_MID), dtype=np.float32)
    z += 0.02 * rng.standard_normal(z.shape).astype(np.float32)
    z_out = np.zeros_like(z)
    sqrt_dt = float(np.sqrt(dt))
    idx = 0
    for step in range(1, n_steps + 1):
        a_t = ALPHA_MIN_MID + (ALPHA_MAX_MID - ALPHA_MIN_MID) * (step * dt) / T_total
        z = z + gt.toggle_drift(z.astype(np.float64), a_t).astype(np.float32) * dt \
            + gt.SIGMA_SDE * sqrt_dt * rng.standard_normal(z.shape).astype(np.float32)
        while idx < n_cells and target_step[idx] == step:
            z_out[idx] = z[idx]
            idx += 1
    W = gt.build_observation_map(seed=seed)
    rng_g = np.random.default_rng(seed)
    X_obs = z_out @ W + gt.NOISE_SIGMA * rng_g.standard_normal((n_cells, W.shape[1])).astype(np.float32)
    pca = PCA(n_components=20, random_state=seed)
    Z_lat = pca.fit_transform(X_obs).astype(np.float32)
    a = ad.AnnData(X=X_obs); a.obs_names = [f"c{i:05d}" for i in range(a.n_obs)]
    a.obsm["X_pca"] = Z_lat; a.obs["pseudotime"] = t_per_cell.astype(np.float32)
    return a, Z_lat, a.obs["pseudotime"].to_numpy().astype(np.float32)


def _r1_per_cell(model, Z_lat, tau):
    device = next(model.parameters()).device
    z = torch.tensor(Z_lat, dtype=torch.float32, device=device)
    t = torch.tensor(tau, dtype=torch.float32, device=device)
    B, D = z.shape
    r1 = np.zeros(B)
    for i in range(B):
        xi = z[i:i + 1].detach().requires_grad_(True)
        ti = t[i:i + 1]
        u = model(xi, ti)
        rows = []
        for j in range(D):
            g = torch.autograd.grad(u[0, j], xi, retain_graph=True, create_graph=False)[0][0]
            rows.append(g.detach().cpu().numpy())
        J = np.stack(rows, axis=0)
        r1[i] = float(np.real(np.linalg.eigvals(J)).max())
    return r1


def _kernel(scal, tau, grid, h, n_eff_min=15.0):
    out = np.full(grid.size, np.nan)
    for k, tc in enumerate(grid):
        w = np.exp(-0.5 * ((tau - tc) / h) ** 2)
        W = w.sum()
        if W < 1e-9: continue
        if W ** 2 / (w * w).sum() < n_eff_min: continue
        out[k] = float((w * scal).sum() / W)
    return out


def _argmax_interior(curve, grid, low=0.05, high=0.95):
    m = (grid >= low) & (grid <= high) & ~np.isnan(curve)
    if not m.any(): return float("nan")
    return float(grid[np.where(m)[0][np.argmax(curve[m])]])


def main():
    from scjdo.tl import fit_drift
    out_dir = REPO / "reproducibility"
    tau_crit = 0.5

    rows = []
    for seed in range(300, 325):   # 25 seeds
        adata, Z, tau = _make_mid_adata(seed=seed)
        t0 = time.time()
        model = fit_drift(adata, rep="X_pca", time_key="pseudotime",
                           n_epochs=800, n_archetypes=4, n_eff_min=20.0, n_boot=10,
                           grid_size=200, seed=seed, verbose=False)
        dt = time.time() - t0
        # (A) matrix aggregation (scjdo cached)
        lam_matrix = np.asarray(adata.uns["scjdo"]["max_real_eig"])
        grid_matrix = np.asarray(adata.uns["scjdo"]["t_centers"])
        h_used = float(adata.uns["scjdo"].get("bandwidth", 0.02))
        argmax_matrix = _argmax_interior(lam_matrix, grid_matrix)
        # (B) per-cell aggregation (manuscript convention)
        r1_pc = _r1_per_cell(model, Z, tau)
        grid_pc = np.linspace(0, 1, 400)
        lam_pc = _kernel(r1_pc, tau, grid_pc, h=h_used)
        argmax_pc = _argmax_interior(lam_pc, grid_pc)
        rows.append({
            "seed": seed, "fit_s": dt,
            "r2": float(adata.uns["scjdo"]["r2"]),
            "h_auto": h_used,
            "argmax_matrix": argmax_matrix,
            "argmax_percell": argmax_pc,
        })
        print(f"  seed={seed}  fit={dt:.1f}s  h={h_used}  "
              f"argmax(matrix)={argmax_matrix:.4f}  argmax(percell)={argmax_pc:.4f}")

    # Distribution summary
    def _dist(vals):
        v = np.asarray(vals)
        return {"n": len(v), "mean": float(v.mean()), "median": float(np.median(v)),
                 "std": float(v.std(ddof=1)),
                 "p10": float(np.percentile(v, 10)),
                 "p25": float(np.percentile(v, 25)),
                 "p75": float(np.percentile(v, 75)),
                 "p90": float(np.percentile(v, 90)),
                 "frac_near_edge_lo": float((v < 0.15).mean()),
                 "frac_near_edge_hi": float((v > 0.85).mean()),
                 "frac_near_saddle": float((abs(v - tau_crit) <= 0.05).mean()),
                 "raw": v.tolist()}
    argmax_matrix = [r["argmax_matrix"] for r in rows]
    argmax_percell = [r["argmax_percell"] for r in rows]
    summary = {
        "tau_crit": tau_crit,
        "matrix_agg_argmax_distribution": _dist(argmax_matrix),
        "percell_agg_argmax_distribution": _dist(argmax_percell),
        "per_seed_rows": rows,
    }
    (out_dir / "data" / "G_midinterval_20seeds.json").write_text(
        json.dumps(summary, indent=2, default=lambda x: float(x))
    )
    print(f"\nSaved: {out_dir}/data/G_midinterval_20seeds.json")

    print("\n== Distribution (matrix-agg argmax) ==")
    d = summary["matrix_agg_argmax_distribution"]
    print(f"  n={d['n']}  mean={d['mean']:.3f}  median={d['median']:.3f}  std={d['std']:.3f}")
    print(f"  IQR=[{d['p25']:.3f}, {d['p75']:.3f}]  p10={d['p10']:.3f}  p90={d['p90']:.3f}")
    print(f"  frac<0.15 (low edge)={d['frac_near_edge_lo']:.2f}   "
          f"frac>0.85 (high edge)={d['frac_near_edge_hi']:.2f}   "
          f"frac within ±0.05 of τ_crit=0.5: {d['frac_near_saddle']:.2f}")

    print("\n== Distribution (per-cell argmax) ==")
    d = summary["percell_agg_argmax_distribution"]
    print(f"  n={d['n']}  mean={d['mean']:.3f}  median={d['median']:.3f}  std={d['std']:.3f}")
    print(f"  IQR=[{d['p25']:.3f}, {d['p75']:.3f}]  p10={d['p10']:.3f}  p90={d['p90']:.3f}")
    print(f"  frac<0.15={d['frac_near_edge_lo']:.2f}   frac>0.85={d['frac_near_edge_hi']:.2f}   "
          f"frac within ±0.05 of τ_crit=0.5: {d['frac_near_saddle']:.2f}")


if __name__ == "__main__":
    main()
