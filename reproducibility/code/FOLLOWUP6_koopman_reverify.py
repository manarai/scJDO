"""
FOLLOWUP6 — cache-only re-verifications.

(A) saddle_koopman_geometry.json — Fig S8 numbers.  Report:
      - Whether cached R5/R6/R7/R8 reproduce on HEAD.
      - Rank-starvation diagnostic: window_half = 8 → 17 pairs; default
        reduced_rank for rank=5 is min(T-1, 4·5+8) = 28.  So the local
        EDMD problem is 28-D from 17 samples, under-determined by 1.6×.
      - Bootstrap variability by re-running with different seeds.

(B) saddle_readouts_midinterval.json — mid-domain τ_crit = 0.5.
(C) saddle_dense_sampling_corrected.json — dense sampling at τ_crit.
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


def _r1r2r3(model, Z_lat, tau):
    device = next(model.parameters()).device
    z = torch.tensor(Z_lat, dtype=torch.float32, device=device)
    t = torch.tensor(tau, dtype=torch.float32, device=device)
    B, D = z.shape
    r1 = np.zeros(B); r2 = np.zeros(B); r3 = np.zeros(B)
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
        r2[i] = float(np.trace(J).real)
        f = u.detach().cpu().numpy().reshape(-1)
        n = float(np.linalg.norm(f))
        if n < 1e-8:
            r3[i] = r1[i]
        else:
            fh = f / n
            P = np.eye(D) - np.outer(fh, fh)
            Jperp = P @ J @ P
            r3[i] = float(np.real(np.linalg.eigvals(Jperp)).max())
    return r1, r2, r3


def _kernel(scalar, tau, grid, h, n_eff_min=15.0):
    out = np.full(grid.size, np.nan)
    for k, tc in enumerate(grid):
        w = np.exp(-0.5 * ((tau - tc) / h) ** 2)
        W = w.sum()
        if W < 1e-9: continue
        n_eff = W ** 2 / (w * w).sum()
        if n_eff < n_eff_min: continue
        out[k] = float((w * scalar).sum() / W)
    return out


def _interior_argmax(curve, grid, low=0.05, high=0.95):
    m = (grid >= low) & (grid <= high) & ~np.isnan(curve)
    if not m.any(): return float("nan")
    return float(grid[np.where(m)[0][np.argmax(curve[m])]])


def _koopman_geometry(J_tensor, window_halves=(4, 8, 12, 16)):
    """
    Compute Henrici, reactivity, transient gain, eigvec_cond descriptors on
    a sliding-window EDMD of the vectorised J trajectory. Also record the
    rank-starvation ratio (reduced_rank / n_pairs_in_window).
    """
    from scjdo.archetypes.koopman import _local_koopman
    T, D, _ = J_tensor.shape
    Z = J_tensor.reshape(T, -1)   # (T, D*D)
    # Reduce via SVD; use the same defaults as archetypes.koopman:
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    rank = 5
    reduced_rank = min(T - 1, 4 * rank + 8)
    Zr = Z @ Vt[:reduced_rank].T   # (T, r)

    results = {}
    for wh in window_halves:
        grid_km = np.linspace(0.02, 0.98, T)
        try:
            K_seq, dt_seq = _local_koopman(Zr, grid_km, window_half=wh, ridge=1e-4)
        except Exception as e:
            results[wh] = {"error": str(e)}
            continue
        # Descriptors per window operator
        henrici_per_t = []
        react_per_t = []
        gain_per_t = []
        eigcond_per_t = []
        for K in K_seq:
            # Henrici departure from normality: ‖K‖_F² − Σ|λ|²
            eig = np.linalg.eigvals(K)
            h_val = float(np.linalg.norm(K, 'fro') ** 2 - np.sum(np.abs(eig) ** 2))
            henrici_per_t.append(max(0.0, h_val) ** 0.5)
            # Reactivity: largest eig of (K + K^T)/2
            sym = 0.5 * (K + K.T)
            react_per_t.append(float(np.linalg.eigvalsh(sym).max()))
            # Transient gain over 30 steps: ‖K^30‖_2 / ρ(K)^30
            n_step = 30
            Kn = np.linalg.matrix_power(K, n_step)
            rho = np.max(np.abs(eig))
            gain_per_t.append(float(np.linalg.norm(Kn, 2) / max(rho ** n_step, 1e-30)))
            # Eigvec conditioning
            try:
                w, V = np.linalg.eig(K)
                cond = float(np.linalg.cond(V))
            except Exception:
                cond = float("nan")
            eigcond_per_t.append(cond)
        results[wh] = {
            "reduced_rank": int(reduced_rank),
            "n_pairs_in_window": int(2 * wh + 1),
            "rank_starvation_ratio": float(reduced_rank / (2 * wh + 1)),
            "henrici": np.array(henrici_per_t),
            "reactivity": np.array(react_per_t),
            "transient_gain": np.array(gain_per_t),
            "eigvec_cond": np.array(eigcond_per_t),
            "grid": grid_km,
        }
    return results


def reverify_koopman(seeds=(42,)):
    from FOLLOWUP2_trained_probe import _build_adata
    from scjdo.tl import fit_drift
    out_dir = REPO / "reproducibility" / "data"
    results = {}
    for seed in seeds:
        print(f"\n== Koopman re-verify, seed={seed} ==")
        adata, Z, tau = _build_adata(seed=seed)
        t0 = time.time()
        model = fit_drift(adata, rep="X_pca", time_key="pseudotime",
                           n_epochs=800, n_archetypes=4, n_eff_min=20.0, n_boot=10,
                           grid_size=200, seed=seed, verbose=False)
        print(f"   fit_time={time.time()-t0:.1f}s   R²={float(adata.uns['scjdo']['r2']):.4f}")

        # Use the J_tensor stored by scjdo
        J_tensor = np.asarray(adata.uns["scjdo"]["J_tensor"])
        print(f"   J_tensor shape = {J_tensor.shape}")

        km = _koopman_geometry(J_tensor, window_halves=(4, 8, 12, 16))
        # Extract argmax τ for each descriptor at the default window_half=8
        seed_row = {}
        for wh, r in km.items():
            if "error" in r: continue
            grid = r["grid"]
            argmax = {
                "henrici_argmax": _interior_argmax(r["henrici"], grid),
                "reactivity_argmax": _interior_argmax(r["reactivity"], grid),
                "transient_gain_argmax": _interior_argmax(r["transient_gain"], grid),
                "eigvec_cond_argmax": _interior_argmax(r["eigvec_cond"], grid),
                "rank_starvation_ratio": r["rank_starvation_ratio"],
                "reduced_rank": r["reduced_rank"],
                "n_pairs_in_window": r["n_pairs_in_window"],
            }
            seed_row[f"wh={wh}"] = argmax
            print(f"   window_half={wh:>2d}  n_pairs={r['n_pairs_in_window']:>2d}  "
                  f"reduced_r={r['reduced_rank']:>2d}  rank/n={r['rank_starvation_ratio']:.2f}")
            print(f"      Henrici argmax = {argmax['henrici_argmax']:.4f}")
            print(f"      Reactivity argmax = {argmax['reactivity_argmax']:.4f}")
            print(f"      Transient gain argmax = {argmax['transient_gain_argmax']:.4f}")
            print(f"      Eigvec cond argmax = {argmax['eigvec_cond_argmax']:.4f}")
        results[seed] = seed_row
    return results


def reverify_midinterval():
    """Rerun the mid-domain (τ_crit = 0.5) fit — takes ~30-40s."""
    from FOLLOWUP2_trained_probe import _per_cell_jacobians
    from scjdo.tl import fit_drift
    import anndata as ad
    import toggle_truth as gt

    # Set up mid-domain saddle: α_min=0.023, α_max=2.003 so τ_crit=0.5.
    alpha_min = 0.023114247535456656
    alpha_max = 2.0031142475354566

    def _make_mid_adata(seed=300, n_cells=1500):
        from sklearn.decomposition import PCA
        rng = np.random.default_rng(seed)
        # Simulate with the shifted α range
        # Reuse gt.simulate_v3 by monkey-patching alpha params: but simpler:
        # implement a mini version here matching the shifted range.
        dt = 0.02; T_total = 10.0
        n_steps = int(np.ceil(T_total / dt))
        target_step = np.sort(rng.integers(1, n_steps + 1, size=n_cells))
        t_per_cell = (target_step * dt) / T_total
        z = np.full((n_cells, 2), gt.sym_fp(alpha_min), dtype=np.float32)
        z += 0.02 * rng.standard_normal(z.shape).astype(np.float32)
        z_out = np.zeros_like(z)
        sqrt_dt = float(np.sqrt(dt))
        idx = 0
        for step in range(1, n_steps + 1):
            a_t = alpha_min + (alpha_max - alpha_min) * (step * dt) / T_total
            z = z + gt.toggle_drift(z.astype(np.float64), a_t).astype(np.float32) * dt \
                + gt.SIGMA_SDE * sqrt_dt * rng.standard_normal(z.shape).astype(np.float32)
            while idx < n_cells and target_step[idx] == step:
                z_out[idx] = z[idx]
                idx += 1
        z2 = z_out
        W = gt.build_observation_map(seed=seed)
        rng_g = np.random.default_rng(seed)
        X_obs = z2 @ W + gt.NOISE_SIGMA * rng_g.standard_normal((z2.shape[0], W.shape[1])).astype(np.float32)
        pca = PCA(n_components=20, random_state=seed)
        Z_lat = pca.fit_transform(X_obs).astype(np.float32)
        a = ad.AnnData(X=X_obs); a.obs_names = [f"c{i:05d}" for i in range(a.n_obs)]
        a.obsm["X_pca"] = Z_lat; a.obs["pseudotime"] = t_per_cell.astype(np.float32)
        return a, Z_lat, a.obs["pseudotime"].to_numpy().astype(np.float32)

    print("\n== Mid-interval re-verify (τ_crit = 0.5 by construction) ==")
    results = []
    for seed in [300, 301]:
        adata, Z, tau = _make_mid_adata(seed=seed)
        model = fit_drift(adata, rep="X_pca", time_key="pseudotime",
                          n_epochs=800, n_archetypes=4, n_eff_min=20.0, n_boot=10,
                          grid_size=200, seed=seed, verbose=False)
        r1_pc, r2_pc, r3_pc = _r1r2r3(model, Z, tau)
        grid = np.linspace(0.0, 1.0, 400)
        h = adata.uns["scjdo"]["bandwidth"]
        r1_curve = _kernel(r1_pc, tau, grid, h=float(h))
        r2_curve = _kernel(r2_pc, tau, grid, h=float(h))
        r3_curve = _kernel(r3_pc, tau, grid, h=float(h))
        argmax_r1 = _interior_argmax(r1_curve, grid)
        argmax_r2p = _interior_argmax(r2_curve, grid)
        # Also the argmin for R2 negative side
        m = (grid >= 0.05) & (grid <= 0.95) & ~np.isnan(r2_curve)
        argmax_r2n = float(grid[np.where(m)[0][np.argmin(r2_curve[m])]])
        argmax_r3 = _interior_argmax(r3_curve, grid)
        results.append({"seed": seed,
                         "argmax_r1": argmax_r1,
                         "argmax_r2_pos": argmax_r2p,
                         "argmax_r2_neg": argmax_r2n,
                         "argmax_r3": argmax_r3})
        print(f"   seed={seed}  argmax R1={argmax_r1:.4f}  R2_pos={argmax_r2p:.4f}  "
              f"R2_neg={argmax_r2n:.4f}  R3={argmax_r3:.4f}   (τ_crit=0.5)")
    return results


def main():
    out_dir = REPO / "reproducibility"

    print("=" * 74)
    print("(A) Koopman geometry re-verify — Fig S8 numbers")
    print("=" * 74)
    km_results = reverify_koopman(seeds=(42,))

    print()
    print("=" * 74)
    print("(B) Mid-interval re-verify — cached tau_crit=0.5")
    print("=" * 74)
    mid_results = reverify_midinterval()

    all_out = {
        "koopman": km_results,
        "midinterval": mid_results,
    }
    def _san(x):
        if isinstance(x, dict): return {k: _san(v) for k, v in x.items()}
        if isinstance(x, list): return [_san(v) for v in x]
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.floating): return float(x)
        if isinstance(x, np.integer): return int(x)
        return x
    (out_dir / "data" / "FOLLOWUP6_koopman_reverify.json").write_text(
        json.dumps(_san(all_out), indent=2)
    )
    print(f"\nSaved: {out_dir}/data/FOLLOWUP6_koopman_reverify.json")


if __name__ == "__main__":
    main()
