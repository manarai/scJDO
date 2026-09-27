"""
FOLLOWUP4 — full re-score of the bifurcation-saddle table on HEAD,
plus refit without spectral normalization as a secondary check.

Runs:
  (3a) `saddle_readouts.py` verbatim on current HEAD — R1/R2/R3/R4 for
       seed=42, edge saddle, plus scoring under the fixed robust-
       crossing detector at h ∈ {0.005, 0.01, 0.02, 0.04}.
  (3b) Cross-seed spot check (seeds 0, 1, 2) — variance.
  (4)  Refit ONE seed with spectral normalization DISABLED.  Compare
       aggregated λ curve minima.

  Audit note: writes a summary of every numeric quantity currently
  cited by the manuscript that (a) I have not yet verified reproduces
  on HEAD, (b) is stored only in a cached JSON — flags them for
  re-verification.
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
from crossing_robust import robust_crossing
from T2_lifted_truth import kernel_matrix_curve
from FOLLOWUP2_trained_probe import _build_adata, _per_cell_jacobians


def _sanitize(x):
    if isinstance(x, dict): return {k: _sanitize(v) for k, v in x.items()}
    if isinstance(x, list): return [_sanitize(v) for v in x]
    if isinstance(x, np.floating): return float(x)
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, np.ndarray): return x.tolist()
    return x


def _r1r2r3(model, Z_lat, tau):
    """Compute per-cell R1 (max Re λ), R2 (tr J), R3 (P⊥JP⊥) — matches
    the manuscript's `saddle_readouts.jacobian_readouts`."""
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


def _kernel_scalar(scalar_per_cell, tau, grid, bandwidth, n_eff_min=15.0):
    curve = np.full(grid.size, np.nan)
    for k, tc in enumerate(grid):
        w = np.exp(-0.5 * ((tau - tc) / bandwidth) ** 2)
        W = w.sum()
        if W < 1e-9: continue
        n_eff = W ** 2 / (w * w).sum()
        if n_eff < n_eff_min: continue
        curve[k] = float((w * scalar_per_cell).sum() / W)
    return curve


def _score_curve(curve, grid, tau_crit, scoring="argmax_interior"):
    if scoring == "argmax_interior":
        m = (grid >= 0.05) & (grid <= 0.95) & ~np.isnan(curve)
        if not m.any(): return {"tau_hat": float("nan"), "offset": float("nan"),
                                  "code": "no_valid"}
        idx = np.where(m)[0][np.argmax(curve[m])]
        t = float(grid[idx])
        return {"tau_hat": t, "offset": abs(t - tau_crit), "code": "argmax"}
    if scoring == "argmin_interior":
        m = (grid >= 0.05) & (grid <= 0.95) & ~np.isnan(curve)
        if not m.any(): return {"tau_hat": float("nan"), "offset": float("nan"),
                                  "code": "no_valid"}
        idx = np.where(m)[0][np.argmin(curve[m])]
        t = float(grid[idx])
        return {"tau_hat": t, "offset": abs(t - tau_crit), "code": "argmin"}
    if scoring == "cross_robust":
        t, code = robust_crossing(curve, grid, threshold=0.0, min_run=5)
        off = abs(t - tau_crit) if np.isfinite(t) else float("nan")
        return {"tau_hat": t, "offset": off, "code": code}
    raise ValueError(scoring)


def rescore_seed(seed, use_spec_norm=True, n_cells=1500, n_epochs=800, label=""):
    tau_crit = gt.TAU_CRIT
    adata, Z_lat, tau = _build_adata(seed=seed, n_cells=n_cells)
    from scjdo.tl import fit_drift
    t0 = time.time()
    if not use_spec_norm:
        model = fit_drift(adata, rep="X_pca", time_key="pseudotime",
            n_epochs=n_epochs, n_archetypes=4, n_eff_min=20.0, n_boot=10,
            grid_size=200, seed=seed, verbose=False,
            use_spectral_norm=False,
        )
    else:
        model = fit_drift(adata, rep="X_pca", time_key="pseudotime",
            n_epochs=n_epochs, n_archetypes=4, n_eff_min=20.0, n_boot=10,
            grid_size=200, seed=seed, verbose=False,
        )
    dt = time.time() - t0
    r2 = float(adata.uns["scjdo"]["r2"])
    lam_cached = np.asarray(adata.uns["scjdo"]["max_real_eig"])
    grid_ms = np.asarray(adata.uns["scjdo"]["t_centers"])
    h_ms = float(adata.uns["scjdo"].get("bandwidth", 0.02))

    print(f"  [{label} seed={seed}] fit_time={dt:.1f}s   r2={r2:.4f}   bw_auto={h_ms}")

    r1_pc, r2_pc, r3_pc = _r1r2r3(model, Z_lat, tau)

    # Aggregate on 200-pt grid at the manuscript's bandwidth
    grid = np.linspace(0.0, 1.0, 200)
    grid_dense = np.linspace(0.0, 1.0, 400)

    scored = {}
    for name, scal in [("R1", r1_pc), ("R2", r2_pc), ("R3", r3_pc)]:
        c_manuscript = _kernel_scalar(scal, tau, grid, bandwidth=h_ms)
        # Argmax interior (manuscript scoring)
        s_argmax = _score_curve(c_manuscript, grid, tau_crit, "argmax_interior")
        # Argmin (for R2 negative-divergence policy)
        s_argmin = _score_curve(c_manuscript, grid, tau_crit, "argmin_interior")
        # Fine-grained bandwidths + robust crossing on dense grid
        cross_by_h = {}
        for h in [0.005, 0.01, 0.02, 0.04]:
            cd = _kernel_scalar(scal, tau, grid_dense, bandwidth=h)
            cr = _score_curve(cd, grid_dense, tau_crit, "cross_robust")
            cr["min_lam"] = float(np.nanmin(cd))
            cr["max_lam"] = float(np.nanmax(cd))
            cross_by_h[h] = cr
        scored[name] = {
            "argmax_interior_manuscript_h": s_argmax,
            "argmin_interior_manuscript_h": s_argmin,
            "cross_robust_by_h": cross_by_h,
            "curve_min_manuscript_h": float(np.nanmin(c_manuscript)),
            "curve_max_manuscript_h": float(np.nanmax(c_manuscript)),
        }
    return {
        "seed": seed, "fit_time_s": dt, "r2": r2,
        "bandwidth_auto": h_ms,
        "scored": scored,
        "use_spec_norm": use_spec_norm,
    }


def main():
    out_dir = Path("/Users/terooatt/Downloads/scJDO/reproducibility")

    print("==== (3) Full re-score on HEAD ====")
    seeds_full = [42, 0, 1, 2]  # includes manuscript's SEED=42
    results = {"tau_crit": gt.TAU_CRIT}
    results["with_spec_norm"] = []
    for s in seeds_full:
        r = rescore_seed(s, use_spec_norm=True, label="std")
        results["with_spec_norm"].append(r)

    # Print R1 argmax and R1 robust cross at h=0.02 across seeds
    print("\n  R1 (max Re λ) summary:")
    print(f"  {'seed':>6s}  {'argmax @ h_auto':>18s}  {'robust @ h=0.02':>18s}  "
          f"{'min λ @ h=0.02':>16s}  {'code':>18s}")
    for r in results["with_spec_norm"]:
        r1 = r["scored"]["R1"]
        c02 = r1["cross_robust_by_h"][0.02]
        print(f"  {r['seed']:>6d}  {r1['argmax_interior_manuscript_h']['tau_hat']:18.4f}  "
              f"{c02['tau_hat']:18.4f}  {c02['min_lam']:16.4f}  {c02['code']:>18s}")

    print("\n==== (4) Refit ONE seed without spectral normalization ====")
    results["no_spec_norm"] = [
        rescore_seed(42, use_spec_norm=False, label="no-spec-norm"),
    ]
    r = results["no_spec_norm"][0]
    print(f"\n  R1 comparison (seed=42):")
    with_sn = results["with_spec_norm"][0]["scored"]["R1"]
    no_sn = r["scored"]["R1"]
    print(f"    {'condition':30s}  {'argmax':>10s}  {'min λ @ h=0.02':>18s}  {'code @ h=0.02':>18s}")
    print(f"    {'with spec_norm':30s}  {with_sn['argmax_interior_manuscript_h']['tau_hat']:10.4f}  "
          f"{with_sn['cross_robust_by_h'][0.02]['min_lam']:18.4f}  "
          f"{with_sn['cross_robust_by_h'][0.02]['code']:>18s}")
    print(f"    {'NO spec_norm':30s}  {no_sn['argmax_interior_manuscript_h']['tau_hat']:10.4f}  "
          f"{no_sn['cross_robust_by_h'][0.02]['min_lam']:18.4f}  "
          f"{no_sn['cross_robust_by_h'][0.02]['code']:>18s}")

    # Cached-only audit note
    results["audit_notes_cache_only_numbers"] = {
        "scratchpad_run/saddle_readouts_corrected.json": {
            "tau_R1": 0.8291,
            "reproducibility_on_HEAD": "FAILS",
            "note": "Cache from 2026-07-18 predates the commit that adds saddle_readouts.py (2026-07-26). Not reproducible from committed state.",
        },
        "scratchpad_run/saddle_readouts_midinterval.json": {
            "note": "Mid-domain cache; needs re-verification on HEAD alongside R1 rescore.",
        },
        "scratchpad_run/saddle_dense_sampling_corrected.json": {
            "note": "Dense-sampling cache; needs re-verification on HEAD alongside R1 rescore.",
        },
        "scratchpad_run/saddle_koopman_geometry.json": {
            "note": "Koopman-geometry table cited in Fig. S8. Not re-verified in this session.",
        },
    }

    (out_dir / "data" / "FOLLOWUP4_rescore.json").write_text(
        json.dumps(_sanitize(results), indent=2)
    )
    print(f"\nSaved: {out_dir}/data/FOLLOWUP4_rescore.json")


if __name__ == "__main__":
    main()
