"""
Scale-stability bandwidth selection.

Idea
----
Sweep bandwidth h.  For each h, compute the aggregated λ curve and the
detector output (τ̂_c or crossing verdict).  For h small enough, the
estimator is noise-limited; for h large enough, it is bias-limited.
In between there should be a "plateau" where the estimate is stable
across a range of h.  Report:

  (a) The plateau interval  [h_lo, h_hi]  where τ̂_c varies by less
      than a target tolerance across neighboring h values.
  (b) The plateau τ̂_c estimate.
  (c) Compare to `τ_crit` — does this select a bandwidth that recovers
      τ_crit *without* knowing τ_crit as an input?

Method
------
For a grid of bandwidths spanning three decades:
  1. Compute the aggregated λ(τ) using kernel_matrix_curve.
  2. Detect the crossing with the robust_crossing (fixed) detector.
  3. Log τ̂_c(h), the verdict code, and the run-length of the pre-neg
     region.

Then look for a run of consecutive bandwidths where τ̂_c is finite
and its variation is below a small threshold (say, one grid step).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import toggle_truth as gt
from T2_lifted_truth import kernel_matrix_curve
from crossing_robust import robust_crossing


def scale_stability_sweep(J_pc, tau, n_cells,
                           bandwidths=None, grid_size=400, min_run=5):
    """
    Sweep bandwidth h; detect crossing at each.  Return per-h summary.
    """
    if bandwidths is None:
        bandwidths = np.logspace(-3, -0.5, 20)   # 0.001 → 0.316
    grid = np.linspace(0.0, 1.0, grid_size)
    rows = []
    for h in bandwidths:
        Jbar, n_eff = kernel_matrix_curve(J_pc, tau, grid,
                                          bandwidth=float(h), n_eff_min=15.0)
        lam = np.array([float(np.real(np.linalg.eigvals(J)).max())
                         if not np.isnan(J).any() else np.nan for J in Jbar])
        tau_hat, code = robust_crossing(lam, grid, threshold=0.0, min_run=min_run)
        min_l = float(np.nanmin(lam))
        max_l = float(np.nanmax(lam))
        rows.append({
            "h": float(h), "tau_hat": tau_hat, "code": code,
            "min_lam": min_l, "max_lam": max_l,
        })
    return rows


def find_plateau(rows, tolerance=0.010):
    """
    Identify the longest run of consecutive bandwidths where all τ̂_c
    are finite and their max−min < tolerance.  Return (h_lo, h_hi,
    tau_hat_plateau, plateau_length).
    """
    n = len(rows)
    finite = np.array([np.isfinite(r["tau_hat"]) for r in rows])
    if not finite.any():
        return None
    best = None
    for i in range(n):
        if not finite[i]:
            continue
        for j in range(i, n):
            if not finite[j]:
                break
            values = np.array([rows[k]["tau_hat"] for k in range(i, j + 1)
                                if finite[k]])
            if values.size == 0:
                continue
            spread = values.max() - values.min()
            if spread <= tolerance:
                candidate = (i, j, spread, values.mean(), values.size)
                if best is None or candidate[4] > best[4]:
                    best = candidate
            else:
                break
    if best is None:
        return None
    i, j, spread, plateau_tau, plateau_len = best
    return {
        "h_lo": rows[i]["h"], "h_hi": rows[j]["h"],
        "plateau_tau": float(plateau_tau), "plateau_length": int(plateau_len),
        "spread_within_plateau": float(spread),
    }


def _build_J_pc(seed, n_cells=3000):
    """Lifted-truth Jacobians per cell."""
    z, tau = gt.simulate_v3(n_cells=n_cells, seed=seed)
    alphas_pc = gt.alpha_of_tau(tau)
    J_pc = np.stack([gt.toggle_jacobian_at(float(zi[0]), float(zi[1]), float(a))
                      for zi, a in zip(z, alphas_pc)])
    return J_pc, tau


def _build_J_pc_trained(seed, n_cells=1500):
    """Trained-model per-cell Jacobians (uses fit_drift on synthetic data)."""
    import sys, time
    sys.path.insert(0, "/Users/terooatt/Downloads/scJDO")
    from FOLLOWUP2_trained_probe import _build_adata, _per_cell_jacobians
    from scjdo.tl import fit_drift
    adata, Z_lat, tau = _build_adata(seed=seed, n_cells=n_cells)
    t0 = time.time()
    model = fit_drift(adata, rep="X_pca", time_key="pseudotime",
                       n_epochs=800, n_archetypes=4, n_eff_min=20.0, n_boot=10,
                       grid_size=200, seed=seed, verbose=False)
    print(f"  trained fit for seed={seed} in {time.time()-t0:.1f}s")
    J_pc = _per_cell_jacobians(model, Z_lat, tau)
    return J_pc, tau


def main():
    out_dir = Path("/Users/terooatt/Downloads/scJDO/reproducibility")
    bandwidths = np.logspace(-3, -0.5, 20)   # 0.001 → 0.316
    tau_crit = gt.TAU_CRIT

    print("Scale-stability sweep on LIFTED TRUTH (5 seeds):")
    lifted_results = {}
    for seed in [42, 1, 2, 3, 4]:
        J_pc, tau = _build_J_pc(seed=seed, n_cells=3000)
        rows = scale_stability_sweep(J_pc, tau, n_cells=3000,
                                      bandwidths=bandwidths, min_run=5)
        plateau = find_plateau(rows, tolerance=0.010)
        lifted_results[seed] = {"rows": rows, "plateau": plateau}
        print(f"\n  seed={seed}:")
        for r in rows:
            print(f"     h={r['h']:.4f}  τ̂={r['tau_hat']}  ({r['code']:20s})  "
                  f"min λ={r['min_lam']:+.4f}")
        if plateau is not None:
            offset = abs(plateau["plateau_tau"] - tau_crit)
            print(f"     plateau: h ∈ [{plateau['h_lo']:.4f}, {plateau['h_hi']:.4f}]  "
                  f"τ̂ = {plateau['plateau_tau']:.4f}  offset = {offset:.4f}  "
                  f"(length={plateau['plateau_length']})")
        else:
            print("     NO plateau of finite τ̂ found across bandwidths.")

    print("\n" + "=" * 70)
    print("Scale-stability sweep on TRAINED MODEL (2 seeds):")
    trained_results = {}
    for seed in [42, 1]:
        try:
            J_pc, tau = _build_J_pc_trained(seed=seed, n_cells=1500)
        except Exception as e:
            print(f"  seed={seed}: skipped ({e})")
            continue
        rows = scale_stability_sweep(J_pc, tau, n_cells=1500,
                                      bandwidths=bandwidths, min_run=5)
        plateau = find_plateau(rows, tolerance=0.010)
        trained_results[seed] = {"rows": rows, "plateau": plateau}
        print(f"\n  seed={seed}:")
        for r in rows:
            print(f"     h={r['h']:.4f}  τ̂={r['tau_hat']}  ({r['code']:20s})  "
                  f"min λ={r['min_lam']:+.4f}")
        if plateau is not None:
            offset = abs(plateau["plateau_tau"] - tau_crit)
            print(f"     plateau: h ∈ [{plateau['h_lo']:.4f}, {plateau['h_hi']:.4f}]  "
                  f"τ̂ = {plateau['plateau_tau']:.4f}  offset = {offset:.4f}  "
                  f"(length={plateau['plateau_length']})")
        else:
            print("     NO plateau of finite τ̂ found across bandwidths.")

    # Save
    def _clean(x):
        if isinstance(x, dict): return {k: _clean(v) for k, v in x.items()}
        if isinstance(x, list): return [_clean(v) for v in x]
        if isinstance(x, np.floating): return float(x)
        if isinstance(x, np.integer): return int(x)
        return x
    out = {
        "tau_crit": tau_crit,
        "bandwidths": bandwidths.tolist(),
        "lifted_truth_seeds": {str(k): v for k, v in _clean(lifted_results).items()},
        "trained_seeds": {str(k): v for k, v in _clean(trained_results).items()},
    }
    (out_dir / "data" / "scale_stability.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_dir}/data/scale_stability.json")


if __name__ == "__main__":
    main()
