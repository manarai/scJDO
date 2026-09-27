"""
FOLLOWUP6 (#1) — Both aggregations on lifted truth, plus Var(J_i) gap.

The question: on lifted analytic truth (where the underlying drift is known),
does matrix-aggregation `Re λ_max(⟨J_i⟩)` pass the positive control that
per-cell aggregation `⟨Re λ_max(J_i)⟩` does not? If yes, the manuscript's
failure is a *convention* artifact, and the correction is about aggregation,
not about what the operator represents.

Measured:
  A. `lam_matrix(τ)` = Re λ_max of kernel-averaged J
  B. `lam_percell(τ)` = kernel-averaged Re λ_max(J_i)
  C. Var(Re λ_max(J_i)) within each τ window — the noise the per-cell average
      is smoothing over.  Also  ‖J_i − ⟨J⟩‖_F variance (matrix-level).

At the manuscript's analytic Jacobian (evaluated per cell at (x_i, y_i, α_i)),
we know matrix-agg should give the SAME curve as evaluating J at the mean
cell within each window (up to smoothing artifacts), while per-cell-agg
inflates the peak wherever J_i varies substantially across cells in the
window.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import toggle_truth as gt
from T2_lifted_truth import kernel_matrix_curve, kernel_curve
from crossing_robust import robust_crossing


def build_at_cell(seed=42, n_cells=3000):
    """Per-cell analytic Jacobians on the toggle switch, as in T2 at_cell."""
    z, tau = gt.simulate_v3(n_cells=n_cells, seed=seed)
    alphas = gt.alpha_of_tau(tau)
    J_pc = np.stack([gt.toggle_jacobian_at(float(zi[0]), float(zi[1]), float(a))
                      for zi, a in zip(z, alphas)])
    return z, tau, J_pc


def both_aggregations(J_pc, tau, grid, h=0.01, n_eff_min=15.0):
    """
    Returns:
      lam_matrix   : Re λ_max( ⟨J⟩_τ )
      lam_percell  : ⟨Re λ_max(J_i)⟩_τ
      var_percell  : Var_i(Re λ_max(J_i)) within each τ window
      frobvar_matrix : ⟨‖J_i − ⟨J⟩_τ‖_F²⟩_τ, the matrix-level spread
    """
    Jbar, _ = kernel_matrix_curve(J_pc, tau, grid, bandwidth=h, n_eff_min=n_eff_min)
    lam_matrix = np.array(
        [float(np.real(np.linalg.eigvals(J)).max()) if not np.isnan(J).any() else np.nan
         for J in Jbar]
    )
    # Per-cell λ scalar
    lam_pc = np.array([float(np.real(np.linalg.eigvals(J)).max()) for J in J_pc])
    lam_percell = kernel_curve(lam_pc, tau, grid, bandwidth=h, n_eff_min=n_eff_min)[0] \
        if False else _kernel_scalar(lam_pc, tau, grid, h, n_eff_min)
    # Variance of per-cell λ within kernel window
    var_percell = _kernel_variance(lam_pc, tau, grid, h, n_eff_min)
    # Matrix-level Frobenius variance
    frobvar = _matrix_frob_var(J_pc, Jbar, tau, grid, h, n_eff_min)
    return lam_matrix, lam_percell, var_percell, frobvar


def _kernel_scalar(scal, tau, grid, h, n_eff_min):
    out = np.full(grid.size, np.nan)
    for k, tc in enumerate(grid):
        w = np.exp(-0.5 * ((tau - tc) / h) ** 2)
        W = w.sum()
        if W < 1e-9: continue
        if W ** 2 / (w * w).sum() < n_eff_min: continue
        out[k] = float((w * scal).sum() / W)
    return out


def _kernel_variance(scal, tau, grid, h, n_eff_min):
    out = np.full(grid.size, np.nan)
    for k, tc in enumerate(grid):
        w = np.exp(-0.5 * ((tau - tc) / h) ** 2)
        W = w.sum()
        if W < 1e-9: continue
        if W ** 2 / (w * w).sum() < n_eff_min: continue
        m = (w * scal).sum() / W
        out[k] = float((w * (scal - m) ** 2).sum() / W)
    return out


def _matrix_frob_var(J_pc, Jbar, tau, grid, h, n_eff_min):
    """Weighted Frobenius variance of J_i around ⟨J⟩ per τ."""
    out = np.full(grid.size, np.nan)
    for k, tc in enumerate(grid):
        w = np.exp(-0.5 * ((tau - tc) / h) ** 2)
        W = w.sum()
        if W < 1e-9: continue
        if W ** 2 / (w * w).sum() < n_eff_min: continue
        if np.isnan(Jbar[k]).any(): continue
        diff = J_pc - Jbar[k]
        fnorm_sq = (diff * diff).sum(axis=(1, 2))
        out[k] = float((w * fnorm_sq).sum() / W)
    return out


def summarize(lam_matrix, lam_percell, var_pc, frobvar, grid, tau_crit=None):
    if tau_crit is None: tau_crit = gt.TAU_CRIT
    def _summary(curve):
        m = (grid >= 0.05) & (grid <= 0.95) & ~np.isnan(curve)
        argmax = float(grid[np.where(m)[0][np.argmax(curve[m])]]) if m.any() else float("nan")
        t, code = robust_crossing(curve, grid, threshold=0.0, min_run=5)
        return {"argmax_interior": argmax, "argmax_offset": abs(argmax - tau_crit),
                 "cross_tau": t, "cross_code": code,
                 "cross_offset": abs(t - tau_crit) if np.isfinite(t) else float("nan"),
                 "min_lam": float(np.nanmin(curve)),
                 "max_lam": float(np.nanmax(curve))}
    return {"matrix_agg": _summary(lam_matrix),
             "percell_agg": _summary(lam_percell),
             "mean_var_percell_lambda": float(np.nanmean(var_pc)),
             "mean_frob_var_J": float(np.nanmean(frobvar))}


def main():
    out_dir = Path("/Users/terooatt/Downloads/scJDO/reproducibility")
    tau_crit = gt.TAU_CRIT

    results = []
    grid = np.linspace(0.0, 1.0, 400)

    # Test at three bandwidths that span the pre- and post-transition regimes.
    for h in [0.005, 0.01, 0.02, 0.04]:
        print(f"\n===== bandwidth h = {h:.4f} (τ_crit = {tau_crit:.4f}) =====")
        by_seed = []
        for seed in [42, 0, 1, 2, 3]:
            z, tau, J_pc = build_at_cell(seed=seed)
            lam_matrix, lam_percell, var_pc, fv = both_aggregations(J_pc, tau, grid, h=h)
            summary = summarize(lam_matrix, lam_percell, var_pc, fv, grid)
            summary["seed"] = seed
            by_seed.append(summary)
            print(f"  seed={seed}  matrix: argmax={summary['matrix_agg']['argmax_interior']:.4f}  "
                  f"cross={summary['matrix_agg']['cross_code']:>18s} τ̂={summary['matrix_agg']['cross_tau']:.4f}  "
                  f"min={summary['matrix_agg']['min_lam']:+.3f}  |  "
                  f"percell: argmax={summary['percell_agg']['argmax_interior']:.4f}  "
                  f"cross={summary['percell_agg']['cross_code']:>18s} τ̂={summary['percell_agg']['cross_tau']:.4f}  "
                  f"min={summary['percell_agg']['min_lam']:+.3f}")
        results.append({"h": h, "by_seed": by_seed})

    def _san(x):
        if isinstance(x, dict): return {k: _san(v) for k, v in x.items()}
        if isinstance(x, list): return [_san(v) for v in x]
        if isinstance(x, np.floating): return float(x)
        if isinstance(x, np.integer): return int(x)
        return x

    (out_dir / "data" / "FOLLOWUP6_aggregation.json").write_text(
        json.dumps(_san(results), indent=2)
    )
    print(f"\nSaved: {out_dir}/data/FOLLOWUP6_aggregation.json")


if __name__ == "__main__":
    main()
