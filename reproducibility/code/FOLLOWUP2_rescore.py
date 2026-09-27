"""
Follow-up 2 — re-scoring with the robust crossing detector.

(1) T4 edge arms + T2.2 lifted-truth rescored with sustained-run
    crossing detector, bootstrap CI, explicit no-crossing return.
(3) T6 at n_seeds = 20, full distribution, diagnose the outlier.
(4) T5 on log-spaced ρ below τ = 0.05; report the slope on uncensored
    ρ only (censored = offset within one grid step of resolution).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import toggle_truth as gt
from T2_lifted_truth import kernel_matrix_curve, kernel_curve
from T4_geometry_factorial import simulate_and_aggregate
from T6_barrier import barrier_curve, linear_extrap_sqrtDU
from crossing_robust import robust_crossing


def rescore_t4_edge(bandwidth=0.01, min_run=5, n_boot=100):
    """T4 edge arms with the robust detector + bootstrap CI."""
    tau_crit = gt.TAU_CRIT
    rows = []
    for agg in ["nw", "local_linear"]:
        for samp in ["uniform", "dense"]:
            for seed in [0, 1, 2, 3, 4]:
                grid, curve, tau = simulate_and_aggregate(
                    alpha_min=gt.ALPHA_MIN, alpha_max=gt.ALPHA_MAX,
                    sampling=samp, aggregation=agg, seed=seed,
                    n_cells=3000, h=bandwidth,
                )
                tau_hat, code = robust_crossing(curve, grid,
                                                 threshold=0.0, min_run=min_run)
                rows.append({
                    "aggregation": agg, "sampling": samp, "seed": seed,
                    "tau_hat": tau_hat, "code": code,
                    "offset": abs(tau_hat - tau_crit) if np.isfinite(tau_hat) else float("nan"),
                    "bandwidth": bandwidth,
                })
                print(f"  agg={agg:14s}  samp={samp}  seed={seed}  h={bandwidth}  "
                      f"τ̂={tau_hat:.5f}  ({code})  "
                      f"offset={rows[-1]['offset']:.5f}")
    return rows


def rescore_t22_lifted(bandwidth=0.01, min_run=5, n_boot=200, seeds=(42, 1, 2, 3, 4)):
    """T2.2 crossing on lifted truth with robust detector + bootstrap CI.

    Uses the at_cell variant (matches manuscript pipeline). The at_fp
    variant is degenerate for the crossing metric (monotone curve).
    """
    tau_crit = gt.TAU_CRIT
    rows = []
    for seed in seeds:
        z_true, tau_true = gt.simulate_v3(n_cells=3000, seed=seed)
        alphas_pc = gt.alpha_of_tau(tau_true)
        J_pc = np.stack([gt.toggle_jacobian_at(float(z[0]), float(z[1]), float(a))
                          for z, a in zip(z_true, alphas_pc)])
        grid = np.linspace(0.0, 1.0, 400)  # 2× denser than manuscript
        Jbar, _ = kernel_matrix_curve(J_pc, tau_true, grid,
                                       bandwidth=bandwidth, n_eff_min=15.0)
        lam = np.array([float(np.real(np.linalg.eigvals(J)).max())
                         if not np.isnan(J).any() else np.nan for J in Jbar])
        tau_hat, code = robust_crossing(lam, grid, threshold=0.0, min_run=min_run)

        # Bootstrap
        rng = np.random.default_rng(seed)
        N = J_pc.shape[0]
        tau_boots = np.full(n_boot, np.nan)
        for b in range(n_boot):
            idx = rng.integers(0, N, N)
            Jb, _ = kernel_matrix_curve(J_pc[idx], tau_true[idx], grid,
                                         bandwidth=bandwidth, n_eff_min=15.0)
            lamb = np.array([float(np.real(np.linalg.eigvals(J)).max())
                              if not np.isnan(J).any() else np.nan for J in Jb])
            tb, cb = robust_crossing(lamb, grid, threshold=0.0, min_run=min_run)
            tau_boots[b] = tb
        finite = np.isfinite(tau_boots)
        lo = float(np.percentile(tau_boots[finite], 2.5)) if finite.sum() >= 3 else float("nan")
        hi = float(np.percentile(tau_boots[finite], 97.5)) if finite.sum() >= 3 else float("nan")
        p_no = float(1.0 - finite.mean())
        rows.append({
            "seed": seed, "tau_hat": tau_hat, "code": code,
            "offset": abs(tau_hat - tau_crit) if np.isfinite(tau_hat) else float("nan"),
            "boot_lo": lo, "boot_hi": hi, "p_no_crossing": p_no,
        })
        print(f"  seed={seed}  τ̂={tau_hat:.5f}  ({code})  "
              f"offset={rows[-1]['offset']:.5f}  "
              f"CI=[{lo:.4f}, {hi:.4f}]  p_no_cross={p_no:.3f}")
    return rows


def t6_at_n_seeds(n_seeds=20, n_cells=6000, T_total=10.0):
    tau_crit = gt.TAU_CRIT
    tau_grid = np.linspace(0.0, 1.0, 200)
    rows = []
    for seed in range(n_seeds):
        z, tau = gt.simulate_v3(n_cells=n_cells, seed=seed, T_total=T_total)
        dU, modes, saddles, bimodal = barrier_curve(
            z, tau, tau_grid, tau_bandwidth=0.02, w_bandwidth=0.05,
            sigma_sde=gt.SIGMA_SDE,
        )
        tau_hat = linear_extrap_sqrtDU(tau_grid, dU, tau_lo=0.05, tau_hi=0.20)
        # Diagnostic: number of qualifying bimodal points in the fit window
        m = (tau_grid >= 0.05) & (tau_grid <= 0.20)
        bimodal_frac = float(bimodal[m].mean())
        # Also the median wf (transverse coordinate) in the extrapolation region
        w_cells = (z[:, 0] - z[:, 1]) / np.sqrt(2.0)
        m_cells = (tau >= 0.05) & (tau <= 0.20)
        w_std_in_window = float(np.std(w_cells[m_cells])) if m_cells.any() else float("nan")
        rows.append({
            "seed": seed, "tau_hat_sqrtDU": tau_hat,
            "offset": abs(tau_hat - tau_crit) if np.isfinite(tau_hat) else float("nan"),
            "bimodal_frac_in_fit_window": bimodal_frac,
            "w_std_in_fit_window": w_std_in_window,
        })
        print(f"  seed={seed:2d}  τ̂={tau_hat:.4f}  offset={rows[-1]['offset']:.4f}  "
              f"bimodal_frac={bimodal_frac:.2f}  w_std={w_std_in_window:.3f}")
    return rows


def t5_log_spaced(rho_values=None, n_cells=6000, seeds=(0, 1, 2), grid_pts=200):
    """T5 with log-spaced very slow ramps to test whether ΔU-lag scales as
    ρ^{1/2} in the uncensored regime."""
    tau_crit = gt.TAU_CRIT
    tau_grid = np.linspace(0.0, 1.0, grid_pts)
    grid_step = tau_grid[1] - tau_grid[0]
    if rho_values is None:
        # Log-spaced ρ ∈ [3e-3, 1] — spans wall times T ∈ [3, 1000]
        rho_values = np.logspace(np.log10(3e-3), 0, 8)
    rows = []
    for rho in rho_values:
        T_total = (gt.ALPHA_MAX - gt.ALPHA_MIN) / rho
        for seed in seeds:
            z, tau = gt.simulate_v3(n_cells=n_cells, seed=seed, T_total=T_total)
            dU, modes, saddles, bimodal = barrier_curve(
                z, tau, tau_grid, tau_bandwidth=0.02, w_bandwidth=0.05,
                sigma_sde=gt.SIGMA_SDE,
            )
            tau_hat = linear_extrap_sqrtDU(tau_grid, dU, tau_lo=0.05, tau_hi=0.20)
            offset = abs(tau_hat - tau_crit) if np.isfinite(tau_hat) else float("nan")
            censored = np.isfinite(offset) and offset < grid_step
            rows.append({
                "T_total": float(T_total), "rho": float(rho), "seed": seed,
                "tau_hat": tau_hat, "offset": offset, "censored": bool(censored),
            })
            print(f"  ρ={rho:.5f}  T={T_total:8.1f}  seed={seed}  "
                  f"τ̂={tau_hat:.4f}  offset={offset:.4f}  "
                  f"{'CENSORED' if censored else ''}")
    return rows


def fit_slope(rows, min_offset=0.0):
    rho = np.array([r["rho"] for r in rows])
    off = np.array([r["offset"] for r in rows])
    m = np.isfinite(off) & (off > min_offset)
    if m.sum() < 3:
        return {"slope": float("nan"), "stderr": float("nan"), "n": int(m.sum())}
    logr = np.log(rho[m])
    logo = np.log(off[m])
    s, i = np.polyfit(logr, logo, 1)
    yhat = i + s * logr
    resid = logo - yhat
    se = np.sqrt(np.sum(resid ** 2) / max(len(resid) - 2, 1)) / (
        np.sqrt(np.sum((logr - logr.mean()) ** 2))
    )
    return {"slope": float(s), "intercept": float(i),
            "stderr": float(se), "n": int(m.sum())}


def main():
    out_dir = Path("/Users/terooatt/Downloads/scJDO/reproducibility")
    all_out = {"grid_step_dense": 0.0005,
               "tau_crit": gt.TAU_CRIT,
               "dlambda_dtau_at_taucrit": 4.4417}

    print("=" * 74)
    print("(1) T4 edge arms rescored with robust detector, h ≈ 2·τ_crit = 0.01")
    print("=" * 74)
    t4_rows = rescore_t4_edge(bandwidth=0.01, min_run=5)
    all_out["t4_rescored"] = t4_rows

    def _mean_off(rs):
        vals = [r["offset"] for r in rs if np.isfinite(r["offset"])]
        return float(np.mean(vals)) if vals else float("nan"), len(vals)

    print("\n  Summary:")
    for agg in ["nw", "local_linear"]:
        for samp in ["uniform", "dense"]:
            rs = [r for r in t4_rows if r["aggregation"] == agg and r["sampling"] == samp]
            no_cross = sum(1 for r in rs if r["code"] == "no_crossing")
            avg, ntot = _mean_off(rs)
            print(f"    {agg:14s}  {samp:8s}  mean offset={avg:.5f}   "
                  f"no_crossing={no_cross}/{len(rs)}")

    print()
    print("=" * 74)
    print("(1b) T2.2 lifted-truth crossing with robust detector + bootstrap CI")
    print("=" * 74)
    t22_rows = rescore_t22_lifted(bandwidth=0.01, min_run=5, n_boot=100,
                                    seeds=(42, 1, 2, 3, 4))
    all_out["t22_rescored"] = t22_rows

    print()
    print("=" * 74)
    print("(3) T6 at 20 seeds — full distribution & outlier diagnosis")
    print("=" * 74)
    t6_rows = t6_at_n_seeds(n_seeds=20)
    all_out["t6_20seeds"] = t6_rows
    # Distribution
    offs = np.array([r["offset"] for r in t6_rows if np.isfinite(r["offset"])])
    print(f"\n  n_finite={len(offs)} / {len(t6_rows)}")
    print(f"  Mean offset = {offs.mean():.4f}   median = {np.median(offs):.4f}   "
          f"IQR = [{np.percentile(offs, 25):.4f}, {np.percentile(offs, 75):.4f}]")
    n_pass_5pct = int((offs <= 0.05).sum())
    print(f"  {n_pass_5pct}/{len(offs)} within the ±0.05 success window")
    n_outlier = int((offs > 0.1).sum())
    print(f"  {n_outlier}/{len(offs)} with offset > 0.1  (outliers)")

    print()
    print("=" * 74)
    print("(4) T5 with log-spaced ρ (very slow ramps)")
    print("=" * 74)
    t5_rows = t5_log_spaced()
    all_out["t5_log_spaced"] = t5_rows
    fit_all = fit_slope(t5_rows)
    n_cens = sum(1 for r in t5_rows if r["censored"])
    fit_unc = fit_slope([r for r in t5_rows if not r["censored"]])
    print(f"\n  Slope on ALL {fit_all['n']} finite points: "
          f"{fit_all['slope']:.3f} ± {fit_all['stderr']:.3f}")
    print(f"  Censored (offset < grid_step): {n_cens}")
    print(f"  Slope on UNCENSORED ρ only: {fit_unc['slope']:.3f} ± {fit_unc['stderr']:.3f}  "
          f"(n={fit_unc['n']})  95% CI ≈ "
          f"[{fit_unc['slope']-1.96*fit_unc['stderr']:.3f}, "
          f"{fit_unc['slope']+1.96*fit_unc['stderr']:.3f}]")
    all_out["t5_fit_all"] = fit_all
    all_out["t5_fit_uncensored"] = fit_unc

    def _sanitize(x):
        if isinstance(x, dict):
            return {k: _sanitize(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_sanitize(v) for v in x]
        if isinstance(x, np.floating): return float(x)
        if isinstance(x, np.integer): return int(x)
        return x
    (out_dir / "data" / "FOLLOWUP2_rescore.json").write_text(
        json.dumps(_sanitize(all_out), indent=2)
    )
    print(f"\nSaved: {out_dir}/data/FOLLOWUP2_rescore.json")


if __name__ == "__main__":
    main()
