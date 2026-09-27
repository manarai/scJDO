"""
Follow-up reruns requested after the first SUMMARY:

(1) Edge-located T4 arms scored on the FULL grid including τ_crit.
    Does the dense-sampling null survive when the argmax scoring window
    no longer excludes τ_crit?
(4) T5 with the ΔU-based estimator: rerun the ρ sweep computing τ̂_c
    from √ΔU extrapolation. Fit log(offset) vs log(ρ). Report slope
    with the same log-log-fit standard error used for the other T5
    metrics.

The conversion factor  dλ_⊥/dτ ≈ 4.4417  (numerical, section verified)
is applied wherever λ shifts need to be compared to the manuscript's
success window in τ.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import toggle_truth as gt
from T4_geometry_factorial import simulate_and_aggregate
from T5_kibble_zurek import fit_scaling, bin_transverse_variance
from T6_barrier import barrier_curve, linear_extrap_sqrtDU


# ---------------------------------------------------------------------------
# (1) Edge-located T4 with grid scored on [0, 1] (full range including τ_crit)
# ---------------------------------------------------------------------------

def argmax_full(curve, grid):
    m = ~np.isnan(curve)
    if not m.any():
        return float("nan")
    return float(grid[np.where(m)[0][np.argmax(curve[m])]])


def crossing_full(curve, grid, threshold=0.0):
    valid = ~np.isnan(curve)
    for i in range(len(curve)):
        if valid[i] and curve[i] > threshold:
            return float(grid[i])
    return float("nan")


def rerun_edge_t4():
    tau_crit = gt.TAU_CRIT
    rows = []
    for agg in ["nw", "local_linear"]:
        for samp in ["uniform", "dense"]:
            for seed in [0, 1, 2, 3, 4]:
                grid, curve, tau = simulate_and_aggregate(
                    alpha_min=gt.ALPHA_MIN, alpha_max=gt.ALPHA_MAX,
                    sampling=samp, aggregation=agg, seed=seed, n_cells=1500,
                )
                am_full = argmax_full(curve, grid)
                cr_full = crossing_full(curve, grid)
                rows.append({
                    "aggregation": agg, "sampling": samp, "seed": seed,
                    "argmax_full": am_full,
                    "crossing_full": cr_full,
                    "argmax_offset": abs(am_full - tau_crit),
                    "crossing_offset": abs(cr_full - tau_crit),
                    "grid_step": float(grid[1] - grid[0]),
                })
                print(f"  agg={agg}  samp={samp}  seed={seed}  "
                      f"argmax_full={am_full:.4f}  cross_full={cr_full:.4f}  "
                      f"|argmax−τ_crit|={abs(am_full-tau_crit):.4f}")
    return rows


# ---------------------------------------------------------------------------
# (4) T5 with the ΔU-based estimator
# ---------------------------------------------------------------------------

def rerun_t5_with_delta_U():
    tau_crit = gt.TAU_CRIT
    T_totals = [5.0, 10.0, 20.0, 40.0, 80.0, 100.0]
    seeds = [0, 1, 2]
    tau_grid = np.linspace(0.0, 1.0, 200)
    rows = []
    for T in T_totals:
        rho = (gt.ALPHA_MAX - gt.ALPHA_MIN) / T
        for seed in seeds:
            z, tau = gt.simulate_v3(n_cells=6000, seed=seed, T_total=T,
                                     dt=0.02, sigma_sde=gt.SIGMA_SDE,
                                     alpha_min=gt.ALPHA_MIN,
                                     alpha_max=gt.ALPHA_MAX)
            dU, modes, saddles, bimodal = barrier_curve(
                z, tau, tau_grid,
                tau_bandwidth=0.02, w_bandwidth=0.05,
                sigma_sde=gt.SIGMA_SDE,
            )
            # Extrapolate √ΔU on early post-transition points [0.05, 0.20]
            tau_hat = linear_extrap_sqrtDU(tau_grid, dU, tau_lo=0.05, tau_hi=0.20)
            offset = abs(tau_hat - tau_crit) if np.isfinite(tau_hat) else float("nan")
            rows.append({
                "T_total": T, "rho": rho, "seed": seed,
                "tau_hat_sqrtDU": tau_hat,
                "offset_sqrtDU": offset,
            })
            print(f"  T={T:6.1f}  ρ={rho:.4f}  seed={seed}  "
                  f"τ̂_ΔU={tau_hat:.4f}  offset={offset:.4f}")
    fit = fit_scaling(rows, key="offset_sqrtDU")
    return rows, fit


def main():
    out_dir = Path("/Users/terooatt/Downloads/scJDO/reproducibility")

    print("=" * 68)
    print("(1) Edge-located T4 with grid scored on [0, 1]")
    print("=" * 68)
    rows1 = rerun_edge_t4()

    # Summarise
    def _mean(rs, key):
        vals = [r[key] for r in rs if np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("nan")
    def _median(rs, key):
        vals = [r[key] for r in rs if np.isfinite(r[key])]
        return float(np.median(vals)) if vals else float("nan")

    print("\n  Summary (mean ± range across 5 seeds):")
    for agg in ["nw", "local_linear"]:
        for samp in ["uniform", "dense"]:
            rs = [r for r in rows1 if r["aggregation"] == agg and r["sampling"] == samp]
            arg_off = _mean(rs, "argmax_offset")
            arg_min = min(r["argmax_offset"] for r in rs)
            arg_max = max(r["argmax_offset"] for r in rs)
            cr_off = _mean(rs, "crossing_offset")
            print(f"    {agg:14s}  {samp:8s}  argmax_off mean={arg_off:.4f} "
                  f"[{arg_min:.4f}, {arg_max:.4f}]  crossing_off mean={cr_off:.4f}")

    print()
    print("=" * 68)
    print("(4) T5 with ΔU-based estimator")
    print("=" * 68)
    rows4, fit4 = rerun_t5_with_delta_U()
    print()
    print(f"  Log-log fit slope (ΔU extrapolation offset ~ ρ^slope):")
    print(f"      slope = {fit4['slope']:.3f} ± {fit4['slope_stderr']:.3f}  (KZ target 0.5)")
    print(f"      95% CI ≈ [{fit4['slope']-1.96*fit4['slope_stderr']:.3f}, "
          f"{fit4['slope']+1.96*fit4['slope_stderr']:.3f}]")

    out = {
        "grid_step": float(np.linspace(0, 1, 200)[1]),
        "tau_crit": gt.TAU_CRIT,
        "dlambda_perp_dtau_at_taucrit": 4.4417,
        "edge_t4_extended_grid": rows1,
        "t5_with_dU_estimator": {"rows": rows4, "fit": fit4},
    }
    (out_dir / "data" / "FOLLOWUP_reruns.json").write_text(
        json.dumps(out, indent=2, default=lambda x: float(x)
                   if isinstance(x, (np.floating,)) else int(x)
                   if isinstance(x, (np.integer,)) else x)
    )
    print(f"\nSaved: {out_dir}/data/FOLLOWUP_reruns.json")


if __name__ == "__main__":
    main()
