"""
T4 — Geometry factorial on the LIFTED-TRUTH readout (no training).

The full-fit factorial (8 cells × 5 seeds × ~500 s per fit ≈ 5.5 hours) is
too expensive for this pass. Instead we run the geometry factorial ON THE
LIFTED-TRUTH SIGNAL — this isolates the geometry effects (edge vs mid-domain
saddle, NW vs local-linear aggregation, uniform vs dense sampling near
τ_crit) from any effect of the neural drift.

Two levers we control:
  1. Where does the analytic pitchfork sit in τ? By choosing α_min and α_max,
     we can put τ_crit near the edge (baseline, τ_crit ≈ 0.004) or at
     mid-domain (τ_crit = 0.5 by construction). This tests whether the
     τ_crit-close-to-boundary geometry is the driver.
  2. τ sampling: uniform in τ vs. 5× density near τ_crit.

Aggregation:
  * Nadaraya–Watson (kernel-weighted average, current pipeline).
  * Local linear regression: for each grid point τ_g fit
       f_i ≈ β_0(τ_g) + β_1(τ_g) · (τ_i − τ_g)
    with Gaussian weights, and use β_0 as the estimate. This eliminates
    the first-order boundary bias.

Metric:
  * The crossing metric from T2.2:  first τ where the signal-subspace λ ≥ 0.
  * argmax over interior [0.05, 0.95] for comparison.

At each factorial cell we simulate the toggle switch with the appropriate
α range and sampling, then compute the aggregated λ_signal(τ) via each
aggregation method.

Outputs
-------
    reproducibility/data/T4_geometry_factorial.json
    reproducibility/figures/T4_geometry_factorial.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import toggle_truth as gt


# ---------------------------------------------------------------------------
# Aggregation methods
# ---------------------------------------------------------------------------

def nw_curve(scalar_per_cell, tau, grid, h, n_eff_min=20.0):
    curve = np.full(grid.size, np.nan)
    for k, tc in enumerate(grid):
        w = np.exp(-0.5 * ((tau - tc) / h) ** 2)
        W = w.sum()
        if W < 1e-9:
            continue
        n_eff = W * W / (w * w).sum()
        if n_eff < n_eff_min:
            continue
        curve[k] = float((w * scalar_per_cell).sum() / W)
    return curve


def local_linear_curve(scalar_per_cell, tau, grid, h, n_eff_min=20.0):
    """
    Local linear regression: f_i ≈ β_0 + β_1 · (τ_i − τ_g), Gaussian
    weights. Returns β_0(τ_g). This is unbiased at O(h²) at boundaries.
    """
    curve = np.full(grid.size, np.nan)
    for k, tc in enumerate(grid):
        dt = tau - tc
        w = np.exp(-0.5 * (dt / h) ** 2)
        W = w.sum()
        if W < 1e-9:
            continue
        n_eff = W * W / (w * w).sum()
        if n_eff < n_eff_min:
            continue
        Sw = w.sum()
        Swx = (w * dt).sum()
        Swxx = (w * dt * dt).sum()
        Swy = (w * scalar_per_cell).sum()
        Swxy = (w * dt * scalar_per_cell).sum()
        # Solve 2x2 normal equations
        det = Sw * Swxx - Swx * Swx
        if abs(det) < 1e-12:
            continue
        beta0 = (Swxx * Swy - Swx * Swxy) / det
        curve[k] = beta0
    return curve


# ---------------------------------------------------------------------------
# One factorial cell (5 seeds each)
# ---------------------------------------------------------------------------

def simulate_and_aggregate(alpha_min, alpha_max, sampling, aggregation,
                            seed=42, n_cells=2000, h=0.04):
    """
    Simulate on the given [alpha_min, alpha_max], obtain a per-cell λ_signal
    scalar via the analytic 2-D Jacobian at each cell's actual (x,y,α), and
    aggregate onto the τ-grid with the chosen method.
    """
    rng = np.random.default_rng(seed)
    n_steps_scale = 500   # baseline

    if sampling == "uniform":
        # Uniform τ ∈ [0, 1] as in the baseline simulator.
        z, tau = gt.simulate_v3(n_cells=n_cells, seed=seed,
                                 alpha_min=alpha_min, alpha_max=alpha_max)
    elif sampling == "dense":
        # Uniform + 5× oversample near τ_crit. τ_crit here is the
        # geometry-appropriate value for this [α_min, α_max].
        z1, tau1 = gt.simulate_v3(n_cells=n_cells, seed=seed,
                                   alpha_min=alpha_min, alpha_max=alpha_max)
        tau_crit_local = (gt.ALPHA_CRIT - alpha_min) / (alpha_max - alpha_min)
        n_extra = 5 * n_cells // 10   # 50% extra density in the window
        # Simulate additional cells and only keep those whose τ falls in the
        # τ_crit window. The scheduler samples target_step uniformly, so
        # rejection-sampling those falling in [max(0, tau_crit - 0.05),
        # tau_crit + 0.05] gives us the dense region.
        low = max(0.0, tau_crit_local - 0.05)
        high = min(1.0, tau_crit_local + 0.05)
        z_list = [z1]; tau_list = [tau1]
        acc = 0
        s = seed + 1
        while acc < n_extra:
            z_i, tau_i = gt.simulate_v3(n_cells=n_extra * 4, seed=s,
                                         alpha_min=alpha_min, alpha_max=alpha_max)
            m = (tau_i >= low) & (tau_i < high)
            z_list.append(z_i[m])
            tau_list.append(tau_i[m])
            acc += int(m.sum())
            s += 1
        z = np.concatenate(z_list, axis=0)
        tau = np.concatenate(tau_list, axis=0)
    else:
        raise ValueError(sampling)

    # Per-cell λ_signal scalar. Signal-subspace = the 2-D toggle plane
    # itself (identity in ground-truth coordinates).
    alphas = gt.alpha_of_tau(tau, alpha_min=alpha_min, alpha_max=alpha_max)
    lam_pc = np.zeros(z.shape[0])
    for i in range(z.shape[0]):
        J = gt.toggle_jacobian_at(float(z[i, 0]), float(z[i, 1]), float(alphas[i]))
        lam_pc[i] = float(np.real(np.linalg.eigvals(J)).max())

    grid = np.linspace(0.0, 1.0, 200)
    if aggregation == "nw":
        curve = nw_curve(lam_pc, tau, grid, h)
    elif aggregation == "local_linear":
        curve = local_linear_curve(lam_pc, tau, grid, h)
    else:
        raise ValueError(aggregation)

    return grid, curve, tau


def find_argmax(curve, grid, low=0.05, high=0.95):
    m = (grid >= low) & (grid <= high) & ~np.isnan(curve)
    if not m.any():
        return float("nan")
    return float(grid[np.where(m)[0][np.argmax(curve[m])]])


def find_crossing(curve, grid, threshold=0.0):
    valid = ~np.isnan(curve)
    for i in range(len(curve)):
        if valid[i] and curve[i] > threshold:
            return float(grid[i])
    return float("nan")


def main():
    out_dir = Path("/Users/terooatt/Downloads/scJDO/reproducibility")

    # 2 saddle locations × 2 aggregations × 2 samplings × 5 seeds = 40 cells
    saddle_locations = {
        "edge": (1.0, 4.0),           # τ_crit ≈ 0.0044
        # For mid-domain we shift α_min *below* α_min=1 so the α ramp spans
        # α_crit near the middle of τ. Use α_min = 0.0, α_max = 2.026.
        "mid":  (0.0, 2.0 * gt.ALPHA_CRIT),  # ⇒ τ_crit ≈ 0.5
    }
    aggregations = ["nw", "local_linear"]
    samplings = ["uniform", "dense"]
    seeds = [0, 1, 2, 3, 4]

    rows = []
    for sad, (a_lo, a_hi) in saddle_locations.items():
        tau_crit_local = float((gt.ALPHA_CRIT - a_lo) / (a_hi - a_lo))
        for agg in aggregations:
            for samp in samplings:
                for s in seeds:
                    grid, curve, tau = simulate_and_aggregate(
                        a_lo, a_hi, sampling=samp, aggregation=agg,
                        seed=s, n_cells=1500,
                    )
                    argmax = find_argmax(curve, grid)
                    cross = find_crossing(curve, grid)
                    rows.append({
                        "saddle": sad,
                        "tau_crit_local": tau_crit_local,
                        "aggregation": agg,
                        "sampling": samp,
                        "seed": s,
                        "argmax": argmax,
                        "crossing": cross,
                        "argmax_offset": abs(argmax - tau_crit_local),
                        "crossing_offset": abs(cross - tau_crit_local),
                    })
                    print(f"  sad={sad}  agg={agg}  samp={samp}  seed={s}  "
                          f"argmax={argmax:.3f}  cross={cross:.4f}  "
                          f"τ_crit_local={tau_crit_local:.4f}")

    # Summarise
    def _agg_across_seeds(sad, agg, samp, key):
        vals = [r[key] for r in rows
                if r["saddle"] == sad and r["aggregation"] == agg and r["sampling"] == samp
                and np.isfinite(r[key])]
        return {"mean": float(np.mean(vals)) if vals else float("nan"),
                "median": float(np.median(vals)) if vals else float("nan"),
                "std": float(np.std(vals)) if vals else float("nan"),
                "n": len(vals)}

    summary = {"tau_crit_edge": (gt.ALPHA_CRIT - 1.0) / 3.0,
               "tau_crit_mid": 0.5,
               "cells": []}
    for sad in saddle_locations:
        for agg in aggregations:
            for samp in samplings:
                summary["cells"].append({
                    "saddle": sad, "aggregation": agg, "sampling": samp,
                    "argmax_offset": _agg_across_seeds(sad, agg, samp, "argmax_offset"),
                    "crossing_offset": _agg_across_seeds(sad, agg, samp, "crossing_offset"),
                })

    (out_dir / "data" / "T4_geometry_factorial.json").write_text(
        json.dumps({"rows": rows, "summary": summary}, indent=2)
    )

    # Table print
    print("\n=== Summary: mean argmax_offset across 5 seeds ===")
    print(f"{'saddle':6s}  {'agg':13s}  {'samp':8s}  {'argmax_off':>10s}  {'crossing_off':>12s}")
    for c in summary["cells"]:
        print(f"{c['saddle']:6s}  {c['aggregation']:13s}  {c['sampling']:8s}  "
              f"{c['argmax_offset']['mean']:10.4f}  {c['crossing_offset']['mean']:12.4f}")


if __name__ == "__main__":
    main()
