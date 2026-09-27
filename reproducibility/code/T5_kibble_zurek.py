"""
T5 — Kibble–Zurek boundary-layer scaling.

Question
--------
When the α ramp is slower (smaller ρ = dα/dτ_wall), does the *offset* between
the true pitchfork τ_crit and the empirically-detected crossing shrink like
ρ^(1/2), as freeze-out theory predicts?

The transverse relaxation time diverges as 1/|λ_⊥| = 1/|−1 + c| ~ 1/|α − α_c|
near the pitchfork. Freeze-out where the relaxation time equals the time
remaining to the transition:

    1/|λ|  ≈  |λ / λ̇|
    → |α − α_c| ~ (k_α ρ)^(1/2)
    → Δτ_boundary  ∝  ρ^(1/2)  (with α range fixed).

Design
------
We hold α_min, α_max, and n_cells fixed; we sweep the total simulated wall
time T_total. Smaller T_total = larger ρ, and vice versa. For each ρ we
simulate the SDE, then compute a *simulator-native* crossing estimate:

    Definition (used here). Bin cells by pseudotime τ into narrow windows,
    fit the transverse variance Var(w) as a function of α, and detect the
    onset of super-stationary behavior — the first τ where Var(w) exceeds
    a threshold above its pre-transition baseline. This is the same
    signature the eigenvalue would give (Var(w) → ∞ at criticality), but
    it uses only the raw snapshot, no drift model, no eigen-decomposition.
    It is a proper Kibble–Zurek observable — it must inherit ρ^(1/2)
    scaling if freeze-out is the operative mechanism.

Outputs
-------
    reproducibility/data/T5_kz.json
    reproducibility/data/T5_kz.npz           per-rho binned Var(w) profiles
    reproducibility/figures/T5_kz_scaling.pdf     log-log ρ vs offset
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import toggle_truth as gt


def bin_transverse_variance(z, tau, n_bins=60, tau_range=(0.0, 1.0)):
    """
    Bin cells into n_bins uniform-in-τ windows and return the bin centres
    together with the sample variance of the transverse coordinate
    w = (x − y)/√2 inside each bin.
    """
    lo, hi = tau_range
    edges = np.linspace(lo, hi, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    w = (z[:, 0] - z[:, 1]) / np.sqrt(2.0)
    var = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    idx = np.searchsorted(edges, tau, side="right") - 1
    for b in range(n_bins):
        mask = idx == b
        counts[b] = mask.sum()
        if counts[b] >= 5:
            var[b] = float(np.var(w[mask], ddof=1))
    return centres, var, counts


def onset_crossing(centres, var, baseline_slice=(0, 5), factor=3.0):
    """
    First τ where Var(w) exceeds ``factor × baseline_variance`` (median of
    the first ``baseline_slice`` bins). Returns np.nan if no crossing is
    found or the baseline is undefined.
    """
    b0, b1 = baseline_slice
    base_vals = var[b0:b1]
    base_vals = base_vals[~np.isnan(base_vals)]
    if len(base_vals) == 0:
        return float("nan"), float("nan")
    baseline = float(np.median(base_vals))
    threshold = factor * baseline
    for i, v in enumerate(var):
        if not np.isnan(v) and v > threshold:
            return float(centres[i]), baseline
    return float("nan"), baseline


def linear_extrapolation_crossing(centres, var, fit_slice=(-15, -5)):
    """
    T5.2 — Extrapolate a linear fit of Var(w) vs τ in a post-transition
    slice back to zero (near-baseline) to estimate τ_c.  This uses only
    points OUTSIDE the boundary layer.
    """
    a, b = fit_slice
    x = centres[a:b]
    y = var[a:b]
    m = ~np.isnan(y)
    if m.sum() < 3:
        return float("nan")
    x, y = x[m], y[m]
    p = np.polyfit(x, y, 1)   # y = p[0]*x + p[1]
    if p[0] == 0:
        return float("nan")
    return float(-p[1] / p[0])


def run_sweep(T_totals, n_cells=4000, seeds=(0, 1, 2)):
    tau_crit = gt.TAU_CRIT
    alpha_min, alpha_max = gt.ALPHA_MIN, gt.ALPHA_MAX
    d_alpha = alpha_max - alpha_min

    rows = []
    curves = {}
    for T_total in T_totals:
        rho = d_alpha / T_total   # dα/dt in wall-clock; ~ ramp rate
        for seed in seeds:
            z, tau = gt.simulate_v3(
                n_cells=n_cells, seed=seed, T_total=T_total, dt=0.02,
                sigma_sde=gt.SIGMA_SDE,
                alpha_min=alpha_min, alpha_max=alpha_max,
            )
            centres, var, counts = bin_transverse_variance(z, tau, n_bins=80)
            tau_onset, baseline = onset_crossing(centres, var)
            tau_extrap = linear_extrapolation_crossing(centres, var,
                                                       fit_slice=(-30, -10))
            rows.append({
                "T_total": T_total,
                "rho": rho,
                "seed": seed,
                "tau_onset": tau_onset,
                "tau_extrap": tau_extrap,
                "baseline_var": baseline,
                "offset_onset": abs(tau_onset - tau_crit),
                "offset_extrap": abs(tau_extrap - tau_crit),
            })
            curves.setdefault(T_total, {}).setdefault(seed, {})
            curves[T_total][seed] = {
                "centres": centres, "var": var, "counts": counts,
            }
            print(f"  T={T_total:6.1f}  ρ={rho:.4f}  seed={seed}  "
                  f"τ_onset={tau_onset:.4f}  offset={abs(tau_onset - tau_crit):.4f}  "
                  f"τ_extrap={tau_extrap:.4f}")
    return rows, curves


def fit_scaling(rows, key="offset_onset"):
    """log(offset) = a + b · log(ρ). Expect b ≈ 1/2 under Kibble–Zurek."""
    rho = np.array([r["rho"] for r in rows], dtype=float)
    off = np.array([r[key] for r in rows], dtype=float)
    m = np.isfinite(off) & (off > 0)
    if m.sum() < 3:
        return {"slope": float("nan"), "intercept": float("nan"),
                "n_used": int(m.sum())}
    logr = np.log(rho[m])
    logo = np.log(off[m])
    slope, intercept = np.polyfit(logr, logo, 1)
    # Standard error of the slope from the residuals
    yhat = intercept + slope * logr
    resid = logo - yhat
    if len(resid) > 2:
        s_e = np.sqrt(np.sum(resid ** 2) / (len(resid) - 2)) / (
            np.sqrt(np.sum((logr - logr.mean()) ** 2))
        )
    else:
        s_e = float("nan")
    return {"slope": float(slope), "intercept": float(intercept),
            "slope_stderr": float(s_e), "n_used": int(m.sum())}


def main():
    out_dir = Path("/Users/terooatt/Downloads/scJDO/reproducibility")

    # Sweep ρ over more than a decade in the SLOW-RAMP regime, where the
    # KZ boundary layer is narrow enough to fit inside τ ∈ [0, 1]. Larger
    # T_total = slower ramp. ρ = 3 / T_total spans 0.03 → 0.6 here.
    T_totals = [5.0, 10.0, 20.0, 40.0, 80.0, 100.0]   # ρ ranges 0.6 → 0.03
    rows, curves = run_sweep(T_totals, n_cells=4000, seeds=(0, 1, 2))

    fit_on = fit_scaling(rows, "offset_onset")
    fit_ex = fit_scaling(rows, "offset_extrap")

    print()
    print(f"onset  scaling slope = {fit_on['slope']:.3f} ± {fit_on['slope_stderr']:.3f}  (KZ expects 0.5)")
    print(f"extrap scaling slope = {fit_ex['slope']:.3f} ± {fit_ex['slope_stderr']:.3f}")

    summary = {
        "tau_crit": gt.TAU_CRIT,
        "T_totals": T_totals,
        "rows": rows,
        "fit_onset": fit_on,
        "fit_extrap": fit_ex,
        "prediction": "Kibble–Zurek predicts slope ≈ 0.5 for offset ∝ ρ^{1/2}.",
    }
    (out_dir / "data" / "T5_kz.json").write_text(json.dumps(summary, indent=2))
    # Save curves in npz (must be non-nested)
    save_dict = {"tau_crit": gt.TAU_CRIT}
    for T, seed_curves in curves.items():
        for seed, cd in seed_curves.items():
            save_dict[f"centres_T{T}_s{seed}"] = cd["centres"]
            save_dict[f"var_T{T}_s{seed}"] = cd["var"]
            save_dict[f"counts_T{T}_s{seed}"] = cd["counts"]
    np.savez_compressed(out_dir / "data" / "T5_kz.npz", **save_dict)

    # ── Plot ────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: Var(w) profiles for each ρ (seed=0)
    for T in T_totals:
        cd = curves[T][0]
        ax[0].plot(cd["centres"], cd["var"], label=f"T={T:.0f}  ρ={gt.ALPHA_MAX-gt.ALPHA_MIN:.0f}/{T:.0f}={((gt.ALPHA_MAX-gt.ALPHA_MIN)/T):.3f}")
    ax[0].axvline(gt.TAU_CRIT, color="red", lw=0.8, ls="--", label=r"$\tau_{\rm crit}$")
    ax[0].set_yscale("log")
    ax[0].set_xlabel(r"pseudotime $\tau$")
    ax[0].set_ylabel(r"transverse variance $\operatorname{Var}(w)$")
    ax[0].set_title("T5: Var(w) profile as function of ramp rate ρ")
    ax[0].legend(fontsize=8, loc="upper left")

    # Right: log-log
    rho = np.array([r["rho"] for r in rows])
    off_on = np.array([r["offset_onset"] for r in rows])
    off_ex = np.array([r["offset_extrap"] for r in rows])
    ax[1].loglog(rho, off_on, "o", label="offset (onset ×3 baseline)", ms=6)
    ax[1].loglog(rho, off_ex, "s", label="offset (linear extrap.)", ms=6, alpha=0.7)
    # Draw KZ 0.5 slope reference through the mean-log center
    m = np.isfinite(off_on) & (off_on > 0)
    if m.sum() > 1:
        c_ref = np.exp(np.mean(np.log(off_on[m])) - 0.5 * np.mean(np.log(rho[m])))
        xr = np.array([rho.min() * 0.8, rho.max() * 1.2])
        ax[1].loglog(xr, c_ref * xr ** 0.5, "k--", lw=1, alpha=0.7,
                     label="ρ^{1/2} reference (KZ)")
    slope = fit_on["slope"]
    ax[1].set_xlabel(r"ramp rate $\rho = d\alpha/dt$")
    ax[1].set_ylabel(r"$|\hat\tau_c - \tau_{\rm crit}|$")
    ax[1].set_title(f"T5: Kibble–Zurek scaling (fit slope onset = {slope:.3f})")
    ax[1].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_dir / "figures" / "T5_kz_scaling.pdf")
    fig.savefig(out_dir / "figures" / "T5_kz_scaling.png", dpi=150)
    print(f"\nSaved: {out_dir}/data/T5_kz.json")
    print(f"       {out_dir}/figures/T5_kz_scaling.pdf")


if __name__ == "__main__":
    main()
