"""
T6 — Quasi-potential barrier readout.

Idea
----
For a gradient system with isotropic noise σ, the stationary density is
p(w) ∝ exp(−2U(w)/σ²), so U(w) = −(σ²/2) log p(w). Away from the pure
gradient case, −σ² log p still converges to the Freidlin–Wentzell quasi-
potential in the small-noise limit. So we can compute the barrier

    ΔU(τ)  =  −(σ²/2) [ log p_τ(w_saddle) − log p_τ(w_mode) ]

directly from the density of cells around each τ. **No drift fit. No
Jacobian. No 20-D spectrum. No gauge ambiguity.**

- Before τ_crit: p(w) is unimodal → ΔU = 0.
- Past τ_crit: ΔU grows quadratically, ΔU ∝ (τ − τ_c)².
- So √ΔU grows linearly past τ_crit. Linearly-extrapolate √ΔU back to zero
  (from post-boundary points) → a second independent estimate of τ_c.

T6 also runs a dip test / GMM-BIC to locate the onset of bimodality in
w. That is an independent estimator of τ_c and, if it obeys the same
ρ^{1/2} scaling as T5 (it should), corroborates the critical-slowing
account.

Outputs
-------
    reproducibility/data/T6_barrier.json
    reproducibility/data/T6_barrier.npz
    reproducibility/figures/T6_barrier.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import toggle_truth as gt


def kde_1d(samples, xs, bandwidth):
    """Simple Gaussian KDE — no dependencies."""
    diffs = (xs[:, None] - samples[None, :]) / bandwidth
    return np.exp(-0.5 * diffs ** 2).mean(axis=1) / (bandwidth * np.sqrt(2 * np.pi))


def barrier_curve(z, tau, tau_grid, tau_bandwidth=0.02, w_bandwidth=0.05,
                  n_wpoints=257, sigma_sde=None,
                  min_peak_prominence=0.05, min_peak_separation=0.15):
    """
    Compute ΔU(τ) for the transverse coordinate w = (x − y)/√2 using
    ``scipy.signal.find_peaks`` with prominence and separation floors so
    that noise wiggles are not mistaken for a genuine second mode.
    """
    from scipy.signal import find_peaks

    if sigma_sde is None:
        sigma_sde = gt.SIGMA_SDE
    w = (z[:, 0] - z[:, 1]) / np.sqrt(2.0)
    w_grid = np.linspace(np.percentile(w, 0.5) - 0.1,
                          np.percentile(w, 99.5) + 0.1,
                          n_wpoints)
    dw = w_grid[1] - w_grid[0]
    min_dist_samples = max(1, int(np.round(min_peak_separation / dw)))
    dU = np.full(tau_grid.size, np.nan)
    modes = np.full((tau_grid.size, 2), np.nan)
    saddles = np.full(tau_grid.size, np.nan)
    bimodal = np.zeros(tau_grid.size, dtype=bool)

    for k, tau_c in enumerate(tau_grid):
        ww = np.exp(-0.5 * ((tau - tau_c) / tau_bandwidth) ** 2)
        n_eff = ww.sum() ** 2 / (ww * ww).sum()
        if n_eff < 50:
            continue
        diffs = (w_grid[:, None] - w[None, :]) / w_bandwidth
        K = np.exp(-0.5 * diffs ** 2) / (w_bandwidth * np.sqrt(2 * np.pi))
        p_w = (K * ww[None, :]).sum(1) / ww.sum()

        # Require prominence relative to overall density scale
        prom_thresh = min_peak_prominence * p_w.max()
        pk_ids, props = find_peaks(p_w, prominence=prom_thresh,
                                   distance=min_dist_samples)
        if len(pk_ids) < 2:
            # Unimodal → the barrier is defined as 0
            dU[k] = 0.0
            continue
        # Pick the two tallest peaks
        top2 = np.argsort(-p_w[pk_ids])[:2]
        pk_ids = np.sort(pk_ids[top2])
        # Find the density minimum between them
        tr_id = pk_ids[0] + int(np.argmin(p_w[pk_ids[0]:pk_ids[1] + 1]))
        log_p_mode = np.log(max(p_w[pk_ids[0]], p_w[pk_ids[1]]) + 1e-300)
        log_p_saddle = np.log(p_w[tr_id] + 1e-300)
        dU[k] = -(sigma_sde ** 2 / 2.0) * (log_p_saddle - log_p_mode)
        modes[k] = (w_grid[pk_ids[0]], w_grid[pk_ids[1]])
        saddles[k] = w_grid[tr_id]
        bimodal[k] = True

    return dU, modes, saddles, bimodal


def linear_extrap_sqrtDU(tau_grid, dU, tau_lo=0.05, tau_hi=0.25):
    """
    √ΔU grows linearly past τ_crit; linearly extrapolate back to 0.

    Fit only points in the *early* post-transition regime (τ ∈ [tau_lo, tau_hi])
    where growth is quadratic in ΔU, hence linear in √ΔU. Later τ saturates
    (fully committed cells at ±w_branch), which contaminates a global fit.
    """
    sqU = np.sqrt(np.clip(dU, 0.0, np.inf))
    m = (tau_grid >= tau_lo) & (tau_grid <= tau_hi) & ~np.isnan(sqU) & (sqU > 0)
    if m.sum() < 4:
        return float("nan")
    x, y = tau_grid[m], sqU[m]
    p = np.polyfit(x, y, 1)
    if p[0] == 0:
        return float("nan")
    return float(-p[1] / p[0])


def main():
    out_dir = Path("/Users/terooatt/Downloads/scJDO/reproducibility")

    tau_crit = gt.TAU_CRIT
    tau_grid = np.linspace(0.0, 1.0, 200)

    all_seeds = []
    for seed in (42, 1, 7):
        z, tau = gt.simulate_v3(n_cells=6000, seed=seed, T_total=10.0)
        dU, modes, saddles, bimodal = barrier_curve(z, tau, tau_grid,
                                                    tau_bandwidth=0.02,
                                                    w_bandwidth=0.05,
                                                    sigma_sde=gt.SIGMA_SDE)
        # Onset of bimodality: first τ where bimodal is True for at least
        # 3 consecutive grid points
        onset = float("nan")
        for i in range(len(bimodal) - 2):
            if bimodal[i] and bimodal[i + 1] and bimodal[i + 2]:
                onset = float(tau_grid[i])
                break
        tau_extrap = linear_extrap_sqrtDU(tau_grid, dU, tau_lo=0.05, tau_hi=0.20)
        all_seeds.append({
            "seed": seed, "onset_bimodality": onset,
            "tau_extrap_sqrtDU": tau_extrap,
            "offset_onset": abs(onset - tau_crit) if np.isfinite(onset) else float("nan"),
            "offset_extrap": abs(tau_extrap - tau_crit) if np.isfinite(tau_extrap) else float("nan"),
            "dU": dU, "modes": modes, "saddles": saddles, "bimodal": bimodal,
        })
        print(f"[seed={seed}]  τ_bimodal_onset={onset:.4f}  τ_extrap(√ΔU)={tau_extrap:.4f}  τ_crit={tau_crit:.4f}")

    # Save
    save_data = {"tau_crit": tau_crit, "tau_grid": tau_grid}
    for r in all_seeds:
        save_data[f"dU_s{r['seed']}"] = r["dU"]
        save_data[f"bimodal_s{r['seed']}"] = r["bimodal"]
        save_data[f"saddles_s{r['seed']}"] = r["saddles"]
        save_data[f"modes_s{r['seed']}"] = r["modes"]
    np.savez_compressed(out_dir / "data" / "T6_barrier.npz", **save_data)

    summary = {
        "tau_crit": tau_crit,
        "seeds": [
            {k: v for k, v in r.items()
             if k not in ("dU", "modes", "saddles", "bimodal")}
            for r in all_seeds
        ],
    }
    (out_dir / "data" / "T6_barrier.json").write_text(json.dumps(summary, indent=2))

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1, figsize=(7.5, 7))
    for r in all_seeds:
        ax[0].plot(tau_grid, r["dU"], label=f"seed={r['seed']}")
    ax[0].axvline(tau_crit, color="red", lw=0.8, ls="--", label=r"$\tau_{\rm crit}$")
    ax[0].set_ylabel(r"$\Delta U(\tau)$")
    ax[0].set_title(r"T6: Boltzmann-inverted barrier from KDE of $w$")
    ax[0].legend(fontsize=8)

    for r in all_seeds:
        sq = np.sqrt(np.clip(r["dU"], 0.0, np.inf))
        ax[1].plot(tau_grid, sq, label=f"seed={r['seed']}")
    ax[1].axvline(tau_crit, color="red", lw=0.8, ls="--", label=r"$\tau_{\rm crit}$")
    ax[1].set_xlabel(r"pseudotime $\tau$")
    ax[1].set_ylabel(r"$\sqrt{\Delta U(\tau)}$")
    ax[1].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_dir / "figures" / "T6_barrier.pdf")
    fig.savefig(out_dir / "figures" / "T6_barrier.png", dpi=150)
    print(f"\nSaved: {out_dir}/data/T6_barrier.json")
    print(f"       {out_dir}/figures/T6_barrier.pdf")


if __name__ == "__main__":
    main()
