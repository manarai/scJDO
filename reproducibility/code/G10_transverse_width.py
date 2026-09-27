"""
G-10 — Transverse conditional width vs DSM σ ladder.

For a τ-conditional OU approximation around the symmetric fixed point,
the transverse coordinate w = (x-y)/√2 has stationary variance

    w_⊥(τ) = σ_SDE² / (2 |λ_⊥(α(τ))|)

with σ_SDE = 0.20 for this benchmark.  Below the pitchfork the
approximation is valid; at the pitchfork w_⊥ diverges; past it the
formula still applies to the (unstable) symmetric fp but the physical
distribution is bimodal.  We report:

  - w_⊥(τ)                       (variance)
  - √w_⊥(τ) = std of w | τ       (width)
  - the DSM σ used in these fits (σ = 0.10 constant, per FOLLOWUP7
    and G-5/G-6/G-8 training)
  - the sigma ladder [σ_min, σ_max] the manuscript nominally uses
    ([1e-3, 5e-1], from user's note)

Report the τ range where:
  (a) w_⊥ (variance) < σ_min² = 1e-6   → structure below the finest noise floor
  (b) √w_⊥ (width)   < σ_DSM  = 0.1    → structure below the training-time noise
  (c) √w_⊥ (width)   < σ_max  = 0.5    → structure below the coarsest noise

The verdict on whether the objective is blind by construction.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import toggle_truth as gt


def main():
    out_dir = Path("/Users/terooatt/Downloads/scJDO/reproducibility")

    # Fine τ grid, log-spaced approach to τ_crit from BOTH sides.
    tau_crit = gt.TAU_CRIT
    alpha_crit = gt.ALPHA_CRIT
    tau_left = tau_crit - np.logspace(-6, np.log10(tau_crit - 1e-6), 200)
    tau_right = tau_crit + np.logspace(-6, np.log10(1.0 - tau_crit - 1e-6), 200)
    tau_regular = np.linspace(0.0, 1.0, 200)
    tau = np.sort(np.unique(np.concatenate([tau_left, tau_right, tau_regular])))
    tau = tau[(tau >= 0.0) & (tau <= 1.0)]

    alpha = gt.alpha_of_tau(tau)
    lam_perp = np.array([gt.transverse_eigenvalue(a) for a in alpha])
    abs_lam = np.abs(lam_perp) + 1e-30
    w_perp_var = gt.SIGMA_SDE ** 2 / (2.0 * abs_lam)
    w_perp_std = np.sqrt(w_perp_var)

    sigma_min = 1e-3
    sigma_max = 5e-1
    sigma_dsm = 0.10

    # (a) w_⊥ < σ_min²
    mask_a = w_perp_var < sigma_min ** 2
    # (b) √w_⊥ < σ_DSM
    mask_b = w_perp_std < sigma_dsm
    # (c) √w_⊥ < σ_max
    mask_c = w_perp_std < sigma_max

    def _range(mask):
        idx = np.where(mask)[0]
        if idx.size == 0:
            return None
        segments = []
        start = idx[0]
        prev = idx[0]
        for i in idx[1:]:
            if i != prev + 1:
                segments.append((float(tau[start]), float(tau[prev])))
                start = i
            prev = i
        segments.append((float(tau[start]), float(tau[prev])))
        return segments

    a_rng = _range(mask_a)
    b_rng = _range(mask_b)
    c_rng = _range(mask_c)

    print("=" * 70)
    print(f"G-10 — Transverse conditional width vs DSM noise scales")
    print("=" * 70)
    print(f"σ_SDE = {gt.SIGMA_SDE}")
    print(f"τ_crit = {tau_crit:.5f}")
    print(f"σ_DSM used in FOLLOWUP7 / G-5..G-8 = {sigma_dsm}")
    print(f"manuscript σ ladder = [{sigma_min}, {sigma_max}]")
    print()
    print(f"{'τ':>8s}  {'α':>8s}  {'|λ_⊥|':>10s}  {'√w_⊥':>10s}  {'w_⊥':>10s}  "
          f"{'√w_⊥ < σ_DSM?':>16s}")
    for i, ti in enumerate(np.linspace(0, 1, 21)):
        j = int(np.searchsorted(tau, ti))
        j = min(j, len(tau) - 1)
        print(f"  {tau[j]:>6.4f}  {alpha[j]:>6.3f}  {abs_lam[j]:>10.4f}  "
              f"{w_perp_std[j]:>10.4f}  {w_perp_var[j]:>10.4f}  "
              f"{'YES' if mask_b[j] else 'no':>16s}")

    print()
    print(f"(a) w_⊥ < σ_min²  (= {sigma_min ** 2:.1e}) : range = {a_rng}")
    print(f"(b) √w_⊥ < σ_DSM  (= {sigma_dsm})          : range = {b_rng}")
    print(f"(c) √w_⊥ < σ_max  (= {sigma_max})          : range = {c_rng}")

    # Extra: min and max of √w_⊥ across [0, 1]
    print()
    print(f"√w_⊥ min = {w_perp_std.min():.4f} at τ = {tau[np.argmin(w_perp_std)]:.4f}")
    print(f"√w_⊥ max = {w_perp_std.max():.4e} at τ = {tau[np.argmax(w_perp_std)]:.4f}")

    out = {
        "sigma_sde": gt.SIGMA_SDE,
        "tau_crit": tau_crit,
        "sigma_dsm_used": sigma_dsm,
        "sigma_manuscript_ladder": [sigma_min, sigma_max],
        "range_variance_lt_sigma_min_sq": a_rng,
        "range_std_lt_sigma_dsm": b_rng,
        "range_std_lt_sigma_max": c_rng,
        "sqrt_wperp_min": float(w_perp_std.min()),
        "sqrt_wperp_max_finite": float(np.max(w_perp_std[np.isfinite(w_perp_std)])),
        "tau_grid": tau.tolist(),
        "sqrt_wperp": w_perp_std.tolist(),
        "wperp_var": w_perp_var.tolist(),
    }
    (out_dir / "data" / "G10_transverse_width.json").write_text(
        json.dumps(out, indent=2, default=lambda x: float(x))
    )
    print(f"\nSaved: {out_dir}/data/G10_transverse_width.json")

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(tau, w_perp_std, "-", label=r"$\sqrt{w_\perp(\tau)} = \sigma_{\rm SDE}/\sqrt{2|\lambda_\perp|}$")
    ax.axhline(sigma_dsm, color="red", ls="--", label=fr"$\sigma_{{\rm DSM}} = {sigma_dsm}$")
    ax.axhline(sigma_min, color="orange", ls=":", label=fr"$\sigma_{{\rm min}} = {sigma_min}$")
    ax.axhline(sigma_max, color="purple", ls=":", label=fr"$\sigma_{{\rm max}} = {sigma_max}$")
    ax.axvline(tau_crit, color="grey", ls="--", lw=0.7, label=r"$\tau_{\rm crit}$")
    ax.set_xlabel(r"$\tau$"); ax.set_ylabel(r"transverse width $\sqrt{w_\perp}$ (log)")
    ax.set_title("G-10 — transverse conditional width vs DSM σ")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "figures" / "G10_transverse_width.pdf")
    fig.savefig(out_dir / "figures" / "G10_transverse_width.png", dpi=150)
    print(f"       {out_dir}/figures/G10_transverse_width.pdf")


if __name__ == "__main__":
    main()
