"""
Empirical Hypothesis E — measure bias from SIMULATED clouds.

The earlier E analysis fit the analytic Taylor prediction to itself
and got R² = 1.0 trivially. The real question is whether the actual,
empirical bias produced by kernel-averaging J over cells scales as
1/|λ_⊥|.

Design
------
For each α in a swept grid:
  1. Simulate the SDE at CONSTANT α for a long time to reach the
     stationary distribution.  Discard the first `burn_in` samples.
  2. Compute J at each simulated cell:  J(z_i; α).
  3. Take the average  J̄  =  (1/N) Σ J(z_i).
  4. Compare  E[Re λ_max(J̄)]  to  λ_perp(α) = -1 + c(α).
     bias(α)  =  E[Re λ_max(J̄)]  −  λ_perp(α).
  5. On the STABLE (pre-transition) branch we have 1/|λ_⊥| finite and
     bias should scale as 1/|λ_⊥| by the OU-Taylor prediction.

Departures from the closed-form prediction (which was ½ tr(Cov ∇²J))
would surface here — higher-order terms, non-Gaussian tails, or
finite-sample effects.

Report
------
  * Empirical R² of a linear fit bias ~ 1/|λ_⊥|.
  * Where the scaling breaks down (which α values fall off the line).
  * Comparison to the analytic Taylor prediction over the same α range.

Outputs
-------
  reproducibility/data/E_empirical.json
  reproducibility/data/E_empirical.npz
  reproducibility/figures/E_empirical.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import toggle_truth as gt
from E_ou_cloud_bias import analytic_bias   # for comparison overlay


def simulate_stationary(alpha, n_cells=4000, burn_in=2000, thin=20,
                         dt=0.02, sigma_sde=gt.SIGMA_SDE, seed=0):
    """
    Simulate the SDE at CONSTANT α starting near the symmetric fp.  After
    ``burn_in`` steps, sample ``n_cells`` cells every ``thin`` steps.
    """
    rng = np.random.default_rng(seed)
    s = gt.sym_fp(alpha)
    # Warm-start slightly off-diagonal to break symmetry, so trajectories
    # can visit either branch past α_crit.
    z = np.full((n_cells, 2), s, dtype=np.float64)
    z += 0.02 * rng.standard_normal(z.shape)
    sqrt_dt = np.sqrt(dt)
    # Warmup — advance ALL cells through burn_in independent steps but with
    # different noise realizations, so each is an approximately-independent
    # sample from the stationary distribution after burn_in.
    for step in range(burn_in):
        f = gt.toggle_drift(z, alpha)
        z = z + f * dt + sigma_sde * sqrt_dt * rng.standard_normal(z.shape)
    return z


def empirical_bias(alpha, n_cells=4000, burn_in=2000, seed=0):
    """
    Simulate cells at CONSTANT α, compute per-cell J, average, extract
    max Re λ.  Return (bias, lam_true, lam_empirical, mean_z, var_w).
    """
    z = simulate_stationary(alpha, n_cells=n_cells, burn_in=burn_in, seed=seed)
    # Per-cell Jacobians
    J = np.zeros((z.shape[0], 2, 2))
    for i in range(z.shape[0]):
        J[i] = gt.toggle_jacobian_at(float(z[i, 0]), float(z[i, 1]), float(alpha))
    Jbar = J.mean(axis=0)
    lam_emp = float(np.real(np.linalg.eigvals(Jbar)).max())
    lam_true = float(gt.transverse_eigenvalue(alpha))
    w = (z[:, 0] - z[:, 1]) / np.sqrt(2.0)
    return lam_emp - lam_true, lam_true, lam_emp, z.mean(axis=0), float(np.var(w))


def main():
    out_dir = Path("/Users/terooatt/Downloads/scJDO/reproducibility")

    # Sample α values.  Pre-pitchfork (α < α_crit) — regime where the OU
    # approximation is well-defined and Var(w) < ∞.  Sub-critical only.
    # Log-spaced approach to α_crit from below.
    alpha_crit = gt.ALPHA_CRIT
    # Deep to shallow adiabatic:
    alphas_deep = np.linspace(0.3, alpha_crit - 0.15, 8)
    # Approaching α_crit:
    alphas_close = alpha_crit - np.logspace(-3, -0.5, 12)
    alphas = np.sort(np.unique(np.concatenate([alphas_deep, alphas_close])))

    rows = []
    for a in alphas:
        bias, lam_true, lam_emp, zbar, var_w = empirical_bias(
            a, n_cells=4000, burn_in=3000, seed=0,
        )
        inv_lam = 1.0 / abs(lam_true)
        # Analytic bias (Taylor closed-form) for comparison
        ana, _ = analytic_bias(a)
        ana_bias = ana["bias_max_Re_lam"] if ana is not None else float("nan")
        rows.append({
            "alpha": float(a), "lam_perp_true": lam_true,
            "lam_empirical": lam_emp,
            "empirical_bias": bias, "analytic_bias": ana_bias,
            "1/|lam_perp|": inv_lam, "var_w": var_w,
        })
        print(f"  α={a:.5f}  λ_⊥={lam_true:+.4f}  E[Re λ_max]={lam_emp:+.4f}  "
              f"bias(emp)={bias:+.4f}  bias(analytic)={ana_bias:+.4f}  "
              f"1/|λ_⊥|={inv_lam:.3f}   Var(w)={var_w:.4f}")

    inv_lam = np.array([r["1/|lam_perp|"] for r in rows])
    bias_emp = np.array([r["empirical_bias"] for r in rows])
    bias_ana = np.array([r["analytic_bias"] for r in rows])
    m = np.isfinite(inv_lam) & np.isfinite(bias_emp)
    slope, intercept = np.polyfit(inv_lam[m], bias_emp[m], 1)
    yhat = intercept + slope * inv_lam[m]
    ss_res = np.sum((bias_emp[m] - yhat) ** 2)
    ss_tot = np.sum((bias_emp[m] - bias_emp[m].mean()) ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
    print(f"\nEmpirical fit: bias ≈ {slope:+.4f} · (1/|λ_⊥|) + {intercept:+.4f}")
    print(f"R² = {r2:.4f}   (n={m.sum()})")

    # Where does the scaling break down?  Sort rows by 1/|λ_⊥| and report
    # residuals.
    order = np.argsort(inv_lam[m])
    residual = bias_emp[m] - (intercept + slope * inv_lam[m])
    print("\nResiduals sorted by 1/|λ_⊥|:")
    print(f"  {'α':>9s}   {'1/|λ⊥|':>10s}   {'bias(emp)':>10s}   "
          f"{'linear fit':>10s}   {'residual':>10s}   {'residual/bias':>13s}")
    for idx in order:
        actual_idx = np.where(m)[0][idx]
        r = rows[actual_idx]
        rr = residual[idx]
        rel = rr / max(abs(r["empirical_bias"]), 1e-6)
        print(f"  {r['alpha']:9.5f}   {inv_lam[actual_idx]:10.3f}   "
              f"{r['empirical_bias']:+10.4f}   "
              f"{intercept + slope*inv_lam[actual_idx]:+10.4f}   "
              f"{rr:+10.4f}   {rel:+13.3f}")

    out = {
        "tau_crit": gt.TAU_CRIT,
        "empirical_fit": {"slope": float(slope), "intercept": float(intercept),
                          "r2": float(r2), "n_points": int(m.sum())},
        "rows": rows,
    }
    (out_dir / "data" / "E_empirical.json").write_text(
        json.dumps(out, indent=2, default=lambda x: float(x))
    )
    np.savez_compressed(
        out_dir / "data" / "E_empirical.npz",
        alphas=alphas, inv_lam=inv_lam, bias_emp=bias_emp, bias_ana=bias_ana,
        slope=slope, intercept=intercept, r2=r2,
    )
    print(f"\nSaved: {out_dir}/data/E_empirical.json")

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].plot(inv_lam, bias_emp, "o-", ms=5, label="empirical")
    ax[0].plot(inv_lam, bias_ana, "s--", ms=4, alpha=0.7, label="analytic Taylor (closed-form)")
    xv = np.linspace(0, np.max(inv_lam) * 1.05, 100)
    ax[0].plot(xv, slope * xv + intercept, "r-", lw=1, alpha=0.7,
                label=f"linear fit: slope={slope:.3f}  R²={r2:.4f}")
    ax[0].set_xlabel(r"$1/|\lambda_\perp|$")
    ax[0].set_ylabel(r"bias in $E[\text{Re }\lambda_{\max}(\bar J)]$")
    ax[0].set_title("E (empirical): bias-vs-1/|λ_⊥| from actual cell clouds")
    ax[0].legend(fontsize=8)

    ax[1].plot(inv_lam, residual, "o-", ms=5)
    ax[1].axhline(0, color="grey", lw=0.5, ls=":")
    ax[1].set_xlabel(r"$1/|\lambda_\perp|$")
    ax[1].set_ylabel("residual (empirical − linear fit)")
    ax[1].set_title("Residuals show where the 1/|λ_⊥| law breaks down")
    plt.tight_layout()
    fig.savefig(out_dir / "figures" / "E_empirical.pdf")
    fig.savefig(out_dir / "figures" / "E_empirical.png", dpi=150)
    print(f"       {out_dir}/figures/E_empirical.pdf")


if __name__ == "__main__":
    main()
