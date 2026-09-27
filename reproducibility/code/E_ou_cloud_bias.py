"""
Hypothesis E — Kernel-aggregated Jacobian has an OU-cloud bias that
diverges near the pitchfork.

Setup
-----
Around the symmetric fixed point (s(α), s(α)), the Ornstein–Uhlenbeck
approximation is (in the eigen-basis (s, w)):

    ds/dt  = λ_∥ · (s - s*) + σ dW
    dw/dt  = λ_⊥ · w        + σ dW

Its stationary covariance is

    Var(s - s*) = σ²/(2 |λ_∥|),
    Var(w)      = σ²/(2 |λ_⊥|)          ← diverges at the pitchfork.

The Jacobian ``J(x, y; α)`` is a smooth function of (x, y). A
Nadaraya–Watson kernel average of J at a fixed τ picks up the density's
first *and* second moments in (x, y). The second-moment contribution is

    E[J(z)] = J(z̄) + ½ tr( Cov(z) · ∇²J(z̄) ) + o(σ²).

The Hessian entries of J that couple to Var(w) are the ones involving
∂²J/∂w², i.e. the eigenvalue's *curvature* in w. Near the pitchfork
these are order unity while Var(w) is order 1/|λ_⊥|, so the bias in
E[Re λ_max(J)] scales as 1/|λ_⊥|.

Bottom line: **the NW-aggregated λ is inflated by a term that diverges
as 1/|λ_⊥|.** In particular, the aggregated λ can be positive even when
the true λ at s* is negative — moving the apparent "crossing" forward
in τ. The observed 0.17-0.22 offset in T2/T4/T5 fits this shape.

Test
----
1. Compute the analytic bias curve  b(α) = ½ tr( diag(σ²/(2|λ_∥|),
   σ²/(2|λ_⊥|)) · ∇²J(s*, s*; α) ).  Overlay on the empirical
   NW-aggregated λ curve. Do they match?
2. Test the 1/|λ_⊥| scaling: the empirical bias
   should scale linearly with 1/|λ_⊥(α)|. Fit and report slope.
3. Local-linear regression Jacobian: fit
       f_i ≈ β_0 + Σ β_k (z_i - z_c)_k
   in a τ-window and read J = β_{1:D}. LLR removes the second-moment
   bias by construction (it's the first-order Taylor coefficient, not
   the local mean). Check whether the LLR aggregated λ tracks the
   analytic λ_⊥(α) — i.e. whether it crosses at τ_crit.

Outputs
-------
    reproducibility/data/E_ou_cloud_bias.json
    reproducibility/data/E_ou_cloud_bias.npz
    reproducibility/figures/E_ou_cloud_bias.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import toggle_truth as gt
from crossing_robust import robust_crossing


# ---------------------------------------------------------------------------
# Analytic bias derivation.  We work in the (x, y) coords for concreteness.
# ---------------------------------------------------------------------------

def _J_at(x, y, alpha):
    a = -4.0 * alpha * y ** 3 / (1.0 + y ** 4) ** 2
    b = -4.0 * alpha * x ** 3 / (1.0 + x ** 4) ** 2
    return np.array([[-1.0, a], [b, -1.0]])


def _dJ_dy(x, y, alpha):
    """d(J)/dy — 2x2 matrix of partial derivatives of each J entry wrt y."""
    a_y = -4.0 * alpha * (3 * y ** 2 * (1 + y ** 4) ** 2
                          - y ** 3 * 2 * (1 + y ** 4) * 4 * y ** 3) / (1 + y ** 4) ** 4
    # d(J[0,0])/dy = 0,  d(J[0,1])/dy = a_y,  d(J[1,0])/dy = 0,  d(J[1,1])/dy = 0
    return np.array([[0.0, a_y], [0.0, 0.0]])


def _dJ_dx(x, y, alpha):
    b_x = -4.0 * alpha * (3 * x ** 2 * (1 + x ** 4) ** 2
                          - x ** 3 * 2 * (1 + x ** 4) * 4 * x ** 3) / (1 + x ** 4) ** 4
    return np.array([[0.0, 0.0], [b_x, 0.0]])


def _d2J(x, y, alpha):
    """Compute the four second derivatives ∂²J/∂x², ∂²J/∂y², ∂²J/(∂x∂y).

    Returns dict with keys 'xx', 'yy', 'xy'. Each value is a 2x2 array of the
    per-entry second derivative of J.
    """
    # J[0,1] = a(y) = -4 α y³ / (1+y⁴)². Let g(y) = y³ / (1+y⁴)². Then a = -4αg,
    #   g'(y) = (3y²(1+y⁴)² − y³·2(1+y⁴)·4y³)/(1+y⁴)⁴ = (3y²(1+y⁴) − 8y⁶) / (1+y⁴)³
    #   g''(y): differentiate  h(y) = 3y²(1+y⁴) - 8y⁶ = 3y² + 3y⁶ - 8y⁶ = 3y² - 5y⁶
    #     h'(y) = 6y - 30y⁵
    #     g''(y) = [h'(y)(1+y⁴)³ − h(y)·3(1+y⁴)²·4y³]/(1+y⁴)⁶
    #            = [h'(y)(1+y⁴) − 12y³ h(y)] / (1+y⁴)⁴
    hy = 3.0 * y ** 2 - 5.0 * y ** 6
    hyp = 6.0 * y - 30.0 * y ** 5
    g_pp_y = (hyp * (1.0 + y ** 4) - 12.0 * y ** 3 * hy) / (1.0 + y ** 4) ** 4
    a_yy = -4.0 * alpha * g_pp_y

    hx = 3.0 * x ** 2 - 5.0 * x ** 6
    hxp = 6.0 * x - 30.0 * x ** 5
    g_pp_x = (hxp * (1.0 + x ** 4) - 12.0 * x ** 3 * hx) / (1.0 + x ** 4) ** 4
    b_xx = -4.0 * alpha * g_pp_x

    xx = np.array([[0.0, 0.0], [b_xx, 0.0]])
    yy = np.array([[0.0, a_yy], [0.0, 0.0]])
    xy = np.zeros((2, 2))
    return {"xx": xx, "yy": yy, "xy": xy}


def _cov_ou_at_fp(alpha, sigma=gt.SIGMA_SDE):
    """Stationary OU covariance (2×2) of z − z* in (x, y) coords.

    In the (s, w) eigen-basis the variance is diag(σ²/(2|λ_∥|),
    σ²/(2|λ_⊥|)). Transforming back to (x, y):
        [ x - x* ]      [ 1/√2   1/√2 ] [ s ]
        [ y - y* ]  =   [ 1/√2  -1/√2 ] [ w ]
    So Cov(x, y) = R diag(V_s, V_w) R.T with R = (1/√2) [[1, 1], [1, -1]].
    """
    lam_par = gt.symmetric_eigenvalue(alpha)   # always < 0
    lam_perp = gt.transverse_eigenvalue(alpha)  # crosses 0 at α_crit
    # Only valid on the STABLE side of the pitchfork: λ_⊥ < 0. Past the
    # pitchfork the symmetric fp is a saddle; the OU approximation there is
    # ill-defined and Var(w) is infinite. We report the pre-transition regime.
    if lam_perp >= -1e-12:
        return None
    V_s = gt.SIGMA_SDE ** 2 / (2.0 * abs(lam_par))
    V_w = gt.SIGMA_SDE ** 2 / (2.0 * abs(lam_perp))
    R = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    Cov = R @ np.diag([V_s, V_w]) @ R.T
    return Cov


def analytic_bias(alpha):
    """
    Second-order Taylor expansion of E[J(z)] − J(z*) under the OU cloud.

    E[J] − J(z*) = ½ Σ_{a,b} Cov[a,b] ∂²J/∂z_a∂z_b.
    We return the aggregated (2×2) matrix and the induced Re(λ_max) bias.
    """
    Cov = _cov_ou_at_fp(alpha)
    if Cov is None:
        return None, float("nan")
    s = gt.sym_fp(alpha)
    d2 = _d2J(s, s, alpha)
    bias_matrix = 0.5 * (Cov[0, 0] * d2["xx"]
                          + Cov[1, 1] * d2["yy"]
                          + 2.0 * Cov[0, 1] * d2["xy"])
    J_fp = _J_at(s, s, alpha)
    J_biased = J_fp + bias_matrix
    lam_max = float(np.real(np.linalg.eigvals(J_biased)).max())
    lam_true = float(np.real(np.linalg.eigvals(J_fp)).max())
    return {
        "alpha": alpha,
        "s_star": s,
        "lam_perp_true": gt.transverse_eigenvalue(alpha),
        "Var_w": Cov[0, 0] + Cov[1, 1] - 2 * Cov[0, 1],  # = 2·V_w/2 = V_w (in original coords)
        "bias_max_Re_lam": lam_max - lam_true,
        "biased_max_Re_lam": lam_max,
        "true_max_Re_lam": lam_true,
    }, lam_max - lam_true


# ---------------------------------------------------------------------------
# Local-linear-regression Jacobian
# ---------------------------------------------------------------------------

def ll_jacobian_at_tau(z, f, tau, tau_c, tau_bandwidth):
    """
    Fit the local-linear regression
        f_i ≈ β_0 + Σ_k β_k (z_i - z̄)_k
    with Gaussian τ-weights of width ``tau_bandwidth`` centered on ``tau_c``.
    Return β_{1:D} ∈ R^{D×D} as J at that τ_c.
    """
    w = np.exp(-0.5 * ((tau - tau_c) / tau_bandwidth) ** 2)
    W = w.sum()
    if W < 1e-9:
        return None, 0.0
    n_eff = W ** 2 / (w * w).sum()
    zbar = (w[:, None] * z).sum(0) / W
    fbar = (w[:, None] * f).sum(0) / W
    dz = z - zbar
    df = f - fbar
    # Solve  Σ w_i dz_i dz_i.T · β = Σ w_i dz_i df_i.T
    A = (w[:, None, None] * dz[:, :, None] * dz[:, None, :]).sum(0)
    B = (w[:, None, None] * dz[:, :, None] * df[:, None, :]).sum(0)
    # β = A^{-1} B, but B[a, b] = Σ w_i dz_i[a] df_i[b], so β[a, b] = ∂ f_b / ∂ z_a
    try:
        beta = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        return None, n_eff
    return beta.T, n_eff   # transpose so beta[b, a] = ∂ f_b / ∂ z_a  (D_out × D_in)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main():
    out_dir = Path("/Users/terooatt/Downloads/scJDO/reproducibility")

    # --- (E.1) Analytic bias curve  b(α) and E[Re λ] under OU cloud ---
    # Dense grid: 800 α values, half of them logarithmically approaching α_crit
    # from below where the bias explodes.
    a_lin = np.linspace(gt.ALPHA_MIN, gt.ALPHA_CRIT - 1e-4, 400)
    a_log = gt.ALPHA_CRIT - np.logspace(-6, np.log10(gt.ALPHA_CRIT - gt.ALPHA_MIN), 400)[::-1]
    alphas = np.sort(np.concatenate([a_lin, a_log]))
    taus = gt.tau_of_alpha(alphas)
    rows = []
    inv_lam_perp = []
    biases = []
    biased_lams = []
    true_lams = []
    for a in alphas:
        r, bias = analytic_bias(a)
        if r is None:
            biases.append(np.nan)
            biased_lams.append(np.nan)
            true_lams.append(np.nan)
            inv_lam_perp.append(np.nan)
        else:
            biases.append(bias)
            biased_lams.append(r["biased_max_Re_lam"])
            true_lams.append(r["true_max_Re_lam"])
            inv_lam_perp.append(1.0 / abs(r["lam_perp_true"]))
    biases = np.array(biases)
    biased_lams = np.array(biased_lams)
    true_lams = np.array(true_lams)
    inv_lam_perp = np.array(inv_lam_perp)

    # (E.2) Fit  bias  ~  const · 1/|λ_⊥|  on the well-defined pre-transition regime
    # Restrict to 1/|λ_⊥| ∈ [2, 500] — leaves the deep-adiabatic regime alone
    # (where OU correction is negligible) and the pitchfork-adjacent regime
    # (where higher-order terms dominate).
    m = np.isfinite(biases) & (inv_lam_perp < 500) & (inv_lam_perp > 2)
    slope, intercept = np.polyfit(inv_lam_perp[m], biases[m], 1)
    yhat = intercept + slope * inv_lam_perp[m]
    ss_res = np.sum((biases[m] - yhat) ** 2)
    ss_tot = np.sum((biases[m] - biases[m].mean()) ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
    print(f"[E.2] bias ≈ {slope:+.4f} · (1/|λ_⊥|) + {intercept:+.4f}   R² = {r2:.4f}")
    print(f"       (fit on {m.sum()} points with 2 < 1/|λ_⊥| < 500)")

    # τ (α) at which the biased λ curve crosses 0 (biased is negative deep in
    # the adiabatic regime because bias is small and λ_true < 0; then positive
    # bias inflates it earlier)
    tau_biased_cross = float("nan")
    valid_idx = np.where(np.isfinite(biased_lams))[0]
    if len(valid_idx) > 1:
        for j in range(1, len(valid_idx)):
            i = valid_idx[j]
            ip = valid_idx[j - 1]
            if biased_lams[ip] < 0 <= biased_lams[i]:
                # Linear interpolation for τ where biased_lam crosses 0
                t = biased_lams[ip] / (biased_lams[ip] - biased_lams[i])
                tau_biased_cross = float(taus[ip] + t * (taus[i] - taus[ip]))
                break
    print(f"[E.1] analytic biased-λ crossing at τ = {tau_biased_cross:.4f}   "
          f"(true τ_crit = {gt.TAU_CRIT:.4f})")

    # --- (E.3) Local-linear-regression Jacobian on simulated data ---
    print("\n[E.3] LLR Jacobian vs NW Jacobian on simulated data")
    from T2_lifted_truth import kernel_matrix_curve   # reuse NW aggregator

    n_cells = 4000
    z, tau = gt.simulate_v3(n_cells=n_cells, seed=42)
    alphas_pc = gt.alpha_of_tau(tau)
    f_true = np.stack([gt.toggle_drift(z_i[None].astype(np.float64), a_i)[0]
                        for z_i, a_i in zip(z, alphas_pc)])
    J_true_per_cell = np.stack([gt.toggle_jacobian_at(float(z_i[0]),
                                                       float(z_i[1]), float(a_i))
                                  for z_i, a_i in zip(z, alphas_pc)])

    grid = np.linspace(0.0, 1.0, 200)
    Jbar_nw, _ = kernel_matrix_curve(J_true_per_cell, tau, grid,
                                     bandwidth=0.04, n_eff_min=20.0)
    lam_nw = np.array([float(np.real(np.linalg.eigvals(J)).max())
                        if not np.isnan(J).any() else np.nan for J in Jbar_nw])

    lam_llr = np.full(grid.size, np.nan)
    for k, tc in enumerate(grid):
        J_llr, n_eff = ll_jacobian_at_tau(z.astype(np.float64),
                                            f_true.astype(np.float64),
                                            tau, tc, tau_bandwidth=0.04)
        if J_llr is None or n_eff < 20.0:
            continue
        lam_llr[k] = float(np.real(np.linalg.eigvals(J_llr)).max())

    # Detect crossings with the sustained-run detector (min_run = 5)
    tau_nw_cross, code_nw = robust_crossing(lam_nw, grid, threshold=0.0, min_run=5)
    tau_llr_cross, code_llr = robust_crossing(lam_llr, grid, threshold=0.0, min_run=5)
    print(f"   NW crossing:  τ̂ = {tau_nw_cross:.4f}   ({code_nw})   "
          f"|τ̂ − τ_crit| = {abs(tau_nw_cross - gt.TAU_CRIT):.4f}")
    print(f"   LLR crossing: τ̂ = {tau_llr_cross:.4f}   ({code_llr})   "
          f"|τ̂ − τ_crit| = {abs(tau_llr_cross - gt.TAU_CRIT):.4f}")

    out = {
        "tau_crit": gt.TAU_CRIT,
        "grid_step": float(grid[1] - grid[0]),
        "dlambda_dtau_at_taucrit": 4.4417,
        "sigma_sde": gt.SIGMA_SDE,
        "E1_biased_lambda_crossing_tau": tau_biased_cross,
        "E2_bias_vs_inv_lambda_perp": {
            "slope": float(slope), "intercept": float(intercept), "r2": float(r2),
            "n_fit_points": int(m.sum()),
        },
        "E3_llr_vs_nw": {
            "nw_crossing_tau": tau_nw_cross, "nw_code": code_nw,
            "nw_offset": abs(tau_nw_cross - gt.TAU_CRIT) if np.isfinite(tau_nw_cross) else float("nan"),
            "llr_crossing_tau": tau_llr_cross, "llr_code": code_llr,
            "llr_offset": abs(tau_llr_cross - gt.TAU_CRIT) if np.isfinite(tau_llr_cross) else float("nan"),
        },
    }

    # --- (E.4) Restrict-to-near-fp cells ---
    # If Hypothesis E is right, conditioning on cells near the symmetric line
    # (small |w|) suppresses the cloud-shape bias and shifts the aggregated
    # crossing closer to τ_crit.
    print("\n[E.4] Restrict to cells with |w| < w_thresh (near the symmetric line)")
    w_cells = (z[:, 0] - z[:, 1]) / np.sqrt(2.0)
    for w_thresh in [1.0, 0.5, 0.25, 0.1]:
        mask = np.abs(w_cells) < w_thresh
        if mask.sum() < 100:
            print(f"   w_thresh={w_thresh}: only {mask.sum()} cells — skipping")
            continue
        Jbar_nw_r, _ = kernel_matrix_curve(J_true_per_cell[mask], tau[mask], grid,
                                          bandwidth=0.04, n_eff_min=20.0)
        lam_nw_r = np.array([
            float(np.real(np.linalg.eigvals(J)).max()) if not np.isnan(J).any() else np.nan
            for J in Jbar_nw_r
        ])
        tau_r, code_r = robust_crossing(lam_nw_r, grid, threshold=0.0, min_run=5)
        print(f"   w_thresh={w_thresh}  n={mask.sum():5d}  "
              f"τ̂ = {tau_r:.4f}  ({code_r})  |τ̂ − τ_crit| = "
              f"{abs(tau_r - gt.TAU_CRIT):.4f}")
        out.setdefault("E4_restrict_near_fp", []).append({
            "w_thresh": w_thresh, "n_cells": int(mask.sum()),
            "tau_hat": tau_r, "code": code_r,
            "offset": abs(tau_r - gt.TAU_CRIT) if np.isfinite(tau_r) else float("nan"),
        })

    # `out` may have been mutated (E.4). Serialize with numpy sanitization.
    def _sanitize(x):
        if isinstance(x, dict):
            return {k: _sanitize(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_sanitize(v) for v in x]
        if isinstance(x, np.floating):
            return float(x)
        if isinstance(x, np.integer):
            return int(x)
        return x
    (out_dir / "data" / "E_ou_cloud_bias.json").write_text(
        json.dumps(_sanitize(out), indent=2)
    )
    np.savez_compressed(
        out_dir / "data" / "E_ou_cloud_bias.npz",
        alphas=alphas, taus=taus, biases=biases,
        biased_lams=biased_lams, true_lams=true_lams,
        inv_lam_perp=inv_lam_perp,
        grid=grid, lam_nw=lam_nw, lam_llr=lam_llr,
    )
    print(f"\nSaved: {out_dir}/data/E_ou_cloud_bias.json")

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].plot(taus, true_lams, "k-", label=r"true $\lambda_\perp(\alpha)$")
    ax[0].plot(taus, biased_lams, "C0-", label="OU-bias-inflated  E[Re λ]")
    ax[0].axhline(0, color="grey", lw=0.5, ls=":")
    ax[0].axvline(gt.TAU_CRIT, color="red", lw=0.8, ls="--", label=r"$\tau_{\rm crit}$")
    ax[0].axvline(tau_biased_cross, color="C1", lw=0.8, ls=":",
                   label=f"biased cross τ={tau_biased_cross:.3f}")
    ax[0].set_xlabel(r"$\tau$"); ax[0].set_ylabel(r"$\lambda$")
    ax[0].set_title("E.1 Analytic OU-bias")
    ax[0].legend(fontsize=7)

    ax[1].plot(inv_lam_perp[m], biases[m], "o", ms=3, alpha=0.5, label="analytic")
    xv = np.linspace(0, 200, 100)
    ax[1].plot(xv, slope * xv + intercept, "r-",
                 label=f"fit: {slope:.3f}/|λ_⊥| + {intercept:.3f}, R²={r2:.3f}")
    ax[1].set_xlabel(r"$1 / |\lambda_\perp|$")
    ax[1].set_ylabel("bias in $E[\\lambda_{\\max}]$")
    ax[1].set_title("E.2 Bias scaling")
    ax[1].legend(fontsize=8)

    ax[2].plot(grid, lam_nw, "C0-", label="NW Jacobian, sim.")
    ax[2].plot(grid, lam_llr, "C2-", label="LLR Jacobian, sim.")
    ax[2].plot(taus, true_lams, "k-", lw=1, alpha=0.6, label="analytic $\\lambda_\\perp(\\alpha)$")
    ax[2].axhline(0, color="grey", lw=0.5, ls=":")
    ax[2].axvline(gt.TAU_CRIT, color="red", lw=0.8, ls="--", label=r"$\tau_{\rm crit}$")
    ax[2].set_xlabel(r"$\tau$"); ax[2].set_ylabel(r"$\max \text{Re}\lambda$")
    ax[2].set_title("E.3 NW vs LLR on simulated toggle")
    ax[2].legend(fontsize=8)
    ax[2].set_xlim(0, 0.6)

    plt.tight_layout()
    fig.savefig(out_dir / "figures" / "E_ou_cloud_bias.pdf")
    fig.savefig(out_dir / "figures" / "E_ou_cloud_bias.png", dpi=150)
    print(f"       {out_dir}/figures/E_ou_cloud_bias.pdf")


if __name__ == "__main__":
    main()
