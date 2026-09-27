"""
T2 — Positive control on the metric.

Build the lifted analytic drift  f_z(z_lat) = A g(A^+ z_lat + μ_z)
where g is the true 2-D toggle drift, A is the linear map from ground truth
to latent coordinates from ``latent_axes``, and A^+ is its Moore-Penrose
pseudoinverse. Feed this through the same readout pipeline used in the
manuscript (Nadaraya-Watson kernel aggregation, argmax over interior grid)
and record where the argmax lands.

Prediction (work order §4):
    Re(λ_max)(τ) restricted to the signal subspace is identically 0 before
    τ_crit and −1 + c(α(τ)) after. It has no interior maximum. argmax should
    land at τ = 1.0. Peak at ≈ 0.83 on truth would indicate a distinct bug.

We also compute:
    * argmax over the full latent spectrum (20-D)
    * argmax over the signal subspace only (col-span of A)
    * argmax over the antisymmetric direction only  (the T2.2 crossing metric)
so the size of the "18 noise eigenvalues" effect is measured, not assumed.

Outputs
-------
reproducibility/data/T2_lifted_truth.npz
    grid, alpha_grid,
    lam_max_full, lam_max_signal, lam_perp_analytic, lam_par_analytic,
    tau_argmax_full, tau_argmax_signal, tau_cross_perp
reproducibility/results/T2_lifted_truth_control.md
reproducibility/figures/T2_profile.pdf
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

import toggle_truth as gt


# ---------------------------------------------------------------------------
# Kernel aggregation — copied verbatim from saddle_readouts.py:113 so it is
# identical to the manuscript pipeline (Nadaraya-Watson, Gaussian kernel).
# ---------------------------------------------------------------------------

def kernel_curve(scalar_per_cell, t_true, grid, bandwidth=0.04, n_eff_min=20.0):
    curve = np.full(grid.shape, np.nan)
    n_eff_out = np.full(grid.shape, np.nan)
    for k, tau in enumerate(grid):
        w = np.exp(-((t_true.astype(np.float64) - tau) ** 2) / (2.0 * bandwidth ** 2))
        W = w.sum()
        if W < 1e-9:
            continue
        n_eff_out[k] = (W * W) / (w * w).sum()
        if n_eff_out[k] < n_eff_min:
            continue
        curve[k] = float((w * scalar_per_cell).sum() / W)
    return curve, n_eff_out


def kernel_matrix_curve(mat_per_cell, t_true, grid, bandwidth=0.04, n_eff_min=20.0):
    """Same as kernel_curve but for an (N, D, D) tensor."""
    N, D, _ = mat_per_cell.shape
    out = np.full((grid.size, D, D), np.nan, dtype=np.float64)
    n_eff = np.full(grid.size, np.nan)
    Flat = mat_per_cell.reshape(N, -1).astype(np.float64)
    for k, tau in enumerate(grid):
        w = np.exp(-((t_true.astype(np.float64) - tau) ** 2) / (2.0 * bandwidth ** 2))
        W = w.sum()
        if W < 1e-9:
            continue
        n_eff[k] = (W * W) / (w * w).sum()
        if n_eff[k] < n_eff_min:
            continue
        out[k] = (w @ Flat / W).reshape(D, D)
    return out, n_eff


def interior_argmax(curve, grid, low=0.05, high=0.95):
    mask = (grid >= low) & (grid <= high) & ~np.isnan(curve)
    if not mask.any():
        return float("nan")
    idx = np.where(mask)[0][np.argmax(curve[mask])]
    return float(grid[idx])


def crossing_tau(curve, grid, threshold=0.0):
    """First grid point where curve crosses ``threshold`` from below."""
    valid = ~np.isnan(curve)
    if not valid.any():
        return float("nan")
    below = curve < threshold
    # find the first False→True transition, i.e. first index i where curve[i]>=thr
    for i in range(len(curve)):
        if valid[i] and not below[i]:
            return float(grid[i])
    return float("nan")


# ---------------------------------------------------------------------------
# Build lifted analytic drift and its per-cell Jacobian
# ---------------------------------------------------------------------------

def lifted_analytic_readouts(seed=42, n_cells=1500, bandwidth=0.04, n_grid=200,
                              n_eff_min=20.0):
    z_true, t_true = gt.simulate_v3(n_cells=n_cells, seed=seed)
    W = gt.build_observation_map(seed=seed)
    Z_lat, A, pca = gt.latent_axes(z_true, W, seed=seed)

    # Ground-truth 2-D Jacobian per cell.
    # Two variants of the positive control:
    #   (a) 'at_cell' — evaluate J at each cell's actual (x_i, y_i, α(τ_i)),
    #       matching the manuscript pipeline exactly. This is what the metric
    #       is fed on a trained model.
    #   (b) 'at_fp'  — pretend each cell sits at the symmetric fixed point
    #       (s(α_i), s(α_i)); evaluate J there. This is the brief's §4
    #       prediction: λ_⊥ = max(0, −1 + c(α)), monotone in τ past τ_crit.
    # We run both.
    alphas = gt.alpha_of_tau(t_true)
    J2_per_cell = np.zeros((n_cells, 2, 2), dtype=np.float64)   # variant (a)
    J2_fp_per_cell = np.zeros((n_cells, 2, 2), dtype=np.float64)  # variant (b)
    for i, (z, a) in enumerate(zip(z_true, alphas)):
        J2_per_cell[i] = gt.toggle_jacobian_at(float(z[0]), float(z[1]), float(a))
        s_a = gt.sym_fp(float(a))
        J2_fp_per_cell[i] = gt.toggle_jacobian_at(s_a, s_a, float(a))

    # Lift to 20-D latent Jacobian.  A: (2, D_lat).
    # The lifted drift is  f_z(z_lat) = A g(A^+ z_lat + b), so
    # J_z = A J_2 A^+, which is a rank-2 matrix living in col(A).
    A_pinv = np.linalg.pinv(A).astype(np.float64)   # (D_lat, 2)
    D_lat = A.shape[1]
    Jlat_per_cell = np.zeros((n_cells, D_lat, D_lat), dtype=np.float64)
    Jlat_fp_per_cell = np.zeros((n_cells, D_lat, D_lat), dtype=np.float64)
    for i in range(n_cells):
        Jlat_per_cell[i] = A.T @ J2_per_cell[i] @ A_pinv.T
        Jlat_fp_per_cell[i] = A.T @ J2_fp_per_cell[i] @ A_pinv.T

    # Wait — the correct chain is:  y = z @ A  (row vectors), so ∂y/∂z_row = A.
    # And  z_est = z_lat @ A^+, so ∂z_est/∂z_lat_row = A^+.
    # Then f_z(z_lat) = A_col_conv @ g(A^+ z_lat_col_conv), and the Jacobian
    # in row-vector convention is  A_col.T @ J_2 @ (A^+_col).T = ...
    # The cleanest bookkeeping is to work column-vector throughout:
    #   f_z : R^D → R^D,  f_z(z) = A_col g(A^+_col z),   A_col = A.T (D, 2).
    #   ∂f_z/∂z = A_col J_2 A^+_col = A.T @ J_2 @ (A^+).T = A.T @ J_2 @ pinv(A.T).
    # That is what the loop above builds. Verify: rank <= 2.
    assert np.linalg.matrix_rank(Jlat_per_cell[0]) <= 2 + 1e-6

    # ── Aggregate onto the τ-grid ────────────────────────────────────────
    grid = np.linspace(0.0, 1.0, n_grid, dtype=np.float64)
    Jbar_lat, n_eff = kernel_matrix_curve(Jlat_per_cell, t_true, grid,
                                          bandwidth=bandwidth, n_eff_min=n_eff_min)
    Jbar_lat_fp, _ = kernel_matrix_curve(Jlat_fp_per_cell, t_true, grid,
                                         bandwidth=bandwidth, n_eff_min=n_eff_min)

    Q, _ = np.linalg.qr(A.T)   # (D, 2) orthonormal basis of signal subspace

    def _spectrum_over(mat_curve):
        """(n_grid, D, D) -> (full 20-D max Re λ, signal-subspace 2-D max Re λ)."""
        lam_full = np.full(n_grid, np.nan)
        lam_signal = np.full(n_grid, np.nan)
        for k in range(n_grid):
            if np.isnan(mat_curve[k]).any():
                continue
            lam_full[k] = float(np.real(np.linalg.eigvals(mat_curve[k])).max())
            J_sig = Q.T @ mat_curve[k] @ Q
            lam_signal[k] = float(np.real(np.linalg.eigvals(J_sig)).max())
        return lam_full, lam_signal

    lam_full, lam_signal = _spectrum_over(Jbar_lat)
    lam_full_fp, lam_signal_fp = _spectrum_over(Jbar_lat_fp)

    # Explicitly compute the analytic transverse eigenvalue at α(τ) using the
    # symmetric fp — this is the "truth" curve that the readout should trace.
    alpha_grid = gt.alpha_of_tau(grid)
    lam_perp_analytic = np.array([gt.transverse_eigenvalue(a) for a in alpha_grid])
    lam_par_analytic = np.array([gt.symmetric_eigenvalue(a) for a in alpha_grid])

    tau_argmax_full = interior_argmax(lam_full, grid)
    tau_argmax_signal = interior_argmax(lam_signal, grid)
    tau_argmax_full_fp = interior_argmax(lam_full_fp, grid)
    tau_argmax_signal_fp = interior_argmax(lam_signal_fp, grid)
    tau_cross_signal = crossing_tau(lam_signal, grid, threshold=0.0)
    tau_cross_signal_fp = crossing_tau(lam_signal_fp, grid, threshold=0.0)
    tau_cross_analytic = crossing_tau(lam_perp_analytic, grid, threshold=0.0)

    # Also record R3 P⊥ J P⊥  where f̂ = A f_2 / |A f_2|. In truth, f_2 is
    # the drift evaluated at each cell; do this per-cell and aggregate as in
    # the manuscript.
    f2_per_cell = gt.toggle_drift(z_true.astype(np.float64), alphas)
    f_lat_per_cell = f2_per_cell @ A               # (N, D_lat)
    lam_r3_per_cell = np.zeros(n_cells)
    lam_full_per_cell = np.zeros(n_cells)
    lam_signal_per_cell = np.zeros(n_cells)
    for i in range(n_cells):
        Ji = Jlat_per_cell[i]
        lam_full_per_cell[i] = float(np.real(np.linalg.eigvals(Ji)).max())
        J_sig_i = Q.T @ Ji @ Q
        lam_signal_per_cell[i] = float(np.real(np.linalg.eigvals(J_sig_i)).max())
        f = f_lat_per_cell[i]
        n = float(np.linalg.norm(f))
        if n < 1e-8:
            lam_r3_per_cell[i] = lam_full_per_cell[i]
            continue
        fh = f / n
        P = np.eye(D_lat) - np.outer(fh, fh)
        Jperp = P @ Ji @ P
        lam_r3_per_cell[i] = float(np.real(np.linalg.eigvals(Jperp)).max())

    lam_r3_curve, _ = kernel_curve(lam_r3_per_cell, t_true, grid,
                                    bandwidth=bandwidth, n_eff_min=n_eff_min)

    return {
        "grid": grid,
        "alpha_grid": alpha_grid,
        "lam_full": lam_full,
        "lam_signal": lam_signal,
        "lam_full_fp": lam_full_fp,
        "lam_signal_fp": lam_signal_fp,
        "lam_perp_analytic": lam_perp_analytic,
        "lam_par_analytic": lam_par_analytic,
        "lam_r3_curve": lam_r3_curve,
        "n_eff": n_eff,
        "tau_argmax_full": tau_argmax_full,
        "tau_argmax_signal": tau_argmax_signal,
        "tau_argmax_full_fp": tau_argmax_full_fp,
        "tau_argmax_signal_fp": tau_argmax_signal_fp,
        "tau_cross_signal": tau_cross_signal,
        "tau_cross_signal_fp": tau_cross_signal_fp,
        "tau_cross_analytic": tau_cross_analytic,
        "tau_argmax_r3": interior_argmax(lam_r3_curve, grid),
        "tau_crit": gt.TAU_CRIT,
        "seed": seed,
    }


def main():
    out_dir = Path("/Users/terooatt/Downloads/scJDO/reproducibility")

    all_seeds = []
    for seed in (42, 1, 2, 3, 4):
        r = lifted_analytic_readouts(seed=seed)
        all_seeds.append(r)
        print(
            f"[seed={seed}]  at_cell: argmax(full)={r['tau_argmax_full']:.3f}  "
            f"argmax(signal)={r['tau_argmax_signal']:.3f}  "
            f"argmax(R3)={r['tau_argmax_r3']:.3f}  "
            f"cross(signal)={r['tau_cross_signal']:.4f}  |  "
            f"at_fp: argmax(full)={r['tau_argmax_full_fp']:.3f}  "
            f"argmax(signal)={r['tau_argmax_signal_fp']:.3f}  "
            f"cross(signal_fp)={r['tau_cross_signal_fp']:.4f}  |  "
            f"cross(analytic)={r['tau_cross_analytic']:.4f}  τ_crit={r['tau_crit']:.4f}"
        )

    # Persist arrays for the seed=42 reference case
    r0 = all_seeds[0]
    np.savez_compressed(
        out_dir / "data" / "T2_lifted_truth.npz",
        grid=r0["grid"],
        alpha_grid=r0["alpha_grid"],
        lam_full=r0["lam_full"],
        lam_signal=r0["lam_signal"],
        lam_full_fp=r0["lam_full_fp"],
        lam_signal_fp=r0["lam_signal_fp"],
        lam_perp_analytic=r0["lam_perp_analytic"],
        lam_par_analytic=r0["lam_par_analytic"],
        lam_r3_curve=r0["lam_r3_curve"],
        n_eff=r0["n_eff"],
        tau_crit=r0["tau_crit"],
    )

    summary = {
        "tau_crit": gt.TAU_CRIT,
        "note": (
            "at_cell = evaluate J at each cell's actual (x,y,α) — matches "
            "manuscript pipeline. at_fp = pretend cell sits at (s(α), s(α)); "
            "matches the brief's §4 analytic prediction. crossing_analytic = "
            "τ where the closed-form −1+c(α(τ)) crosses 0."
        ),
        "seeds": [
            {
                "seed": r["seed"],
                "at_cell": {
                    "tau_argmax_full_spectrum_20D": r["tau_argmax_full"],
                    "tau_argmax_signal_subspace_2D": r["tau_argmax_signal"],
                    "tau_argmax_R3_transverse_projector": r["tau_argmax_r3"],
                    "tau_crossing_signal_subspace": r["tau_cross_signal"],
                },
                "at_fp": {
                    "tau_argmax_full_spectrum_20D": r["tau_argmax_full_fp"],
                    "tau_argmax_signal_subspace_2D": r["tau_argmax_signal_fp"],
                    "tau_crossing_signal_subspace": r["tau_cross_signal_fp"],
                },
                "tau_crossing_analytic_lambda_perp": r["tau_cross_analytic"],
            }
            for r in all_seeds
        ],
    }
    (out_dir / "data" / "T2_lifted_truth_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    # ── Plot the reference profile ─────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1, figsize=(7, 6.5), sharex=True)
    grid = r0["grid"]
    ax[0].plot(grid, r0["lam_perp_analytic"], "k-", lw=2, label=r"analytic $\lambda_\perp(\tau)=-1+c$")
    ax[0].plot(grid, r0["lam_par_analytic"], "k--", lw=1, label=r"analytic $\lambda_\parallel(\tau)=-1-c$")
    ax[0].axhline(0, color="grey", lw=0.5, ls=":")
    ax[0].axvline(r0["tau_crit"], color="red", lw=0.8, ls="--", label=r"$\tau_{\rm crit}$")
    ax[0].set_ylabel(r"eigenvalue at $\{(s,s)\}$")
    ax[0].legend(loc="best", fontsize=8)
    ax[0].set_title("T2: lifted-truth control — analytic transverse eigenvalue")

    ax[1].plot(grid, r0["lam_full"], "C0-", label=r"at_cell 20-D $\max \text{Re}\,\lambda$")
    ax[1].plot(grid, r0["lam_signal"], "C1-", label=r"at_cell signal-subspace $\max \text{Re}\,\lambda$")
    ax[1].plot(grid, r0["lam_full_fp"], "C3--", label=r"at_fp 20-D $\max \text{Re}\,\lambda$ (brief's §4)")
    ax[1].plot(grid, r0["lam_signal_fp"], "C4--", label=r"at_fp signal-subspace $\max \text{Re}\,\lambda$")
    ax[1].plot(grid, r0["lam_r3_curve"], "C2-", alpha=0.6, label=r"R3 = $\max \text{Re}\,\lambda(P_\perp J P_\perp)$")
    ax[1].axhline(0, color="grey", lw=0.5, ls=":")
    ax[1].axvline(r0["tau_crit"], color="red", lw=0.8, ls="--", label=r"$\tau_{\rm crit}$")
    ax[1].set_xlabel(r"pseudotime $\tau$")
    ax[1].set_ylabel(r"aggregated eigenvalue")
    ax[1].legend(loc="best", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "figures" / "T2_profile.pdf")
    fig.savefig(out_dir / "figures" / "T2_profile.png", dpi=150)

    print(f"\nSaved: {out_dir}/data/T2_lifted_truth.npz")
    print(f"       {out_dir}/data/T2_lifted_truth_summary.json")
    print(f"       {out_dir}/figures/T2_profile.pdf")


if __name__ == "__main__":
    main()
