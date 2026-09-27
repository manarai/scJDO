"""Task 1 — Gate 2 redo temporal-contrast check per PREREG_Task1_contrast.md.
Reuses existing R/G/L Jacobian tensors from gate2_redo_arms.npz. No new fits."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "New_analysis" / "gates_r26" / "gate2_redo"


def leading_eigvec(M):
    """Signed leading eigenvector of a (D, D) matrix, sign-normalised."""
    w, v = np.linalg.eig(M.astype(np.float64))
    k = int(np.argmax(np.real(w)))
    vv = np.real(v[:, k])
    if vv[np.argmax(np.abs(vv))] < 0:
        vv = -vv
    return vv / (np.linalg.norm(vv) + 1e-12)


def contrast_profile(J):
    """|J(t) - Jbar|_F / |Jbar|_F per grid point."""
    Jbar = J.mean(axis=0)
    denom = np.linalg.norm(Jbar) + 1e-12
    return np.array([np.linalg.norm(J[i] - Jbar) / denom for i in range(J.shape[0])])


def sign_flip_cos(u, v):
    return abs(float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)))


def frob_corr(A, B):
    a = A.reshape(-1).astype(np.float64); b = B.reshape(-1).astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    print("=" * 80); print("Task 1 — Gate 2 redo temporal-contrast"); print("=" * 80)
    t0 = time.time()
    z = np.load(OUT / "gate2_redo_arms.npz", allow_pickle=True)
    print(f"loaded arms: keys = {list(z.keys())}")
    J_R = z["J_R"]                              # (150, 30, 30)
    v_ref = z["R_v_ref"]                        # (3060, 30)
    X_pca = z["X_pca"]                          # (3060, 30)
    tau = z["tau"]
    t_centers = z["t_centers"]
    seeds = [0, 1, 2]
    J_G = np.stack([z[f"G_seed{s}_J"] for s in seeds], axis=0)  # (3, 150, 30, 30)
    J_L = np.stack([z[f"L_seed{s}_J"] for s in seeds], axis=0)
    T, D = J_R.shape[0], J_R.shape[1]
    print(f"J_R {J_R.shape}, J_G {J_G.shape}, J_L {J_L.shape}")

    # ── M1 — temporal contrast per fit ─────────────────────────────────
    print("\n[M1] temporal contrast per fit")
    contr_R = contrast_profile(J_R)
    contr_G = np.stack([contrast_profile(J_G[s]) for s in range(3)], axis=0)  # (3, T)
    contr_L = np.stack([contrast_profile(J_L[s]) for s in range(3)], axis=0)
    med_R = float(np.median(contr_R))
    med_G = float(np.median(contr_G))
    med_L = float(np.median(contr_L))
    spread_G = float(np.std(np.median(contr_G, axis=1)))
    spread_L = float(np.std(np.median(contr_L, axis=1)))
    print(f"  median contrast R    = {med_R:.4f}")
    print(f"  median contrast G    = {med_G:.4f} (across-seed sd {spread_G:.4f})")
    print(f"  median contrast L    = {med_L:.4f} (across-seed sd {spread_L:.4f})")
    print(f"  contrast(L) / contrast(R) = {med_L / (med_R + 1e-12):.3f}  "
          f"(reading A threshold ≤ 1.2)")

    # ── M2 — per-window L vs R ─────────────────────────────────────────
    print("\n[M2] per-window L vs R")
    cos_LR = np.zeros((3, T)); frob_LR = np.zeros((3, T))
    for s in range(3):
        for i in range(T):
            v_R = leading_eigvec(J_R[i])
            v_L = leading_eigvec(J_L[s, i])
            cos_LR[s, i] = sign_flip_cos(v_R, v_L)
            frob_LR[s, i] = float(np.linalg.norm(J_L[s, i] - J_R[i]))
    cos_LR_avg = cos_LR.mean(axis=0)  # (T,) avg over 3 seeds
    frob_LR_avg = frob_LR.mean(axis=0)
    print(f"  |cos(v_L, v_R)| median-over-tau (avg over 3 L seeds) = {float(np.median(cos_LR_avg)):.4f}")
    print(f"  |cos(v_L, v_R)| mean-over-tau  = {float(cos_LR_avg.mean()):.4f}")
    print(f"  Frobenius(L, R) median = {float(np.median(frob_LR_avg)):.4f}")

    # ── M3 — global linear baseline ────────────────────────────────────
    print("\n[M3] global linear baseline J_lin")
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X_pca, v_ref)
    J_lin = lr.coef_.T.astype(np.float64)  # (D, D)
    v_lin = leading_eigvec(J_lin)
    print(f"  J_lin shape={J_lin.shape}, |J_lin|_F={np.linalg.norm(J_lin):.3f}")

    # cos(J_lin, J_arm(τ)) per grid point per seed
    def cos_arm(J_arm_ts):
        """J_arm_ts: (T, D, D) → (T,) |cos(leading v_lin, leading v_arm_ts[i])|"""
        out = np.zeros(J_arm_ts.shape[0])
        for i in range(J_arm_ts.shape[0]):
            v_arm = leading_eigvec(J_arm_ts[i])
            out[i] = sign_flip_cos(v_lin, v_arm)
        return out
    def frob_arm(J_arm_ts):
        out = np.zeros(J_arm_ts.shape[0])
        for i in range(J_arm_ts.shape[0]):
            out[i] = frob_corr(J_lin, J_arm_ts[i])
        return out

    cos_lin_R = cos_arm(J_R)
    frob_lin_R = frob_arm(J_R)
    cos_lin_G = np.stack([cos_arm(J_G[s]) for s in range(3)], axis=0).mean(axis=0)
    frob_lin_G = np.stack([frob_arm(J_G[s]) for s in range(3)], axis=0).mean(axis=0)
    cos_lin_L = np.stack([cos_arm(J_L[s]) for s in range(3)], axis=0).mean(axis=0)
    frob_lin_L = np.stack([frob_arm(J_L[s]) for s in range(3)], axis=0).mean(axis=0)
    print(f"  |cos(v_lin, v_R)| mean over τ = {float(cos_lin_R.mean()):.4f}")
    print(f"  |cos(v_lin, v_G)| mean over τ = {float(cos_lin_G.mean()):.4f}")
    print(f"  |cos(v_lin, v_L)| mean over τ = {float(cos_lin_L.mean()):.4f}")
    print(f"  Frob corr(J_lin, J_R) mean = {float(frob_lin_R.mean()):.4f}")
    print(f"  Frob corr(J_lin, J_G) mean = {float(frob_lin_G.mean()):.4f}")
    print(f"  Frob corr(J_lin, J_L) mean = {float(frob_lin_L.mean()):.4f}")

    # ── Reading rule ────────────────────────────────────────────────────
    ratio = med_L / (med_R + 1e-12)
    cos_ll_mean = float(cos_lin_L.mean())
    reading_A = (ratio <= 1.2) and (cos_ll_mean > 0.95)
    reading = "A" if reading_A else "B"
    print(f"\n[reading] contrast(L)/contrast(R) = {ratio:.3f}, "
          f"cos(J_lin, J_L) = {cos_ll_mean:.4f}")
    print(f"[reading] READING {reading}: "
          f"{'velocity supervision recovers the global linear response' if reading_A else 'velocity supervision yields a time-varying operator that removes seed dependence'}")

    # ── Downstream (Reading B only) ────────────────────────────────────
    heldout = None
    if not reading_A:
        print("\n[held-out] Reading B → predict held-out cells' labelled velocity")
        from sklearn.model_selection import KFold
        from sklearn.linear_model import RidgeCV
        from sklearn.metrics import r2_score
        kf = KFold(n_splits=5, shuffle=True, random_state=0)

        # per-cell features from J_L (per seed)
        def per_cell_vel_from_J(J_tensor, X, tau_arr, t_centers_arr):
            idx = np.searchsorted(t_centers_arr, tau_arr).clip(0, len(t_centers_arr) - 1)
            mu_per_bin = np.zeros((len(t_centers_arr), X.shape[1]), dtype=np.float32)
            for i in range(len(t_centers_arr)):
                in_bin = (idx == i)
                if in_bin.any():
                    mu_per_bin[i] = X[in_bin].mean(axis=0)
            v = np.zeros_like(X)
            for i in range(X.shape[0]):
                b = int(idx[i])
                v[i] = J_tensor[b] @ (X[i] - mu_per_bin[b])
            return v

        r2_JL = []  # 3 seeds × 5 folds = 15
        r2_expr = []  # 5 folds
        for s in seeds:
            v_JL_features = per_cell_vel_from_J(J_L[s], X_pca, tau, t_centers)
            for tr, te in kf.split(X_pca):
                # target: labelled velocity (v_ref)
                # feature source: per-cell J_L features (or X_pca for expression baseline)
                # Ridge regression predicting v_ref from v_JL_features
                model = RidgeCV(alphas=np.logspace(-3, 3, 13))
                model.fit(v_JL_features[tr], v_ref[tr])
                pred = model.predict(v_JL_features[te])
                r2_JL.append(r2_score(v_ref[te], pred, multioutput="variance_weighted"))
        for tr, te in kf.split(X_pca):
            model = RidgeCV(alphas=np.logspace(-3, 3, 13))
            model.fit(X_pca[tr], v_ref[tr])
            pred = model.predict(X_pca[te])
            r2_expr.append(r2_score(v_ref[te], pred, multioutput="variance_weighted"))
        r2_JL = np.array(r2_JL); r2_expr = np.array(r2_expr)
        # bootstrap CI
        rng = np.random.default_rng(0)
        boot_JL = [float(np.mean(r2_JL[rng.integers(0, len(r2_JL), size=len(r2_JL))])) for _ in range(2000)]
        boot_expr = [float(np.mean(r2_expr[rng.integers(0, len(r2_expr), size=len(r2_expr))])) for _ in range(2000)]
        heldout = {
            "R2_J_L": {"mean": float(r2_JL.mean()), "sd": float(r2_JL.std()),
                        "ci95": (float(np.percentile(boot_JL, 2.5)),
                                  float(np.percentile(boot_JL, 97.5))),
                        "n": len(r2_JL)},
            "R2_ridge_expr": {"mean": float(r2_expr.mean()), "sd": float(r2_expr.std()),
                                "ci95": (float(np.percentile(boot_expr, 2.5)),
                                          float(np.percentile(boot_expr, 97.5))),
                                "n": len(r2_expr)},
        }
        print(f"  R² from J_L features → v_ref: mean={heldout['R2_J_L']['mean']:+.4f} "
              f"CI95 {heldout['R2_J_L']['ci95']} (n={heldout['R2_J_L']['n']})")
        print(f"  R² from RidgeCV(X_pca) → v_ref: mean={heldout['R2_ridge_expr']['mean']:+.4f} "
              f"CI95 {heldout['R2_ridge_expr']['ci95']} (n={heldout['R2_ridge_expr']['n']})")

    # ── Save ────────────────────────────────────────────────────────────
    def _tn(v):
        if isinstance(v, (np.floating,)): return float(v)
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, (list, tuple)): return [_tn(x) for x in v]
        if isinstance(v, dict): return {k: _tn(x) for k, x in v.items()}
        return v
    payload = _tn({
        "reading": reading,
        "M1_temporal_contrast": {
            "R_median": med_R, "R_profile": contr_R,
            "G_median_per_seed": np.median(contr_G, axis=1),
            "G_median_mean": med_G, "G_median_sd": spread_G,
            "L_median_per_seed": np.median(contr_L, axis=1),
            "L_median_mean": med_L, "L_median_sd": spread_L,
            "contrast_L_over_R": ratio,
        },
        "M2_per_window": {
            "cos_LR_mean_over_tau": float(cos_LR_avg.mean()),
            "cos_LR_median_over_tau": float(np.median(cos_LR_avg)),
            "frob_LR_median": float(np.median(frob_LR_avg)),
            "cos_LR_profile_seed_avg": cos_LR_avg,
            "frob_LR_profile_seed_avg": frob_LR_avg,
        },
        "M3_global_linear_baseline": {
            "cos_lin_R_mean": float(cos_lin_R.mean()),
            "cos_lin_G_mean": float(cos_lin_G.mean()),
            "cos_lin_L_mean": float(cos_lin_L.mean()),
            "frob_corr_lin_R_mean": float(frob_lin_R.mean()),
            "frob_corr_lin_G_mean": float(frob_lin_G.mean()),
            "frob_corr_lin_L_mean": float(frob_lin_L.mean()),
        },
        "heldout": heldout,
        "n_seeds": len(seeds), "T": T, "D": D,
    })
    (OUT / "gate2_task1_contrast.json").write_text(json.dumps(payload, indent=2))

    # ── Figure ──────────────────────────────────────────────────────────
    print("\n[figure] rendering τ profile figure ...")
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), constrained_layout=True)
    t = t_centers
    # Panel 1: contrast profiles
    axes[0].plot(t, contr_R, "k-", lw=2, label="R (reference)")
    for s in range(3):
        axes[0].plot(t, contr_G[s], "-", color="steelblue", alpha=0.5,
                      label="G (geom-only)" if s == 0 else None)
        axes[0].plot(t, contr_L[s], "-", color="firebrick", alpha=0.5,
                      label="L (matching)" if s == 0 else None)
    axes[0].set_xlabel("pseudotime τ"); axes[0].set_ylabel("‖J(τ) − J̄‖_F / ‖J̄‖_F")
    axes[0].set_title(f"M1 temporal contrast profile "
                       f"(median: R={med_R:.3f}, G={med_G:.3f}, L={med_L:.3f})")
    axes[0].legend()
    # Panel 2: cos(L, R) per window
    axes[1].plot(t, cos_LR_avg, "-", color="firebrick", lw=2, label="mean over 3 L seeds")
    axes[1].axhline(0.5, color="grey", ls=":", label="ref line 0.5")
    axes[1].set_xlabel("pseudotime τ"); axes[1].set_ylabel("|cos(v_L, v_R)| leading")
    axes[1].set_title(f"M2 per-window L vs R (median = {float(np.median(cos_LR_avg)):.3f})")
    axes[1].set_ylim(0, 1); axes[1].legend()
    # Panel 3: cos(J_lin, J_arm)
    axes[2].plot(t, cos_lin_R, "-", color="black", lw=2, label=f"vs R (mean {cos_lin_R.mean():.3f})")
    axes[2].plot(t, cos_lin_G, "-", color="steelblue", lw=2, label=f"vs G (mean {cos_lin_G.mean():.3f})")
    axes[2].plot(t, cos_lin_L, "-", color="firebrick", lw=2, label=f"vs L (mean {cos_lin_L.mean():.3f})")
    axes[2].set_xlabel("pseudotime τ"); axes[2].set_ylabel("|cos(v_lin, v_arm)|")
    axes[2].set_title("M3 global linear J_lin vs arm-Jacobian leading eigvec")
    axes[2].set_ylim(0, 1); axes[2].legend()
    fig.suptitle(f"Task 1 — Gate 2 redo temporal-contrast (reading {reading})")
    fig.savefig(OUT / "fig_gate2_contrast.pdf", dpi=150)
    fig.savefig(OUT / "fig_gate2_contrast.png", dpi=150)
    plt.close(fig)
    print(f"[figure] wrote fig_gate2_contrast.pdf/.png")

    print(f"\ntotal {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
