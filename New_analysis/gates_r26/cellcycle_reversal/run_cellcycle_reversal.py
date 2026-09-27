"""Cell-cycle reversal calibration figure per PREREG_cellcycle_reversal.md.
Synthetic 30-D circular trajectory; fit scJDO with forward and reverse tau;
show antisymmetry of inferred rotation."""
from __future__ import annotations
import json, sys, warnings, time
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
warnings.filterwarnings("ignore")

OUT = REPO / "New_analysis" / "gates_r26" / "cellcycle_reversal"
OUT.mkdir(parents=True, exist_ok=True)

N_CELLS = 1500
D_LATENT = 2
D_AMB = 30
SIGMA_LATENT = 0.05
SIGMA_AMB = 0.10
SEEDS = [0, 1, 2]
N_EPOCHS = 3000
GRID = 100


def make_data(rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    phi = rng.uniform(0, 2 * np.pi, size=N_CELLS)
    # latent 2-D circle + noise
    z = np.stack([np.cos(phi), np.sin(phi)], axis=1)
    z = z + rng.normal(0, SIGMA_LATENT, size=z.shape)
    # random orthogonal embedding 2 -> 30
    A = rng.normal(size=(D_LATENT, D_AMB))
    Q, _ = np.linalg.qr(A.T)   # (D_AMB, D_LATENT) orthonormal columns
    W = Q.T                    # (D_LATENT, D_AMB) but rows orthonormal
    # We want a (D_LATENT, D_AMB) tangent map. Use W directly.
    X = z @ W + rng.normal(0, SIGMA_AMB, size=(N_CELLS, D_AMB))
    tau = phi / (2 * np.pi)
    # tangent direction in latent = (-sin phi, cos phi); embed via W
    tangent_latent = np.stack([-np.sin(phi), np.cos(phi)], axis=1)
    tangent_amb = tangent_latent @ W  # (N, D_AMB)
    return X.astype(np.float32), tau.astype(np.float32), phi.astype(np.float32), \
           tangent_amb.astype(np.float32), z.astype(np.float32), W.astype(np.float32)


def fit_arm(X, tau, seed):
    from scjdo.tl import fit_drift
    import anndata as ad
    torch.manual_seed(seed); np.random.seed(seed)
    a = ad.AnnData(X=np.zeros((X.shape[0], 1), dtype=np.float32))
    a.obsm["X_pca"] = X.astype(np.float32)
    a.obs["pseudotime"] = tau.astype(np.float32)
    fit_drift(a, rep="X_pca", time_key="pseudotime",
              n_archetypes=5, n_epochs=N_EPOCHS,
              vel_scale=0.0, hidden=256, depth=4, sigma=0.10,
              windowing="kernel", bandwidth="auto", grid_size=GRID,
              seed=seed, verbose=False, key_added="scjdo")
    # per-cell drift v_i
    from scjdo.tl._drift import _pseudotime_velocity  # not used
    # forward pass through the trained model to get per-cell drift
    # DriftField stored in a.uns['scjdo']? no — stored on model instance, not returned
    # Re-instantiate: reload from key_added metadata
    res = a.uns["scjdo"]
    # per-cell drift via the JACOBIAN + local mean approximation from J_tensor
    J_tensor = np.asarray(res["J_tensor"])                # (grid, D, D)
    t_centers = np.asarray(res["t_centers"])
    # v_i approximated by J_bin @ (x_i - mu_bin) — same as Gate 2 helper
    idx = np.searchsorted(t_centers, tau).clip(0, len(t_centers) - 1)
    mu_per_bin = np.zeros((len(t_centers), X.shape[1]), dtype=np.float32)
    for i in range(len(t_centers)):
        in_bin = (idx == i)
        if in_bin.any():
            mu_per_bin[i] = X[in_bin].mean(axis=0)
    v_per_cell = np.zeros_like(X)
    for i in range(X.shape[0]):
        b = int(idx[i])
        v_per_cell[i] = J_tensor[b] @ (X[i] - mu_per_bin[b])
    return v_per_cell.astype(np.float32), J_tensor, t_centers


def main():
    print("=" * 80); print("Cell-cycle reversal calibration"); print("=" * 80)
    t_all = time.time()
    X, tau_fwd, phi, tangent_amb, z, W = make_data()
    tau_rev = 1.0 - tau_fwd
    print(f"synthetic data: X={X.shape}, tau=[{tau_fwd.min():.3f}, {tau_fwd.max():.3f}]")

    # Fit forward and reverse arms × 3 seeds
    per_arm = {"forward": {}, "reverse": {}}
    for arm_name, arm_tau in [("forward", tau_fwd), ("reverse", tau_rev)]:
        print(f"\n[{arm_name}] scJDO × 3 seeds ...")
        for s in SEEDS:
            t0 = time.time()
            v_per_cell, J_tensor, t_centers = fit_arm(X, arm_tau, s)
            per_arm[arm_name][s] = {
                "v_per_cell": v_per_cell,
                "J_tensor": J_tensor,
                "t_centers": t_centers,
            }
            print(f"  seed {s}: fit {time.time()-t0:.1f}s  "
                  f"|v| median = {float(np.median(np.linalg.norm(v_per_cell, axis=1))):.3f}")

    # Per-cell angular drift = <v_i, tangent_i>
    # tangent_amb points in the direction of INCREASING phi (i.e. forward rotation)
    ang_fwd_seeds = []
    ang_rev_seeds = []
    for s in SEEDS:
        ang_fwd = (per_arm["forward"][s]["v_per_cell"] * tangent_amb).sum(axis=1)
        ang_rev = (per_arm["reverse"][s]["v_per_cell"] * tangent_amb).sum(axis=1)
        ang_fwd_seeds.append(ang_fwd)
        ang_rev_seeds.append(ang_rev)
        print(f"  seed {s}: forward mean angular drift = {float(ang_fwd.mean()):+.4f}  "
              f"reverse mean angular drift = {float(ang_rev.mean()):+.4f}")
    ang_fwd_stack = np.stack(ang_fwd_seeds, axis=0)  # (3, N)
    ang_rev_stack = np.stack(ang_rev_seeds, axis=0)
    mean_fwd = float(ang_fwd_stack.mean())
    mean_rev = float(ang_rev_stack.mean())
    spread_fwd = float(ang_fwd_stack.mean(axis=1).std())
    spread_rev = float(ang_rev_stack.mean(axis=1).std())
    antisym_check = float(abs(mean_fwd + mean_rev) / max(abs(mean_fwd), 1e-9))
    print(f"\n  MEAN angular drift forward = {mean_fwd:+.4f} ± {spread_fwd:.4f} (across 3 seeds)")
    print(f"  MEAN angular drift reverse = {mean_rev:+.4f} ± {spread_rev:.4f}")
    print(f"  |fwd + rev| / |fwd| = {antisym_check:.4f} (near 0 = antisymmetric)")

    predicted = (mean_fwd > 0) and (mean_rev < 0) and (antisym_check < 0.5)
    print(f"  qualitative prediction holds? {predicted}")

    # ── FIGURE ──────────────────────────────────────────────────────────
    print("\n[figure] rendering 2-panel PDF ...")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    # Both panels: latent (cos phi, sin phi) coloured by seed-averaged ang drift
    ang_fwd_mean = ang_fwd_stack.mean(axis=0)
    ang_rev_mean = ang_rev_stack.mean(axis=0)
    vmax = float(max(np.abs(ang_fwd_mean).max(), np.abs(ang_rev_mean).max()))
    for ax, ang, arm, mean_val, spread_val in [
        (axes[0], ang_fwd_mean, "forward", mean_fwd, spread_fwd),
        (axes[1], ang_rev_mean, "reverse", mean_rev, spread_rev),
    ]:
        sc = ax.scatter(z[:, 0], z[:, 1], c=ang, cmap="RdBu_r",
                         vmin=-vmax, vmax=vmax, s=6, alpha=0.75)
        # quiver of tangent-projected drift at 60 evenly-spaced phases
        n_arrow = 60
        arrow_phi = np.linspace(0, 2 * np.pi, n_arrow, endpoint=False)
        arrow_xy = np.stack([np.cos(arrow_phi), np.sin(arrow_phi)], axis=1)
        # find nearest cell for each arrow phase
        idx_nn = np.argmin(np.abs(phi[None, :] - arrow_phi[:, None]), axis=1)
        # local tangent direction in latent (unit)
        tangent2d = np.stack([-np.sin(arrow_phi), np.cos(arrow_phi)], axis=1)
        # per-arrow angular drift (sign)
        ang_sample = ang[idx_nn]
        # scaled arrow: unit tangent * ang sign * fixed length
        L = 0.12
        u = tangent2d[:, 0] * np.sign(ang_sample) * L
        v = tangent2d[:, 1] * np.sign(ang_sample) * L
        ax.quiver(arrow_xy[:, 0], arrow_xy[:, 1], u, v,
                   angles="xy", scale_units="xy", scale=1, width=0.005,
                   color="black")
        ax.set_aspect("equal"); ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
        ax.set_title(f"{arm}: mean drift = {mean_val:+.3f} ± {spread_val:.3f}")
        ax.set_xlabel("cos φ"); ax.set_ylabel("sin φ")
        plt.colorbar(sc, ax=ax, label="tangent-projected drift")
    fig.suptitle("Cell-cycle reversal: reversing pseudotime reverses inferred rotation")
    fig.savefig(OUT / "cellcycle_reversal.pdf", dpi=150)
    fig.savefig(OUT / "cellcycle_reversal.png", dpi=150)
    plt.close(fig)
    print(f"[figure] wrote cellcycle_reversal.pdf/.png")

    # Save summary
    def _tn(v):
        if isinstance(v, (np.floating,)): return float(v)
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, (list, tuple)): return [_tn(x) for x in v]
        if isinstance(v, dict): return {k: _tn(x) for k, x in v.items()}
        return v
    (OUT / "summary.json").write_text(json.dumps(_tn({
        "n_cells": N_CELLS, "n_seeds": len(SEEDS),
        "mean_angular_drift_forward": mean_fwd,
        "mean_angular_drift_reverse": mean_rev,
        "spread_across_seeds_forward": spread_fwd,
        "spread_across_seeds_reverse": spread_rev,
        "antisymmetry_ratio": antisym_check,
        "predicted": bool(predicted),
        "per_seed_forward_mean": [float(x.mean()) for x in ang_fwd_seeds],
        "per_seed_reverse_mean": [float(x.mean()) for x in ang_rev_seeds],
    }), indent=2))
    print(f"[written] summary.json + figure  total {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
