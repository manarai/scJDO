"""Task 2 — add vel_scale=2.0 arms to cell-cycle reversal + 4-panel figure.
Per PREREG_Task2_cellcycle.md."""
from __future__ import annotations
import json, sys, warnings, time, pickle
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "reproducibility" / "gates_r26" / "cellcycle_reversal"))
warnings.filterwarnings("ignore")

from run_cellcycle_reversal import make_data, N_CELLS, N_EPOCHS, GRID, SEEDS

OUT = REPO / "reproducibility" / "gates_r26" / "cellcycle_reversal"


def fit_arm_vel(X, tau, seed, vel_scale):
    """Fit scJDO with the given vel_scale, return per-cell drift + J_tensor."""
    from scjdo.tl import fit_drift
    import anndata as ad
    torch.manual_seed(seed); np.random.seed(seed)
    a = ad.AnnData(X=np.zeros((X.shape[0], 1), dtype=np.float32))
    a.obsm["X_pca"] = X.astype(np.float32)
    a.obs["pseudotime"] = tau.astype(np.float32)
    fit_drift(a, rep="X_pca", time_key="pseudotime",
              n_archetypes=5, n_epochs=N_EPOCHS,
              vel_scale=vel_scale, hidden=256, depth=4, sigma=0.10,
              windowing="kernel", bandwidth="auto", grid_size=GRID,
              seed=seed, verbose=False, key_added="scjdo")
    res = a.uns["scjdo"]
    J_tensor = np.asarray(res["J_tensor"]).astype(np.float32)
    t_centers = np.asarray(res["t_centers"]).astype(np.float32)
    # per-cell drift v_i = J(τ_i) @ (x_i - μ_bin_i)
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


def A12_profile(J_tensor, W):
    """(T, D, D) → (T,) A[0,1] on the cycle plane."""
    T = J_tensor.shape[0]
    out = np.zeros(T, dtype=np.float64)
    for k in range(T):
        J2d = W @ J_tensor[k].astype(np.float64) @ W.T   # (2, 2)
        A = 0.5 * (J2d - J2d.T)
        out[k] = A[0, 1]
    return out


def main():
    print("=" * 80); print("Task 2 — cell-cycle reversal + vel_scale=2"); print("=" * 80)
    t_all = time.time()
    X, tau_fwd, phi, tangent_amb, z, W = make_data()
    tau_rev = 1.0 - tau_fwd
    print(f"substrate: X={X.shape}, W={W.shape}")

    # vel0 arms — try to load from earlier run's saved caches
    # (the earlier runner didn't save J_tensors; re-fit vel0 arms here for
    # A12 metric computation)
    per_arm = {"vel0_fwd": {}, "vel0_rev": {}, "vel2_fwd": {}, "vel2_rev": {}}

    # cache vel-arm fits so a re-run is fast
    for arm_name, arm_tau, vel in [
        ("vel0_fwd", tau_fwd, 0.0),
        ("vel0_rev", tau_rev, 0.0),
        ("vel2_fwd", tau_fwd, 2.0),
        ("vel2_rev", tau_rev, 2.0),
    ]:
        print(f"\n[{arm_name}] vel_scale={vel}, 3 seeds")
        for s in SEEDS:
            ck = OUT / f"task2_{arm_name}_seed{s}.pkl"
            if ck.exists():
                print(f"  seed {s}: cache hit")
                per_arm[arm_name][s] = pickle.loads(ck.read_bytes())
                continue
            t0 = time.time()
            v_per_cell, J_tensor, t_centers = fit_arm_vel(X, arm_tau, s, vel)
            per_arm[arm_name][s] = {
                "v_per_cell": v_per_cell,
                "J_tensor": J_tensor,
                "t_centers": t_centers,
            }
            ck.write_bytes(pickle.dumps(per_arm[arm_name][s]))
            print(f"  seed {s}: {time.time()-t0:.1f}s  "
                  f"|v| median = {float(np.median(np.linalg.norm(v_per_cell, axis=1))):.3f}")

    # Metric 1: signed tangential drift per arm
    print("\n[metrics] signed mean tangential drift")
    metric_M1 = {}
    for arm_name, arm_data in per_arm.items():
        per_seed_mean = []
        for s in SEEDS:
            v = arm_data[s]["v_per_cell"]
            ang = (v * tangent_amb).sum(axis=1)  # per cell
            per_seed_mean.append(float(ang.mean()))
        metric_M1[arm_name] = {
            "per_seed_mean": per_seed_mean,
            "mean": float(np.mean(per_seed_mean)),
            "spread": float(np.std(per_seed_mean)),
        }
        print(f"  {arm_name}: mean angular drift = {metric_M1[arm_name]['mean']:+.5f}  "
              f"± {metric_M1[arm_name]['spread']:.5f}")

    # Metric 2: A_12 (antisymmetric projected on cycle plane)
    print("\n[metrics] antisymmetric part on cycle plane")
    metric_M2 = {}
    for arm_name, arm_data in per_arm.items():
        per_seed_med = []; per_seed_mean = []
        for s in SEEDS:
            prof = A12_profile(arm_data[s]["J_tensor"], W)
            per_seed_med.append(float(np.median(prof)))
            per_seed_mean.append(float(prof.mean()))
        metric_M2[arm_name] = {
            "median_per_seed": per_seed_med,
            "mean_per_seed": per_seed_mean,
            "median_mean": float(np.mean(per_seed_med)),
            "mean_mean": float(np.mean(per_seed_mean)),
            "median_spread": float(np.std(per_seed_med)),
        }
        print(f"  {arm_name}: median A_12 over τ (mean over seeds) = "
              f"{metric_M2[arm_name]['median_mean']:+.5f}  "
              f"± {metric_M2[arm_name]['median_spread']:.5f}")

    # Prediction test
    print("\n[prediction] sign flip at vel_scale = 2, ≈0 at vel_scale = 0")
    for vel_name in ("vel0", "vel2"):
        fwd = metric_M1[f"{vel_name}_fwd"]["mean"]
        rev = metric_M1[f"{vel_name}_rev"]["mean"]
        A_fwd = metric_M2[f"{vel_name}_fwd"]["median_mean"]
        A_rev = metric_M2[f"{vel_name}_rev"]["median_mean"]
        sign_flip_drift = (fwd > 0 and rev < 0) or (fwd < 0 and rev > 0)
        sign_flip_A = (A_fwd > 0 and A_rev < 0) or (A_fwd < 0 and A_rev > 0)
        near_zero = (abs(fwd) < 1e-3 and abs(rev) < 1e-3)
        print(f"  {vel_name}: drift fwd={fwd:+.5f} rev={rev:+.5f}  "
              f"sign-flip? {sign_flip_drift}  near-zero? {near_zero}")
        print(f"  {vel_name}: A_12   fwd={A_fwd:+.5f} rev={A_rev:+.5f}  "
              f"sign-flip? {sign_flip_A}")

    # ── FIGURE ──────────────────────────────────────────────────────────
    print("\n[figure] 4-panel figure ...")
    # collect per-cell tangent-projected drifts, seed-averaged, for coloring
    def cell_ang_map(arm_data):
        stack = np.stack([(arm_data[s]["v_per_cell"] * tangent_amb).sum(axis=1)
                           for s in SEEDS], axis=0)
        return stack.mean(axis=0)
    ang_maps = {a: cell_ang_map(per_arm[a]) for a in per_arm}
    vmax = max(np.abs(v).max() for v in ang_maps.values())
    layout = [("vel0_fwd", "vel0 forward"), ("vel0_rev", "vel0 reverse"),
              ("vel2_fwd", "vel2 forward"), ("vel2_rev", "vel2 reverse")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    n_arrow = 60
    arrow_phi = np.linspace(0, 2 * np.pi, n_arrow, endpoint=False)
    arrow_xy = np.stack([np.cos(arrow_phi), np.sin(arrow_phi)], axis=1)
    tangent2d = np.stack([-np.sin(arrow_phi), np.cos(arrow_phi)], axis=1)
    idx_nn = np.argmin(np.abs(phi[None, :] - arrow_phi[:, None]), axis=1)
    for ax, (arm_name, title) in zip(axes.flat, layout):
        ang = ang_maps[arm_name]
        m1 = metric_M1[arm_name]; m2 = metric_M2[arm_name]
        sc_h = ax.scatter(z[:, 0], z[:, 1], c=ang, cmap="RdBu_r",
                          vmin=-vmax, vmax=vmax, s=6, alpha=0.75)
        ang_sample = ang[idx_nn]
        L = 0.12
        u = tangent2d[:, 0] * np.sign(ang_sample) * L
        v = tangent2d[:, 1] * np.sign(ang_sample) * L
        ax.quiver(arrow_xy[:, 0], arrow_xy[:, 1], u, v,
                   angles="xy", scale_units="xy", scale=1, width=0.005,
                   color="black")
        ax.set_aspect("equal"); ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
        ax.set_xlabel("cos φ"); ax.set_ylabel("sin φ")
        ax.set_title(
            f"{title}\n"
            f"mean drift = {m1['mean']:+.4f} ± {m1['spread']:.4f}\n"
            f"median A_12 = {m2['median_mean']:+.4f} ± {m2['median_spread']:.4f}"
        )
        plt.colorbar(sc_h, ax=ax, label="tangent-projected drift")
    fig.suptitle("Cell-cycle reversal calibration — vel_scale = 0 vs vel_scale = 2")
    fig.savefig(OUT / "fig_cellcycle_reversal.pdf", dpi=150)
    fig.savefig(OUT / "fig_cellcycle_reversal.png", dpi=150)
    plt.close(fig)
    print("[figure] wrote fig_cellcycle_reversal.pdf/.png")

    # Save summary
    def _tn(v):
        if isinstance(v, (np.floating,)): return float(v)
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, (list, tuple)): return [_tn(x) for x in v]
        if isinstance(v, dict): return {k: _tn(x) for k, x in v.items()}
        return v
    (OUT / "cellcycle_summary_task2.json").write_text(json.dumps(_tn({
        "M1_signed_mean_tangential_drift": metric_M1,
        "M2_A12_antisymmetric": metric_M2,
        "n_seeds": len(SEEDS), "n_cells": N_CELLS,
    }), indent=2))
    print(f"\n[written] cellcycle_summary_task2.json + fig  total {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
