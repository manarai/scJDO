"""
G-11 — G-5/G-6/G-8 re-run at N = 100,000.  Same widths, same seeds.
Report cells-per-window at each width alongside every metric.

30 per-τ fits + 6 uniform baseline = 36 fits × ~2 min = ~72 min.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path("/Users/terooatt/Downloads/scJDO")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "reproducibility" / "code"))

# Enable sklearn KNN speedup for N ≥ 50k
from _knn_speedup import enable_sklearn_knn_speedup
enable_sklearn_knn_speedup()

import toggle_truth as gt
from FOLLOWUP2_trained_probe import _build_adata


def _bespoke_train(adata, seed, shuffle_tau=False, batch_mode="uniform",
                   window=0.05, n_epochs=800):
    from scjdo.models.drift import DriftField, DriftConfig
    from scjdo.losses import denoising_score_matching, control_energy
    from scjdo.tl._drift import _pseudotime_velocity
    torch.manual_seed(seed); np.random.seed(seed)
    device = "cpu"
    X_np = adata.obsm["X_pca"].astype(np.float32)
    T_np = adata.obs["pseudotime"].values.astype(np.float32)
    N, D = X_np.shape
    if shuffle_tau:
        rng = np.random.default_rng(seed + 999)
        T_used = T_np[rng.permutation(N)].astype(np.float32)
    else:
        T_used = T_np
    X = torch.tensor(X_np, device=device)
    T = torch.tensor(T_used, device=device)
    V_np = _pseudotime_velocity(X_np, T_used, k=15)
    V = torch.tensor(V_np, device=device)
    cfg = DriftConfig(dim=D, hidden=256, depth=4, beta=0.1,
                       use_spectral_norm=True, use_velocity_prior=True,
                       vel_scale=2.0, vel_k=15, vel_time_mode="flat")
    model = DriftField(cfg, X_ref=X, V_ref=V).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    losses = []
    labels_t = T
    cells_per_batch = []
    for step in range(n_epochs):
        if batch_mode == "uniform":
            idx = torch.randint(0, N, (512,), device=device)
            cells_per_batch.append(N)
        else:
            tau_c = float(torch.rand(1, device=device).item())
            in_win = torch.where(torch.abs(labels_t - tau_c) <= float(window))[0]
            cells_per_batch.append(int(in_win.numel()))
            if in_win.numel() >= 1:
                sel = torch.randint(0, in_win.numel(), (512,), device=device)
                idx = in_win[sel]
            else:
                _, nn = torch.topk(-torch.abs(labels_t - tau_c), 512)
                idx = nn
        xb, tb = X[idx], T[idx]
        loss = denoising_score_matching(model, xb, tb, sigma=0.1)
        loss = loss + cfg.alpha_control * control_energy(model(xb, tb))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        losses.append(float(loss.item()))
    return model, np.array(losses), np.asarray(cells_per_batch)


def _time_probe(model, Z_lat, seed, n_probe=500):
    device = next(model.parameters()).device
    rng = np.random.default_rng(seed + 1000)
    idx = rng.choice(Z_lat.shape[0], size=min(n_probe, Z_lat.shape[0]), replace=False)
    z_probe = torch.tensor(Z_lat[idx], dtype=torch.float32, device=device)
    tau_grid = np.linspace(0.02, 0.98, 25)
    with torch.no_grad():
        f_grid = np.zeros((len(tau_grid), z_probe.shape[0], z_probe.shape[1]))
        for k, tv in enumerate(tau_grid):
            tvec = torch.full((z_probe.shape[0],), float(tv), device=device)
            f_grid[k] = model(z_probe, tvec).detach().cpu().numpy()
    dτ = tau_grid[1] - tau_grid[0]
    df_dtau = np.gradient(f_grid, dτ, axis=0)
    norm_f = np.linalg.norm(f_grid, axis=-1) + 1e-12
    norm_df = np.linalg.norm(df_dtau, axis=-1)
    ratio = (norm_df / norm_f).mean()
    ilo = int(np.argmin(np.abs(tau_grid - 0.1)))
    ihi = int(np.argmin(np.abs(tau_grid - 0.9)))
    a, b = f_grid[ilo], f_grid[ihi]
    cos_lohi = (a * b).sum(-1) / (
        (np.linalg.norm(a, axis=-1) + 1e-12) * (np.linalg.norm(b, axis=-1) + 1e-12)
    )
    return float(ratio), float(cos_lohi.mean())


def _signal_cos(model, Z_lat, tau, seed):
    from sklearn.decomposition import PCA
    z2, tau2 = gt.simulate_v3(n_cells=len(tau), seed=seed)
    assert np.allclose(tau, tau2)
    W = gt.build_observation_map(seed=seed)
    rng_g = np.random.default_rng(seed)
    X_obs = z2 @ W + gt.NOISE_SIGMA * rng_g.standard_normal((z2.shape[0], W.shape[1])).astype(np.float32)
    pca = PCA(n_components=20, random_state=seed).fit(X_obs)
    A_lift = (W @ pca.components_.T).astype(np.float64)
    alphas = gt.alpha_of_tau(tau)
    f2 = gt.toggle_drift(z2.astype(np.float64), alphas)
    f_lifted = f2 @ A_lift
    Q, _ = np.linalg.qr(A_lift.T)
    device = next(model.parameters()).device
    # Chunked forward to save memory at N=100k
    z_all = torch.tensor(Z_lat, dtype=torch.float32, device=device)
    t_all = torch.tensor(tau, dtype=torch.float32, device=device)
    parts = []
    with torch.no_grad():
        for i in range(0, z_all.shape[0], 4096):
            parts.append(model(z_all[i:i+4096], t_all[i:i+4096]).cpu().numpy())
    f_train = np.concatenate(parts, axis=0).astype(np.float64)
    f_train_sig = f_train @ Q @ Q.T
    f_lifted_sig = f_lifted @ Q @ Q.T
    na = np.linalg.norm(f_train_sig, axis=1) + 1e-12
    nb = np.linalg.norm(f_lifted_sig, axis=1) + 1e-12
    cos_sig = (f_train_sig * f_lifted_sig).sum(1) / (na * nb)
    off_frac = float(np.median(
        np.linalg.norm(f_train - f_train_sig, axis=1) /
        (np.linalg.norm(f_train, axis=1) + 1e-12)
    ))
    mask_pre = tau < 0.05
    return {
        "cos_signal_p50_all": float(np.median(cos_sig)),
        "cos_signal_p50_pre_saddle": (
            float(np.median(cos_sig[mask_pre])) if mask_pre.any() else float("nan")
        ),
        "cos_signal_p10_pre_saddle": (
            float(np.percentile(cos_sig[mask_pre], 10)) if mask_pre.any() else float("nan")
        ),
        "cos_signal_p90_pre_saddle": (
            float(np.percentile(cos_sig[mask_pre], 90)) if mask_pre.any() else float("nan")
        ),
        "off_manifold_frac_p50": off_frac,
    }


def run_one(seed, batch_mode, shuffle_tau, N, window=0.05):
    adata, Z_lat, tau = _build_adata(seed=seed, n_cells=N)
    t0 = time.time()
    model, losses, cpb = _bespoke_train(adata, seed=seed, shuffle_tau=shuffle_tau,
                                         batch_mode=batch_mode, window=window,
                                         n_epochs=800)
    dt = time.time() - t0
    final = float(losses[-50:].mean())
    ratio, cos_lohi = _time_probe(model, Z_lat, seed=seed)
    sig = _signal_cos(model, Z_lat, tau, seed=seed)
    return {
        "seed": seed, "batch_mode": batch_mode, "shuffle_tau": shuffle_tau,
        "window": window, "N": N,
        "cells_per_window_mean": float(cpb.mean()),
        "cells_per_window_p10": float(np.percentile(cpb, 10)),
        "cells_per_window_p90": float(np.percentile(cpb, 90)),
        "train_time_s": dt, "final_dsm_loss": final,
        "mean_relative_dfdtau": ratio, "cos_lohi": cos_lohi, **sig,
    }


def main():
    out_dir = REPO / "reproducibility"
    N = 100_000
    widths = [0.01, 0.02, 0.05, 0.10, 0.20]
    seeds = [42, 0, 1]
    results = []

    print(f"=== G-11 at N={N} ===", flush=True)
    print("== BASELINE: uniform batching ==", flush=True)
    for shuffle in [False, True]:
        for s in seeds:
            r = run_one(s, batch_mode="uniform", shuffle_tau=shuffle, N=N)
            results.append(r)
            print(f"  uniform  shuffle={shuffle}  seed={s}  loss={r['final_dsm_loss']:.4f}  "
                  f"cos(0.1,0.9)={r['cos_lohi']:+.4f}  ratio={r['mean_relative_dfdtau']:.3f}  "
                  f"cos_sig(pre)={r['cos_signal_p50_pre_saddle']:+.3f}  "
                  f"time={r['train_time_s']:.1f}s", flush=True)
            (out_dir / "data" / "G11_N100k_sweep.json").write_text(
                json.dumps(results, indent=2, default=lambda x: float(x)))

    for w in widths:
        print(f"\n== PER-τ  window={w} ==", flush=True)
        for shuffle in [False, True]:
            for s in seeds:
                r = run_one(s, batch_mode="per_tau", shuffle_tau=shuffle, N=N, window=w)
                results.append(r)
                print(f"  per_tau w={w:.2f}  shuffle={shuffle}  seed={s}  "
                      f"loss={r['final_dsm_loss']:.4f}  "
                      f"cells/win mean={r['cells_per_window_mean']:.0f} "
                      f"[p10={r['cells_per_window_p10']:.0f}, p90={r['cells_per_window_p90']:.0f}]  "
                      f"cos(0.1,0.9)={r['cos_lohi']:+.4f}  ratio={r['mean_relative_dfdtau']:.3f}  "
                      f"cos_sig(pre)={r['cos_signal_p50_pre_saddle']:+.3f}  "
                      f"time={r['train_time_s']:.1f}s", flush=True)
                (out_dir / "data" / "G11_N100k_sweep.json").write_text(
                    json.dumps(results, indent=2, default=lambda x: float(x)))

    print("\n=== SUMMARY: Δloss (shuffle − normal) per width, N=100k ===")
    for mode, w in [("uniform", None)] + [("per_tau", w) for w in widths]:
        norm_rows = [r for r in results if r["batch_mode"] == mode
                     and not r["shuffle_tau"] and (mode != "per_tau" or r["window"] == w)]
        shuf_rows = [r for r in results if r["batch_mode"] == mode
                     and r["shuffle_tau"] and (mode != "per_tau" or r["window"] == w)]
        ln = np.array([r["final_dsm_loss"] for r in norm_rows])
        ls = np.array([r["final_dsm_loss"] for r in shuf_rows])
        cos_sig = np.array([r["cos_signal_p50_pre_saddle"] for r in norm_rows])
        ratio = np.array([r["mean_relative_dfdtau"] for r in norm_rows])
        cos_lo = np.array([r["cos_lohi"] for r in norm_rows])
        label = mode if mode == "uniform" else f"per_tau w={w:.2f}"
        cpw = np.mean([r["cells_per_window_mean"] for r in norm_rows]) if mode == "per_tau" else N
        pooled = np.hypot(ln.std(ddof=1), ls.std(ddof=1))
        delta = ls.mean() - ln.mean()
        print(f"  {label:20s}  cells/win={cpw:6.0f}  Δloss={delta:+.4f}  "
              f"pooled_sd={pooled:.4f}  Δ/sd={delta/max(pooled,1e-6):+.2f}  "
              f"cos_sig(pre) mean={cos_sig.mean():+.3f}  "
              f"ratio mean={ratio.mean():.3f}  cos(0.1,0.9) mean={cos_lo.mean():+.4f}")


if __name__ == "__main__":
    main()
