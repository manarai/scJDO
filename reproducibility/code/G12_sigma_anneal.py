"""
G-12 — σ-schedule sweep at N = 100,000.

At each σ ∈ {0.10, 0.05, 0.02}, run:
  - normal training (real τ labels)
  - shuffled-τ training (same-N, same-σ confound)

Three seeds each. Uniform batching (matches manuscript pipeline).
No loss-term changes; σ is the only lever.

Prediction from G-10:
  σ = 0.10 : transverse structure width ≈ σ, borderline.
  σ = 0.05 : ratio ≈ 2×, should expose conditioning if it exists.
  σ = 0.02 : ratio ≈ 5×, strongest test.

If Δloss (shuffle − normal) exceeds the seed-to-seed floor at
smaller σ, the objective DOES carry τ-conditioning gradient but it
was swamped by DSM noise at σ = 0.10.
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

from _knn_speedup import enable_sklearn_knn_speedup
enable_sklearn_knn_speedup()

import toggle_truth as gt
from FOLLOWUP2_trained_probe import _build_adata
from G11_N100k_sweep import _bespoke_train, _time_probe, _signal_cos


def run_one(seed, N, sigma, shuffle_tau):
    adata, Z_lat, tau = _build_adata(seed=seed, n_cells=N)
    # Rebuild the trainer to accept σ as an argument
    from scjdo.models.drift import DriftField, DriftConfig
    from scjdo.losses import denoising_score_matching, control_energy
    from scjdo.tl._drift import _pseudotime_velocity
    torch.manual_seed(seed); np.random.seed(seed)
    device = "cpu"
    X_np = adata.obsm["X_pca"].astype(np.float32)
    T_np = adata.obs["pseudotime"].values.astype(np.float32)
    N_, D = X_np.shape
    if shuffle_tau:
        rng = np.random.default_rng(seed + 999)
        T_used = T_np[rng.permutation(N_)].astype(np.float32)
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
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=800)
    losses = []
    t0 = time.time()
    for step in range(800):
        idx = torch.randint(0, N_, (512,), device=device)
        xb, tb = X[idx], T[idx]
        loss = denoising_score_matching(model, xb, tb, sigma=float(sigma))
        loss = loss + cfg.alpha_control * control_energy(model(xb, tb))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        losses.append(float(loss.item()))
    dt = time.time() - t0
    final = float(np.asarray(losses[-50:]).mean())
    ratio, cos_lohi = _time_probe(model, Z_lat, seed=seed)
    sig = _signal_cos(model, Z_lat, tau, seed=seed)
    return {"seed": seed, "N": N, "sigma": float(sigma),
             "shuffle_tau": shuffle_tau, "train_time_s": dt,
             "final_dsm_loss": final,
             "mean_relative_dfdtau": ratio, "cos_lohi": cos_lohi, **sig}


def main():
    out_dir = REPO / "reproducibility"
    N = 100_000
    sigmas = [0.10, 0.05, 0.02]
    seeds = [42, 0, 1]
    results = []
    print(f"=== G-12 σ-anneal at N={N}, uniform batching ===", flush=True)
    for sigma in sigmas:
        print(f"\n== σ = {sigma} ==", flush=True)
        for shuffle in [False, True]:
            for s in seeds:
                r = run_one(s, N=N, sigma=sigma, shuffle_tau=shuffle)
                results.append(r)
                print(f"  σ={sigma}  shuffle={shuffle}  seed={s}  "
                      f"loss={r['final_dsm_loss']:.4f}  "
                      f"cos(0.1,0.9)={r['cos_lohi']:+.4f}  "
                      f"ratio={r['mean_relative_dfdtau']:.3f}  "
                      f"cos_sig(pre)={r['cos_signal_p50_pre_saddle']:+.3f}  "
                      f"time={r['train_time_s']:.1f}s", flush=True)
                (out_dir / "data" / "G12_sigma_anneal.json").write_text(
                    json.dumps(results, indent=2, default=lambda x: float(x)))

    print("\n=== SUMMARY: Δloss (shuffle − normal) per σ ===")
    for sigma in sigmas:
        n = [r for r in results if r["sigma"] == sigma and not r["shuffle_tau"]]
        s = [r for r in results if r["sigma"] == sigma and r["shuffle_tau"]]
        ln = np.array([r["final_dsm_loss"] for r in n])
        ls = np.array([r["final_dsm_loss"] for r in s])
        cos_sig = np.array([r["cos_signal_p50_pre_saddle"] for r in n])
        ratio = np.array([r["mean_relative_dfdtau"] for r in n])
        cos_lo = np.array([r["cos_lohi"] for r in n])
        pooled = np.hypot(ln.std(ddof=1), ls.std(ddof=1))
        delta = ls.mean() - ln.mean()
        print(f"  σ={sigma:.3f}   Δloss={delta:+.4f}  pooled_sd={pooled:.4f}  "
              f"Δ/sd={delta/max(pooled,1e-6):+.2f}  |  "
              f"cos_sig(pre) mean={cos_sig.mean():+.3f}  "
              f"ratio mean={ratio.mean():.3f}  cos(0.1,0.9) mean={cos_lo.mean():+.4f}")


if __name__ == "__main__":
    main()
