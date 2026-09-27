"""
FOLLOWUP7 — the decisive G test: τ-shuffle loss ablation.

Two training runs, same seed and data, differing only in whether
pseudotime is aligned with cell identity or shuffled across cells.

  1. **normal**  — cells paired with their true τ.
  2. **shuffle** — τ vector shuffled once at the start (permutation of
                    training labels; each cell keeps its z but is given
                    someone else's τ).

If final DSM loss under `shuffle` is statistically indistinguishable
from `normal`, the DSM objective carries no gradient pushing the
network to use τ. No architectural fix will help; a τ-distinguishability
loss term is needed.

Also runs the same post-training probes as FOLLOWUP6_time_collapse:
  - ‖∂f/∂τ‖/‖f‖ and cos(f(z, τ=0.1), f(z, τ=0.9))
  - so we can compare τ-usage before vs after shuffle.
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

import toggle_truth as gt
from FOLLOWUP2_trained_probe import _build_adata


def _bespoke_train(adata, seed, shuffle_tau=False, n_epochs=800):
    """Local implementation of fit_drift's training loop so we can control
    exactly what gets passed to the model each step and record the loss."""
    from scjdo.models.drift import DriftField, DriftConfig
    from scjdo.losses import denoising_score_matching, control_energy
    from scjdo.tl._drift import _pseudotime_velocity
    from tqdm.auto import trange

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cpu"
    X_np = adata.obsm["X_pca"].astype(np.float32)
    T_np = adata.obs["pseudotime"].values.astype(np.float32)
    N, D = X_np.shape

    if shuffle_tau:
        rng = np.random.default_rng(seed + 999)
        T_np = T_np[rng.permutation(N)].astype(np.float32)

    X = torch.tensor(X_np, device=device)
    T = torch.tensor(T_np, device=device)

    V_np = _pseudotime_velocity(X_np, T_np, k=15)
    V = torch.tensor(V_np, device=device)

    cfg = DriftConfig(dim=D, hidden=256, depth=4, beta=0.1,
                       use_spectral_norm=True, use_velocity_prior=True,
                       vel_scale=2.0, vel_k=15, vel_time_mode="flat")
    model = DriftField(cfg, X_ref=X, V_ref=V).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    losses = []
    for step in range(n_epochs):
        idx = torch.randint(0, N, (512,), device=device)
        xb, tb = X[idx], T[idx]
        loss = denoising_score_matching(model, xb, tb, sigma=0.1)
        loss = loss + cfg.alpha_control * control_energy(model(xb, tb))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(float(loss.item()))
    return model, np.array(losses), T_np, X_np


def _time_probe(model, Z_lat, seed=42, n_cells_probe=200):
    device = next(model.parameters()).device
    rng = np.random.default_rng(seed + 1000)
    idx = rng.choice(Z_lat.shape[0], size=min(n_cells_probe, Z_lat.shape[0]),
                     replace=False)
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
    a = f_grid[ilo]; b = f_grid[ihi]
    cos_lohi = (a * b).sum(-1) / (
        (np.linalg.norm(a, axis=-1) + 1e-12) * (np.linalg.norm(b, axis=-1) + 1e-12)
    )
    return {"mean_relative_dfdtau": float(ratio),
             "cos_lohi_mean": float(cos_lohi.mean()),
             "cos_lohi_p10": float(np.percentile(cos_lohi, 10)),
             "cos_lohi_p90": float(np.percentile(cos_lohi, 90))}


def _film_stats(model):
    """Log γ, β statistics at various τ.  If γ ≈ 0 and β ≈ 0 (recall FiLM
    is `h * (1 + γ) + β`, so γ=0 means multiplier is 1), the modulation
    path is dead."""
    stats = {}
    device = next(model.parameters()).device
    for tv in [0.05, 0.5, 0.95]:
        with torch.no_grad():
            emb = model.score.emb(torch.tensor([float(tv)], device=device))
            per_layer = []
            for film in model.score.films:
                g = film.gamma(emb).cpu().numpy().flatten()
                b = film.beta(emb).cpu().numpy().flatten()
                per_layer.append(dict(gamma_mean=float(g.mean()),
                                       gamma_std=float(g.std()),
                                       gamma_absmean=float(np.abs(g).mean()),
                                       beta_mean=float(b.mean()),
                                       beta_std=float(b.std()),
                                       beta_absmean=float(np.abs(b).mean())))
            stats[f"tau={tv:.2f}"] = per_layer
    # Also compute the difference gamma(τ=0.05) vs gamma(τ=0.95): if the
    # FiLM depends on τ, these should differ.
    with torch.no_grad():
        e05 = model.score.emb(torch.tensor([0.05], device=device))
        e95 = model.score.emb(torch.tensor([0.95], device=device))
        deltas = []
        for film in model.score.films:
            g_delta = (film.gamma(e05) - film.gamma(e95)).cpu().numpy().flatten()
            b_delta = (film.beta(e05) - film.beta(e95)).cpu().numpy().flatten()
            deltas.append(dict(delta_gamma_absmean=float(np.abs(g_delta).mean()),
                                delta_beta_absmean=float(np.abs(b_delta).mean())))
        stats["delta_tau_0.05_vs_0.95"] = deltas
    return stats


def main():
    out_dir = REPO / "reproducibility"
    results = {}
    for shuffle in [False, True]:
        label = "shuffle_tau" if shuffle else "normal"
        print(f"\n== {label} training ==")
        for seed in [42, 0, 1]:
            adata, Z_lat, tau = _build_adata(seed=seed)
            t0 = time.time()
            model, losses, T_used, X_used = _bespoke_train(adata, seed=seed,
                                                             shuffle_tau=shuffle,
                                                             n_epochs=800)
            dt = time.time() - t0
            # Compare final DSM loss (last 50 iters)
            final_loss = float(losses[-50:].mean())
            # Time-collapse probe on the actual latent (using original tau ordering
            # — we're probing the trained field's τ dependence, not the labels).
            probe = _time_probe(model, Z_lat, seed=seed)
            # FiLM stats
            film = _film_stats(model)
            results.setdefault(label, []).append({
                "seed": seed, "train_time_s": dt,
                "final_dsm_loss": final_loss,
                "loss_curve_last_10": [float(l) for l in losses[-10:]],
                "time_probe": probe,
                "film_stats": film,
            })
            print(f"  seed={seed}  train={dt:.1f}s  final_loss={final_loss:.4f}  "
                  f"cos(0.1,0.9)={probe['cos_lohi_mean']:+.4f}  "
                  f"‖∂f/∂τ‖/‖f‖={probe['mean_relative_dfdtau']:.3f}")

    # Compare normal vs shuffle final-loss distributions
    lns = np.array([r["final_dsm_loss"] for r in results["normal"]])
    lss = np.array([r["final_dsm_loss"] for r in results["shuffle_tau"]])
    print(f"\n== Comparison ==")
    print(f"  normal   final DSM: mean={lns.mean():.4f}  std={lns.std():.4f}  values={list(lns)}")
    print(f"  shuffle  final DSM: mean={lss.mean():.4f}  std={lss.std():.4f}  values={list(lss)}")
    print(f"  Δ (shuffle − normal): {lss.mean() - lns.mean():+.4f} (pooled std ≈ {np.hypot(lns.std(), lss.std()):.4f})")

    (out_dir / "data" / "FOLLOWUP7_tau_shuffle.json").write_text(
        json.dumps(results, indent=2, default=lambda x: float(x))
    )
    print(f"\nSaved: {out_dir}/data/FOLLOWUP7_tau_shuffle.json")


if __name__ == "__main__":
    main()
