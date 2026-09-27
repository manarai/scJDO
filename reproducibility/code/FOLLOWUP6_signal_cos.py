"""
FOLLOWUP6 (#2) — Signal-subspace cos, per-cell distribution.

The prior cos ≈ 0 test computed cosine in the full 20-D latent. But the
lifted analytic drift lives entirely in the rank-2 signal subspace
col(A^T) = col(W V^T). A field with a large off-manifold component
can score cos ≈ 0 in 20-D even when its projection onto col(A^T) is
correct.

Test:
  1. Project f_train and f_lifted onto col(A^T) via Q Q^T where Q is
     an orthonormal basis of the 2-D subspace.
  2. Compute per-cell cos(P·f_train, P·f_lifted) — full distribution,
     not just the mean.
  3. Also compute the OFF-manifold energy fraction:
        ‖(I − P)·f_train‖ / ‖f_train‖
     If off-manifold energy dominates, "field is not tracking the
     manifold" is the right description (worse case). If off-manifold
     energy is small and in-manifold cos is high, it is "right object,
     partially rotated" — a milder claim.

Runs on the same fits as FOLLOWUP5_F_tests.
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


def signal_probe(seed=42, n_cells=1500):
    from sklearn.decomposition import PCA
    from scjdo.tl import fit_drift

    z2, tau = gt.simulate_v3(n_cells=n_cells, seed=seed)
    W = gt.build_observation_map(seed=seed)
    rng_g = np.random.default_rng(seed)
    X_obs = z2 @ W + gt.NOISE_SIGMA * rng_g.standard_normal((z2.shape[0], W.shape[1])).astype(np.float32)
    pca = PCA(n_components=20, random_state=seed).fit(X_obs)
    Z_lat = pca.transform(X_obs).astype(np.float32)

    import anndata as ad
    adata = ad.AnnData(X=X_obs); adata.obs_names = [f"c{i:05d}" for i in range(adata.n_obs)]
    adata.obsm["X_pca"] = Z_lat; adata.obs["pseudotime"] = tau

    t0 = time.time()
    model = fit_drift(adata, rep="X_pca", time_key="pseudotime",
                       n_epochs=800, n_archetypes=4, n_eff_min=20.0, n_boot=10,
                       grid_size=200, seed=seed, verbose=False)
    print(f"    fit_time={time.time() - t0:.1f}s   R²={float(adata.uns['scjdo']['r2']):.4f}")

    device = next(model.parameters()).device
    z_t = torch.tensor(Z_lat, dtype=torch.float32, device=device)
    t_t = torch.tensor(tau, dtype=torch.float32, device=device)
    with torch.no_grad():
        f_train = model(z_t, t_t).detach().cpu().numpy().astype(np.float64)

    # Lifted analytic drift
    alphas = gt.alpha_of_tau(tau)
    f2 = gt.toggle_drift(z2.astype(np.float64), alphas)
    A_lift = (W @ pca.components_.T).astype(np.float64)   # (2, 20)
    f_lifted = f2 @ A_lift

    # Signal-subspace basis
    Q, _ = np.linalg.qr(A_lift.T)   # (20, 2), orthonormal columns

    # Project f_train and f_lifted onto Q
    f_train_sig = f_train @ Q @ Q.T   # (N, 20) but rank-2
    f_lifted_sig = f_lifted @ Q @ Q.T
    f_train_off = f_train - f_train_sig
    f_lifted_off = f_lifted - f_lifted_sig  # should be ~0 in truth

    def _cos_rowwise(a, b):
        na = np.linalg.norm(a, axis=1) + 1e-12
        nb = np.linalg.norm(b, axis=1) + 1e-12
        return (a * b).sum(1) / (na * nb)

    cos_full = _cos_rowwise(f_train, f_lifted)
    cos_signal = _cos_rowwise(f_train_sig, f_lifted_sig)

    off_train = np.linalg.norm(f_train_off, axis=1) / (np.linalg.norm(f_train, axis=1) + 1e-12)
    off_lifted = np.linalg.norm(f_lifted_off, axis=1) / (np.linalg.norm(f_lifted, axis=1) + 1e-12)

    return {"seed": seed, "tau": tau,
             "cos_full": cos_full, "cos_signal": cos_signal,
             "off_manifold_train": off_train,
             "off_manifold_lifted": off_lifted}


def _percentile_summary(x, prefix, tau=None, tau_bins=None):
    """Print p10/p50/p90 across all cells and (optionally) per τ bin."""
    p10, p50, p90 = np.percentile(x, [10, 50, 90])
    out = {f"{prefix}_p10": float(p10), f"{prefix}_p50": float(p50), f"{prefix}_p90": float(p90),
           f"{prefix}_mean": float(x.mean())}
    if tau is not None and tau_bins is not None:
        for lo, hi, label in tau_bins:
            m = (tau >= lo) & (tau < hi)
            if m.any():
                p10b, p50b, p90b = np.percentile(x[m], [10, 50, 90])
                out[f"{prefix}_{label}_p10"] = float(p10b)
                out[f"{prefix}_{label}_p50"] = float(p50b)
                out[f"{prefix}_{label}_p90"] = float(p90b)
    return out


def main():
    out_dir = REPO / "reproducibility"
    tau_bins = [
        (0.0, 0.05, "pre_saddle"),
        (0.05, 0.20, "early_post"),
        (0.20, 0.80, "committed"),
        (0.80, 1.0, "far_committed"),
    ]

    all_results = []
    for seed in [42, 0, 1]:
        print(f"\n== signal-cos probe, seed={seed} ==")
        r = signal_probe(seed=seed)
        row = {"seed": seed}
        row.update(_percentile_summary(r["cos_full"], "cos_full_20D", r["tau"], tau_bins))
        row.update(_percentile_summary(r["cos_signal"], "cos_signal_2D", r["tau"], tau_bins))
        row.update(_percentile_summary(r["off_manifold_train"], "offman_train", r["tau"], tau_bins))
        row.update(_percentile_summary(r["off_manifold_lifted"], "offman_lifted", r["tau"], tau_bins))
        all_results.append(row)
        print(f"   cos_full 20-D    p50={row['cos_full_20D_p50']:+.3f}  "
              f"in pre_saddle p50={row.get('cos_full_20D_pre_saddle_p50', float('nan')):+.3f}")
        print(f"   cos_signal 2-D   p50={row['cos_signal_2D_p50']:+.3f}  "
              f"in pre_saddle p50={row.get('cos_signal_2D_pre_saddle_p50', float('nan')):+.3f}  "
              f"early_post p50={row.get('cos_signal_2D_early_post_p50', float('nan')):+.3f}")
        print(f"   off-manifold ‖(I-P)f_train‖/‖f_train‖: p50={row['offman_train_p50']:.3f}  "
              f"p90={row['offman_train_p90']:.3f}")
        print(f"   off-manifold in lifted (should be ~0): p50={row['offman_lifted_p50']:.4f}")

    (out_dir / "data" / "FOLLOWUP6_signal_cos.json").write_text(
        json.dumps(all_results, indent=2, default=lambda x: float(x))
    )
    print(f"\nSaved: {out_dir}/data/FOLLOWUP6_signal_cos.json")


if __name__ == "__main__":
    main()
