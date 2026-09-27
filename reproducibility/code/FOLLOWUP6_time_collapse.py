"""
FOLLOWUP6 (#3) — Time-conditioning collapse test.

Two probes:
  (a) At a fixed z*, sweep τ ∈ [0, 1] and measure  ‖∂f/∂τ‖ / ‖f‖  — how
      fast does the drift change with τ, relative to its own magnitude?
      Values ≪ 1 mean the drift is nearly τ-invariant → time conditioning
      has collapsed.
  (b) cos(f(z*, τ=0.1), f(z*, τ=0.9)) — same z, two very different τ.
      Values → +1 mean the field is τ-invariant.

Runs for both default (vel_scale=2.0) and ablated (vel_scale=0.0) fits.
Averages the probes over a sample of cells drawn from the training set.

Prediction under F's collapsed-time interpretation:
  - default:  ∂f/∂τ small; cos(τ=0.1, τ=0.9) close to 1 in the direction
    of ∇pseudotime, less close in other directions.
  - ablated:  cos(τ=0.1, τ=0.9) close to +1 across all directions →
    time collapse.
  - If neither exhibits collapse, then the τ-dependence exists but points
    in the "wrong" direction — that is a separate finding from collapse.
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


def collapse_probe(seed, vel_scale=None, n_probe_cells=200):
    from scjdo.tl import fit_drift
    adata, Z_lat, tau = _build_adata(seed=seed)
    kwargs = dict(rep="X_pca", time_key="pseudotime", n_epochs=800,
                  n_archetypes=4, n_eff_min=20.0, n_boot=10, grid_size=200,
                  seed=seed, verbose=False)
    if vel_scale is not None:
        kwargs["vel_scale"] = float(vel_scale)
    model = fit_drift(adata, **kwargs)

    device = next(model.parameters()).device
    rng = np.random.default_rng(seed + 1000)
    idx = rng.choice(Z_lat.shape[0], size=min(n_probe_cells, Z_lat.shape[0]),
                     replace=False)
    z_probe = torch.tensor(Z_lat[idx], dtype=torch.float32, device=device)

    tau_grid = np.linspace(0.02, 0.98, 25)
    with torch.no_grad():
        f_grid = np.zeros((len(tau_grid), z_probe.shape[0], z_probe.shape[1]))
        for k, tv in enumerate(tau_grid):
            tvec = torch.full((z_probe.shape[0],), float(tv), device=device)
            f_grid[k] = model(z_probe, tvec).detach().cpu().numpy()

    # (a) Relative time-derivative:  ‖∂f/∂τ‖ / ‖f‖ evaluated by central
    # difference over the τ-grid.
    dτ = tau_grid[1] - tau_grid[0]
    df_dtau = np.gradient(f_grid, dτ, axis=0)              # (T, N, D)
    norm_f = np.linalg.norm(f_grid, axis=-1) + 1e-12       # (T, N)
    norm_df = np.linalg.norm(df_dtau, axis=-1)              # (T, N)
    ratio = (norm_df / norm_f).mean()                       # scalar

    # (b) cos(f(z, τ=0.1), f(z, τ=0.9)) per cell.
    idx_lo = int(np.argmin(np.abs(tau_grid - 0.1)))
    idx_hi = int(np.argmin(np.abs(tau_grid - 0.9)))
    a = f_grid[idx_lo]; b = f_grid[idx_hi]
    cos_lohi = (a * b).sum(-1) / (
        (np.linalg.norm(a, axis=-1) + 1e-12) * (np.linalg.norm(b, axis=-1) + 1e-12)
    )

    # Broader τ-sweep pairwise cosine
    # cos(f(τ_i), f(τ_j)) averaged over cells then averaged over |i-j|
    per_lag = {}
    for lag in [1, 2, 5, 10, 20]:
        pairs = []
        for i in range(len(tau_grid) - lag):
            j = i + lag
            aa = f_grid[i]; bb = f_grid[j]
            c = (aa * bb).sum(-1) / (
                (np.linalg.norm(aa, axis=-1) + 1e-12) * (np.linalg.norm(bb, axis=-1) + 1e-12)
            )
            pairs.append(c.mean())
        per_lag[f"lag_{lag}"] = float(np.mean(pairs))

    return {
        "seed": seed, "vel_scale": vel_scale,
        "mean_relative_dfdtau": float(ratio),
        "cos_tau0.1_tau0.9_mean": float(cos_lohi.mean()),
        "cos_tau0.1_tau0.9_p10": float(np.percentile(cos_lohi, 10)),
        "cos_tau0.1_tau0.9_p90": float(np.percentile(cos_lohi, 90)),
        "cos_by_lag": per_lag,
    }


def main():
    out_dir = REPO / "reproducibility"
    all_out = []
    for vel_scale in [None, 0.0]:
        label = "default (vel=2.0)" if vel_scale is None else "ABLATED (vel=0)"
        print(f"\n== {label} ==")
        for seed in [42, 0, 1]:
            r = collapse_probe(seed=seed, vel_scale=vel_scale)
            all_out.append({"label": label, **r})
            print(f"  seed={seed}  mean(‖∂f/∂τ‖/‖f‖) = {r['mean_relative_dfdtau']:.3f}  "
                  f"cos(τ=0.1, τ=0.9) mean={r['cos_tau0.1_tau0.9_mean']:+.3f}  "
                  f"[p10={r['cos_tau0.1_tau0.9_p10']:+.3f}, p90={r['cos_tau0.1_tau0.9_p90']:+.3f}]")
            print(f"       cos by lag: {r['cos_by_lag']}")

    (out_dir / "data" / "FOLLOWUP6_time_collapse.json").write_text(
        json.dumps(all_out, indent=2, default=lambda x: float(x))
    )
    print(f"\nSaved: {out_dir}/data/FOLLOWUP6_time_collapse.json")


if __name__ == "__main__":
    main()
