"""
FOLLOWUP6 (#4) — Paul15 retrained on masked single branch.

The multiome fits are per-branch (restricted via `branch_masks`). Paul15
fit in `examples/results/01_paul15/paul15_scjdo.h5ad` is all-cells.

Does restricting Paul15 to a single-branch subset (e.g. Ery lineage +
early progenitors) break the always_positive pattern?

Fit two variants:
  - "all"     — no restriction (matches shipped fit).
  - "ery_lineage" — only cells with clusters in {early progenitor + Ery}.

Then compare aggregated λ curves and the always_positive verdict under
the fixed detector.
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

import anndata as ad
from crossing_robust import robust_crossing


def _fit_and_score(adata, tag, n_epochs=1500):
    from scjdo.tl import fit_drift
    t0 = time.time()
    model = fit_drift(adata, rep="X_pca", time_key="dpt_pseudotime",
                       n_epochs=n_epochs, n_archetypes=5, n_eff_min=20.0,
                       n_boot=10, grid_size=200, seed=42, verbose=False)
    dt = time.time() - t0
    r2 = float(adata.uns["scjdo"]["r2"])
    lam = np.asarray(adata.uns["scjdo"]["max_real_eig"])
    grid = np.asarray(adata.uns["scjdo"]["t_centers"])
    h = adata.uns["scjdo"].get("bandwidth")
    t, code = robust_crossing(lam, grid, threshold=0.0, min_run=5)
    print(f"  [{tag}]  n_cells={adata.n_obs:>4d}  fit={dt:.1f}s  R²={r2:.4f}  "
          f"h_auto={h}  min λ={float(np.nanmin(lam)):+.4f}  "
          f"max λ={float(np.nanmax(lam)):+.4f}  code={code}  "
          f"frac_neg={float((lam < 0).mean()):.3f}")
    return {"tag": tag, "n_cells": int(adata.n_obs), "fit_time": dt, "r2": r2,
             "h_auto": h, "min_lam": float(np.nanmin(lam)),
             "max_lam": float(np.nanmax(lam)), "code": code,
             "frac_neg": float((lam < 0).mean()),
             "lam": lam, "grid": grid}


def main():
    out_dir = REPO / "reproducibility"
    path = "examples/results/01_paul15/paul15_scjdo.h5ad"
    a0 = ad.read_h5ad(str(REPO / path))
    print(f"Loaded Paul15: n_cells={a0.n_obs}, X_pca shape={a0.obsm['X_pca'].shape}")

    # 'all' variant — just re-fit on the whole thing to verify.
    print("\n== 'all' variant ==")
    a_all = a0.copy()
    r_all = _fit_and_score(a_all, tag="all")

    # 'ery lineage' variant — Ery clusters + early progenitors.
    # Paul15 clusters: 7MEP is the branch point; 9GMP/10GMP are common progenitors
    # for the Mo/Neu side. For an "Ery" lineage: 7MEP + Ery clusters.
    ery_clusters = {"7MEP", "1Ery", "2Ery", "3Ery", "4Ery", "5Ery", "6Ery"}
    m = a0.obs["paul15_clusters"].isin(ery_clusters)
    a_ery = a0[m].copy()
    print("\n== 'ery' variant (7MEP + all Ery clusters) ==")
    r_ery = _fit_and_score(a_ery, tag="ery")

    # 'mo lineage' variant — Mo clusters + early progenitors
    mo_clusters = {"9GMP", "10GMP", "14Mo", "15Mo"}
    m = a0.obs["paul15_clusters"].isin(mo_clusters)
    a_mo = a0[m].copy()
    print("\n== 'mo' variant ==")
    r_mo = _fit_and_score(a_mo, tag="mo")

    out = {
        "all_variant": {k: v for k, v in r_all.items() if k not in ("lam", "grid")},
        "ery_variant": {k: v for k, v in r_ery.items() if k not in ("lam", "grid")},
        "mo_variant":  {k: v for k, v in r_mo.items() if k not in ("lam", "grid")},
    }
    (out_dir / "data" / "FOLLOWUP6_paul15_branchmask.json").write_text(
        json.dumps(out, indent=2, default=lambda x: float(x))
    )
    # curves
    np.savez_compressed(
        out_dir / "data" / "FOLLOWUP6_paul15_branchmask.npz",
        lam_all=r_all["lam"], grid_all=r_all["grid"],
        lam_ery=r_ery["lam"], grid_ery=r_ery["grid"],
        lam_mo=r_mo["lam"], grid_mo=r_mo["grid"],
    )
    print(f"\nSaved: {out_dir}/data/FOLLOWUP6_paul15_branchmask.json")


if __name__ == "__main__":
    main()
