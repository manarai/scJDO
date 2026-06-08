"""
Regression test for the kernel/fixed windowing branch in fit_drift.

Runs both windowing modes on Paul15 with a reduced epoch budget and verifies:
  * fixed mode still produces J_tensor of shape (n_windows, D, D) and a
    sensible lambda_max curve;
  * kernel mode produces J_tensor of shape (grid_size, D, D), a selected
    bandwidth, and stores the diagnostic keys downstream code depends on.

Not a pytest-style assertion-heavy test - this is intended as a smoke /
regression script you can rerun after editing the windowing pipeline.
"""
from __future__ import annotations

import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import scanpy as sc

import scjdo as sjd


def _prep():
    adata = sc.datasets.paul15()
    sjd.pp.prepare_trajectory(adata, groupby="paul15_clusters", root="7MEP")
    return adata


def run_fixed(epochs: int = 800):
    adata = _prep()
    t0 = time.time()
    sjd.tl.fit_drift(
        adata,
        n_epochs=epochs,
        windowing="fixed",
        n_windows=100,
        overlap=0.80,
        smooth_sigma=1.5,
        verbose=False,
    )
    dt = time.time() - t0
    r = adata.uns["scjdo"]
    print(f"\n[fixed]   {dt:.1f}s  J_tensor={r['J_tensor'].shape}  "
          f"peak λ={r['max_real_eig'].max():.3f} @ τ={r['t_centers'][np.argmax(r['max_real_eig'])]:.3f}")
    print(f"          windowing={r['windowing']!r}  T_eval={r['params']['T_eval']}  "
          f"bandwidth={r['bandwidth']!r}")
    return r


def run_kernel(epochs: int = 800):
    adata = _prep()
    t0 = time.time()
    sjd.tl.fit_drift(
        adata,
        n_epochs=epochs,
        windowing="kernel",
        bandwidth="auto",
        bandwidth_grid=(0.02, 0.03, 0.05, 0.08),
        grid_size=200,
        n_boot=10,
        verbose=False,
    )
    dt = time.time() - t0
    r = adata.uns["scjdo"]
    print(f"\n[kernel]  {dt:.1f}s  J_tensor={r['J_tensor'].shape}  "
          f"peak λ={r['max_real_eig'].max():.3f} @ τ={r['t_centers'][np.argmax(r['max_real_eig'])]:.3f}")
    print(f"          windowing={r['windowing']!r}  T_eval={r['params']['T_eval']}  "
          f"bandwidth={r['bandwidth']!r}  n_eff_min={r['n_eff'].min():.1f}")
    print(f"          score: {r['kernel_score']}")
    if r["kernel_sweep"]:
        for row in r["kernel_sweep"]:
            print(f"            h={row['h']:.3f}  R={row['R']:.3f}  C={row['C']:.3f}  "
                  f"L={row['L']:.3f}  S={row['S']:.4f}  n_eff_min={row['n_eff_interior_min']:.1f}")
    return r


def run_kernel_manual(epochs: int = 800):
    adata = _prep()
    t0 = time.time()
    sjd.tl.fit_drift(
        adata,
        n_epochs=epochs,
        windowing="kernel",
        bandwidth=0.05,
        grid_size=200,
        verbose=False,
    )
    dt = time.time() - t0
    r = adata.uns["scjdo"]
    print(f"\n[manual]  {dt:.1f}s  J_tensor={r['J_tensor'].shape}  "
          f"peak λ={r['max_real_eig'].max():.3f}  bandwidth={r['bandwidth']!r}")
    return r


if __name__ == "__main__":
    EPOCHS = 800
    print(f"Paul15 regression test (epochs={EPOCHS})")
    r_fixed   = run_fixed(EPOCHS)
    r_kernel  = run_kernel(EPOCHS)
    r_manual  = run_kernel_manual(EPOCHS)
    print("\nALL THREE MODES COMPLETED.")
