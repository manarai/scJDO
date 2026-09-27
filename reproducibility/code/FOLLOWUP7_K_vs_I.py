"""
FOLLOWUP7 — Koopman ‖K − I‖ and spectrum diagnostic.

If the Jacobian is essentially τ-invariant (as G confirmed), the reduced
Koopman operator K fit from a near-constant vectorised-J sequence should
be close to identity.  Descriptors of the identity are degenerate:

    Henrici departure from normality  H(I) = 0
    Reactivity  = λmax((I + I^T)/2) = 1
    Transient gain over N steps  = ‖I^N‖/ρ(I)^N = 1
    κ(eigvecs)  = κ(I) = 1

If the manuscript's Koopman descriptors are all measuring numerical
noise about K ≈ I, then the "four descriptors converge" story doesn't
survive.

Measured per window (window_half ∈ {4, 8, 12, 16}):
   ‖K − I‖_F         (Frobenius distance to identity)
   ρ(K)              (spectral radius)
   sorted |eig(K)|   (largest and smallest)
   κ(V)              (condition number of eigenvector matrix)

Prediction: ‖K − I‖_F small (dominated by numerical / ridge terms),
and spectrum tightly clustered around 1.
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

from FOLLOWUP2_trained_probe import _build_adata


def main():
    from scjdo.tl import fit_drift
    from scjdo.archetypes.koopman import _local_koopman

    out_dir = REPO / "reproducibility"
    results = {}
    for seed in [42, 0, 1]:
        adata, Z, tau = _build_adata(seed=seed)
        t0 = time.time()
        model = fit_drift(adata, rep="X_pca", time_key="pseudotime",
                           n_epochs=800, n_archetypes=4, n_eff_min=20.0, n_boot=10,
                           grid_size=200, seed=seed, verbose=False)
        print(f"\nseed={seed}  fit={time.time()-t0:.1f}s  R²={float(adata.uns['scjdo']['r2']):.4f}")

        J_tensor = np.asarray(adata.uns["scjdo"]["J_tensor"])   # (T=200, D=20, D=20)
        T, D, _ = J_tensor.shape

        # Reduce vectorised trajectory via SVD (same as archetypes.koopman defaults)
        Z_flat = J_tensor.reshape(T, -1)
        U, S, Vt = np.linalg.svd(Z_flat, full_matrices=False)
        rank = 5
        reduced_rank = min(T - 1, 4 * rank + 8)
        Zr = Z_flat @ Vt[:reduced_rank].T
        print(f"   J_tensor shape={J_tensor.shape}  reduced to r={reduced_rank}")

        # Report SVD spectrum tail — a τ-invariant J should have S[1:] ≪ S[0]
        s_norm = S / S[0]
        print(f"   SVD S ratios (normalised): S[0..5] = {s_norm[:5].tolist()}   "
              f"S[10]/S[0] = {s_norm[10]:.3e}   S[20]/S[0] = {s_norm[20]:.3e}")

        for wh in [4, 8, 12, 16]:
            grid_km = np.linspace(0.02, 0.98, T)
            try:
                K_seq, _ = _local_koopman(Zr, grid_km, window_half=wh, ridge=1e-4)
            except Exception as e:
                print(f"   window_half={wh}: EDMD failed ({e})")
                continue
            K_stack = np.stack(K_seq, axis=0)   # (T, r, r)
            I_r = np.eye(reduced_rank)
            frob_diff = np.linalg.norm(K_stack - I_r, axis=(1, 2))
            rho = np.array([np.max(np.abs(np.linalg.eigvals(K))) for K in K_seq])
            eig_min = np.array([np.min(np.abs(np.linalg.eigvals(K))) for K in K_seq])
            cond_V = np.array([np.linalg.cond(np.linalg.eig(K)[1]) for K in K_seq])
            print(f"   window_half={wh:>2d}  "
                  f"‖K − I‖_F: mean={frob_diff.mean():.4f}, median={np.median(frob_diff):.4f}  "
                  f"ρ(K): [{rho.min():.4f}, {rho.max():.4f}]  |min eig|: [{eig_min.min():.2e}, {eig_min.max():.2e}]  "
                  f"κ(V): median={np.median(cond_V):.1e}")
            results.setdefault(seed, []).append({
                "window_half": wh,
                "K_minus_I_F_mean": float(frob_diff.mean()),
                "K_minus_I_F_median": float(np.median(frob_diff)),
                "rho_K_min": float(rho.min()),
                "rho_K_max": float(rho.max()),
                "min_abs_eig_K_min": float(eig_min.min()),
                "min_abs_eig_K_max": float(eig_min.max()),
                "cond_V_median": float(np.median(cond_V)),
                "reduced_rank": int(reduced_rank),
            })

    (out_dir / "data" / "FOLLOWUP7_K_vs_I.json").write_text(
        json.dumps(results, indent=2, default=lambda x: float(x))
    )
    print(f"\nSaved: {out_dir}/data/FOLLOWUP7_K_vs_I.json")


if __name__ == "__main__":
    main()
