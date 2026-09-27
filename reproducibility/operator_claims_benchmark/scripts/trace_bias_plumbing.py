"""r25 diagnostic — is bias_strength actually reaching the model, or is
its effect swamped by the pseudotime-gradient prior + init?

Reproduces the exact sequence fit_drift_branches uses to build V for the
Ery branch, and compares V under bias_strength = {0.0, 1.5} on identical
inputs. If V_bias == V_plain to machine precision, the bias code path is
a no-op. If they differ but the basin-selection outcome is invariant,
that's a different story (model dynamics dominate bias signal)."""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "reproducibility" / "operator_claims_benchmark" / "scripts"))

from run_operator_claims import prepare_marrow_ery
from scjdo.tl._drift import _biased_velocity, _pseudotime_velocity

def main():
    ery = prepare_marrow_ery()
    X = ery.obsm["X_fa"]
    pt = ery.obs["pseudotime"].values.astype(np.float32)

    # Match r24's cell_fate labelling
    pt_low, pt_hi = np.quantile(pt, 0.3), np.quantile(pt, 0.7)
    prog_mask = pt <= pt_low
    term_mask = pt >= pt_hi
    prog_cen = X[prog_mask].mean(axis=0)
    term_cen = X[term_mask].mean(axis=0)
    print(f"n_prog={prog_mask.sum()}  n_term={term_mask.sum()}")
    print(f"progenitor centroid ||: {np.linalg.norm(prog_cen):.4f}")
    print(f"terminal centroid ||:   {np.linalg.norm(term_cen):.4f}")
    print(f"prog→term axis ||:      {np.linalg.norm(term_cen - prog_cen):.4f}")

    V_pt    = _pseudotime_velocity(X, pt, k=15)
    V_b0    = _biased_velocity(X, pt, terminal_centroid=term_cen,
                                 progenitor_centroid=prog_cen,
                                 bias_strength=0.0, k=15)
    V_b1p5  = _biased_velocity(X, pt, terminal_centroid=term_cen,
                                 progenitor_centroid=prog_cen,
                                 bias_strength=1.5, k=15)

    def _cmp(name, A, B):
        diff = A - B
        maxabs = float(np.abs(diff).max())
        cos_per_cell = ((A * B).sum(axis=1) /
                        (np.linalg.norm(A, axis=1) *
                          np.linalg.norm(B, axis=1) + 1e-9))
        print(f"  {name}: max |A-B| = {maxabs:.6f}  "
              f"mean per-cell cos = {cos_per_cell.mean():.6f}  "
              f"min per-cell cos = {cos_per_cell.min():.6f}  "
              f"||A|| mean = {np.linalg.norm(A, axis=1).mean():.4f}  "
              f"||B|| mean = {np.linalg.norm(B, axis=1).mean():.4f}")

    print("\n=== _pseudotime_velocity vs _biased_velocity(bias=0.0) ===")
    _cmp("V_pt vs V_b0", V_pt, V_b0)
    print("\n=== _biased_velocity(bias=0.0) vs _biased_velocity(bias=1.5) ===")
    _cmp("V_b0 vs V_b1.5", V_b0, V_b1p5)
    print("\n=== _pseudotime_velocity vs _biased_velocity(bias=1.5) ===")
    _cmp("V_pt vs V_b1.5", V_pt, V_b1p5)
    print()

    # Also check what fit_drift_branches actually passes to fit_drift:
    # under bias>0 it monkey-patches _pseudotime_velocity to return V_biased.
    # Under bias==0 it does NOT monkey-patch — plain _pseudotime_velocity runs.
    # So the effective V is:
    #   bias=0.0 → V_pt (plain _pseudotime_velocity, not V_b0)
    #   bias=1.5 → V_b1.5 (the monkey-patched biased velocity)
    print("*** effective V comparison (what actually reaches DriftField) ***")
    _cmp("bias=0.0 (V_pt) vs bias=1.5 (V_b1.5)", V_pt, V_b1p5)

if __name__ == "__main__":
    main()
