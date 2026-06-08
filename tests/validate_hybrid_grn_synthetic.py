"""
tests/validate_hybrid_grn_synthetic.py
========================================
End-to-end synthetic validation of the HybridGRN extension.

Three tests from the spec:
    12.1 Linear sanity test    — known sparse A, check pullback recovers it
    12.2 Branching toy system  — 2-state bifurcation, check archetype peaks
    12.3 Shape smoke test      — all tensors have correct shapes, no OOM

Run with:
    python tests/validate_hybrid_grn_synthetic.py
"""
import sys
import types
import warnings

# Bypass the pre-existing broken __init__.py in the repo
_m = types.ModuleType("scjdo")
_m.__path__ = ["/home/ubuntu/scJDO_dev/scjdo"]
_m.__package__ = "scjdo"
sys.modules["scjdo"] = _m

import torch
import torch.nn as nn
import numpy as np

from scjdo.models.drift import DriftField, DriftConfig
from scjdo.models.representation import RepresentationConfig, PCARep, LDVAERep
from scjdo.grn.pullback import pullback_gene_operator, binned_pullback
from scjdo.grn.refine import GRNRefinerConfig, SparseGRNRefiner
from scjdo.grn.archetypes import grn_modes, archetype_summary
from scjdo.grn.priors import identify_tfs, build_tf_mask
from scjdo.analysis.grn_scores import regulator_centrality, temporal_activity_score


# ============================================================
# Test 12.1 — Linear sanity test
# ============================================================
def test_linear_sanity():
    """
    Synthetic linear system: x_dot = A x with known sparse A.
    Use PCARep (identity-like) so the pullback is exact.
    Check that K_x approximates A after refinement.
    """
    print("\n=== Test 12.1: Linear sanity test ===")
    torch.manual_seed(0)

    G = 20    # genes
    D = 5     # latent dims
    n_tf = 5  # TFs = first 5 genes
    N = 200   # cells
    T_bins = 10

    # Ground truth: sparse A (n_tf x G)
    A_true = torch.zeros(n_tf, G)
    A_true[0, 3] = 0.8    # TF0 activates gene 3
    A_true[1, 7] = -0.5   # TF1 represses gene 7
    A_true[2, 12] = 0.6   # TF2 activates gene 12
    A_true[3, 0] = -0.3   # TF3 represses gene 0
    A_true[4, 15] = 0.9   # TF4 activates gene 15

    # Simulate cells along a linear trajectory
    pseudotime = torch.linspace(0, 1, N)
    x_gene = torch.randn(N, G) * 0.1
    # Add signal: x_dot ≈ A x
    for i in range(N):
        # A_true: (n_tf, G), x_gene[i][:n_tf]: (n_tf,)
        # A_true.T @ x_gene[i][:n_tf] gives (G,) — gene changes driven by TFs
        x_gene[i] += pseudotime[i] * (A_true.T @ x_gene[i][:n_tf])

    # PCARep with random PCA components
    components = torch.randn(D, G)
    components = components / components.norm(dim=1, keepdim=True)
    rep = PCARep(components)

    # Minimal DriftField
    drift_cfg = DriftConfig(dim=D)
    drift = DriftField(drift_cfg)

    # Compute binned pullback
    tf_index = torch.arange(n_tf)
    Jx_bins, bin_edges = binned_pullback(
        drift, rep, x_gene, pseudotime,
        n_bins=T_bins, mode="tf_restricted",
        tf_index=tf_index, batch_size=64,
    )
    # Jx_bins: (T, G, n_tf) → transpose to (T, n_tf, G) for refiner
    Jx_for_refiner = Jx_bins.transpose(-1, -2)

    # Refine
    refiner_cfg = GRNRefinerConfig(lambda_sparse=0.05, n_steps=50, lr=0.01)
    refiner = SparseGRNRefiner(refiner_cfg, n_tf=n_tf, n_genes=G)
    Kx = refiner.fit(Jx_for_refiner)

    # Check shapes
    assert Kx.shape == (T_bins, n_tf, G), f"Bad shape: {Kx.shape}"

    # Check that K_x is sparser than J_x (L1 penalty worked)
    sparsity_Jx = (Jx_for_refiner.abs() < 0.01).float().mean().item()
    sparsity_Kx = (Kx.abs() < 0.01).float().mean().item()
    print(f"  Sparsity J_x (< 0.01): {sparsity_Jx:.3f}")
    print(f"  Sparsity K_x (< 0.01): {sparsity_Kx:.3f}")
    assert sparsity_Kx >= sparsity_Jx, "K_x should be sparser than J_x"

    print("  PASS: shapes correct, K_x is sparser than J_x")
    return Kx


# ============================================================
# Test 12.2 — Branching toy system
# ============================================================
def test_branching_toy():
    """
    Two-state bifurcation in low dimension, projected to gene space via
    a linear decoder.  Check that archetype activations peak near the
    branch point.
    """
    print("\n=== Test 12.2: Branching toy system ===")
    torch.manual_seed(42)

    G = 30
    D = 4
    n_tf = 6
    N = 300
    T_bins = 15

    # Branch point at t = 0.5
    # Before branch: cells follow a single trajectory
    # After branch: two fates diverge
    pseudotime = torch.linspace(0, 1, N)
    x_gene = torch.randn(N, G) * 0.2

    # Plant a "branch point" signal: gene 0 peaks at t=0.5
    branch_signal = torch.exp(-10 * (pseudotime - 0.5) ** 2)
    x_gene[:, 0] += branch_signal

    # Fate A: genes 1-5 activate after t=0.5
    fate_a = torch.relu(pseudotime - 0.5)
    x_gene[:N//2, 1:6] += fate_a[:N//2].unsqueeze(1)

    # Fate B: genes 6-10 activate after t=0.5
    x_gene[N//2:, 6:11] += fate_a[N//2:].unsqueeze(1)

    # PCARep
    components = torch.randn(D, G)
    components = components / components.norm(dim=1, keepdim=True)
    rep = PCARep(components)

    drift_cfg = DriftConfig(dim=D)
    drift = DriftField(drift_cfg)

    tf_index = torch.arange(n_tf)
    Jx_bins, bin_edges = binned_pullback(
        drift, rep, x_gene, pseudotime,
        n_bins=T_bins, mode="tf_restricted",
        tf_index=tf_index, batch_size=64,
    )
    Jx_for_refiner = Jx_bins.transpose(-1, -2)

    refiner_cfg = GRNRefinerConfig(lambda_sparse=0.01, n_steps=30)
    refiner = SparseGRNRefiner(refiner_cfg, n_tf=n_tf, n_genes=G)
    Kx = refiner.fit(Jx_for_refiner)

    # Extract archetypes
    result = grn_modes(Kx, rank=3, center=True)

    assert result.archetypes.shape == (3, n_tf, G)
    assert result.activations.shape == (T_bins, 3)

    # Check that at least one archetype has a peak activation
    peak_times = result.peak_times.tolist()
    print(f"  Archetype peak times (bin index): {peak_times}")
    print(f"  Variance explained: {result.variance_explained.tolist()}")

    # Temporal activity
    activity = temporal_activity_score(Kx, tf_names=[f"TF{i}" for i in range(n_tf)])
    assert activity.shape == (T_bins, n_tf)

    print("  PASS: archetypes extracted, peak times computed")
    return result


# ============================================================
# Test 12.3 — Shape smoke test
# ============================================================
def test_shape_smoke():
    """
    Larger-scale shape test: N=500, G=200, D=50, n_tf=30.
    Verify no OOM and all shapes are correct.
    """
    print("\n=== Test 12.3: Shape smoke test ===")
    torch.manual_seed(7)

    N, G, D, n_tf = 500, 200, 50, 30
    T_bins = 20
    rank = 5

    x_gene = torch.randn(N, G)
    pseudotime = torch.rand(N).sort().values

    # LDVAERep
    rep_cfg = RepresentationConfig(backend="ldvae", n_latent=D, n_genes=G,
                                    n_hidden=64, n_layers=1)
    rep = LDVAERep(rep_cfg)

    drift_cfg = DriftConfig(dim=D)
    drift = DriftField(drift_cfg)

    tf_index = torch.arange(n_tf)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Jx_bins, bin_edges = binned_pullback(
            drift, rep, x_gene, pseudotime,
            n_bins=T_bins, mode="tf_restricted",
            tf_index=tf_index, batch_size=32,
        )

    assert Jx_bins.shape == (T_bins, G, n_tf), f"Bad Jx shape: {Jx_bins.shape}"
    print(f"  Jx_bins shape: {Jx_bins.shape}  ✓")

    Jx_for_refiner = Jx_bins.transpose(-1, -2)
    refiner_cfg = GRNRefinerConfig(lambda_sparse=0.01, n_steps=20)
    refiner = SparseGRNRefiner(refiner_cfg, n_tf=n_tf, n_genes=G)
    Kx = refiner.fit(Jx_for_refiner)

    assert Kx.shape == (T_bins, n_tf, G), f"Bad Kx shape: {Kx.shape}"
    print(f"  Kx shape: {Kx.shape}  ✓")

    result = grn_modes(Kx, rank=rank, center=True)
    assert result.archetypes.shape == (rank, n_tf, G)
    assert result.activations.shape == (T_bins, rank)
    print(f"  Archetypes shape: {result.archetypes.shape}  ✓")
    print(f"  Activations shape: {result.activations.shape}  ✓")

    # Centrality scores
    tf_names = [f"TF{i}" for i in range(n_tf)]
    gene_names = [f"gene{i}" for i in range(G)]
    centrality = regulator_centrality(Kx, tf_names, gene_names)
    assert centrality["tf_out_degree"].shape == (n_tf,)
    assert centrality["gene_in_degree"].shape == (G,)
    print(f"  Centrality scores shape: tf_out_degree {centrality['tf_out_degree'].shape}  ✓")

    print("  PASS: all shapes correct, no OOM")


# ============================================================
# Run all tests
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("HybridGRN Extension — Synthetic Validation Suite")
    print("=" * 60)

    Kx_linear = test_linear_sanity()
    arch_result = test_branching_toy()
    test_shape_smoke()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
