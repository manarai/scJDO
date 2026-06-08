"""
tests/test_hybrid_grn.py
=========================
Minimal acceptance tests for the HybridGRN extension.

These tests verify that the mathematical objects (J_z, J_x, K_x) have
the correct shapes and that the pullback logic is consistent.
"""
import pytest
import torch
import numpy as np

from scjdo.models.drift import DriftField, DriftConfig
from scjdo.models.representation import (
    RepresentationConfig,
    PCARep,
    LDVAERep,
    VegaRep,
)
from scjdo.grn.pullback import pullback_gene_operator
from scjdo.grn.refine import GRNRefinerConfig, SparseGRNRefiner
from scjdo.grn.archetypes import grn_modes


@pytest.fixture
def dummy_data():
    B, G, D = 10, 50, 5
    x = torch.randn(B, G)
    z = torch.randn(B, D)
    t = torch.rand(B)
    Jz = torch.randn(B, D, D)
    return x, z, t, Jz, B, G, D


def test_pca_rep_pullback(dummy_data):
    x, z, t, Jz, B, G, D = dummy_data
    
    # Create PCARep
    components = torch.randn(D, G)
    rep = PCARep(components)
    
    # Test linear pullback
    Jx = pullback_gene_operator(rep, x, z, Jz, mode="linear")
    assert Jx.shape == (B, G, G)
    
    # Test tf_restricted pullback
    tf_index = torch.tensor([0, 5, 10])
    Jx_tf = pullback_gene_operator(rep, x, z, Jz, mode="tf_restricted", tf_index=tf_index)
    assert Jx_tf.shape == (B, G, 3)
    
    # Check consistency
    assert torch.allclose(Jx[:, :, tf_index], Jx_tf, atol=1e-5)


def test_ldvae_rep_pullback(dummy_data):
    x, z, t, Jz, B, G, D = dummy_data
    
    cfg = RepresentationConfig(backend="ldvae", n_latent=D, n_genes=G)
    rep = LDVAERep(cfg)
    
    # Test autograd pullback
    Jx = pullback_gene_operator(rep, x, z, Jz, mode="autograd")
    assert Jx.shape == (B, G, G)
    
    # Test tf_restricted pullback
    tf_index = torch.tensor([1, 2, 3, 4])
    Jx_tf = pullback_gene_operator(rep, x, z, Jz, mode="tf_restricted", tf_index=tf_index)
    assert Jx_tf.shape == (B, G, 4)
    
    # Check consistency
    assert torch.allclose(Jx[:, :, tf_index], Jx_tf, atol=1e-4)


def test_sparse_grn_refiner():
    T, n_tf, G = 5, 10, 50
    Jx = torch.randn(T, n_tf, G)
    
    cfg = GRNRefinerConfig(lambda_sparse=0.1, n_steps=10)
    refiner = SparseGRNRefiner(cfg, n_tf=n_tf, n_genes=G)
    
    Kx = refiner.fit(Jx)
    assert Kx.shape == (T, n_tf, G)
    
    # Check that L1 penalty induced some sparsity (or at least shrinkage)
    assert Kx.abs().mean() < Jx.abs().mean()


def test_grn_modes():
    T, n_tf, G = 20, 15, 100
    Kx = torch.randn(T, n_tf, G)
    
    rank = 3
    result = grn_modes(Kx, rank=rank, center=True)
    
    assert result.archetypes.shape == (rank, n_tf, G)
    assert result.activations.shape == (T, rank)
    assert result.singular_values.shape == (rank,)
    assert result.tf_scores.shape == (rank, n_tf)
    assert result.gene_scores.shape == (rank, G)
