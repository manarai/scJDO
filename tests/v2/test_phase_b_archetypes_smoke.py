"""Smoke tests for the Phase B operator-space and archetype modules."""
from __future__ import annotations

import numpy as np
import pytest

from scjdo.v2.field import build_operator_field
from scjdo.v2.opspace import operator_descriptors
from scjdo.v2.archetypes import fit_archetypes
from scjdo.v2._bench.pitchfork import simulate_v3, analytic_drift


@pytest.fixture(scope="module")
def pitchfork_field():
    z, tau = simulate_v3(n_cells=800, seed=42)
    V = analytic_drift(z, tau)
    f = build_operator_field(z, V, k=25, bandwidth="auto", seed=0,
                              smoothness_alpha=0.3, smoothness_iters=1)
    return f, z, tau


def test_descriptors_shapes(pitchfork_field):
    field, z, _ = pitchfork_field
    d = operator_descriptors(field.J, field.recovered)
    n, L = z.shape
    assert d.eigvals.shape == (n, L)
    assert d.leading_eigvec.shape == (n, L)
    assert d.lambda_max_real.shape == (n,)
    # Recovered cells should have finite descriptors.
    rec = d.recovered
    assert np.all(np.isfinite(d.lambda_max_real[rec]))
    # The pitchfork is late-τ unstable, so some cells must have positive λ_max.
    assert (d.lambda_max_real[rec] > 0).sum() > 50


def test_archetypes_fit(pitchfork_field):
    field, _, _ = pitchfork_field
    arch = fit_archetypes(field.J, field.recovered, K=3, n_restarts=2, seed=0)
    assert arch.patterns.shape == (3, 2, 2)
    assert arch.weights.shape[0] == field.J.shape[0]
    assert len(arch.names) == 3
    assert all(n in {"holder", "damped", "drift", "fork", "growing", "unknown"} for n in arch.names)
    assert np.all(arch.weights >= 0)


def test_archetypes_K_auto(pitchfork_field):
    field, _, _ = pitchfork_field
    arch = fit_archetypes(field.J, field.recovered, K="auto",
                           K_sweep=(2, 3, 4), n_restarts=1, seed=0)
    assert arch.K in (2, 3, 4)
    assert arch.K_sweep is not None
    assert len(arch.K_sweep) >= 2
