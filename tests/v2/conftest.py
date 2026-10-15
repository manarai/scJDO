"""Shared fixtures for scjdo v2 tests.

Mirrors tests/conftest.py but scoped to v2. Keeps v1 and v2 test isolation
intact: v1 tests see only the v1 conftest, v2 tests see this one plus the
root conftest via pytest's rootdir rules.
"""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _v2_reproducibility_seed():
    seed = 13
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@pytest.fixture
def pitchfork_tau_crits() -> tuple[float, ...]:
    """τ_crit grid for the 2-branch pitchfork benchmark (EL08).

    The toggle-switch ground truth in reproducibility/code/toggle_truth.py
    sweeps α so that τ_crit lands at these pseudotime positions. The
    predeclared tolerance window is ±0.05 pseudotime.
    """
    return (0.1, 0.3, 0.5)


@pytest.fixture
def pitchfork_seeds() -> tuple[int, ...]:
    return (42, 43)


@pytest.fixture
def pitchfork_tol() -> float:
    return 0.05
