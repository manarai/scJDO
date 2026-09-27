"""
Regression fixtures for the bifurcation-saddle investigation.

T2.3 — Lifted-truth positive control: the analytic 2-D toggle switch, lifted
into the 20-D latent frame via the same PCA projection used on the real
pipeline, must produce (in the at_fp variant, i.e. J evaluated at the
symmetric fixed point):

    argmax_{τ ∈ [0.05, 0.95]}  Re(λ_max(J̄_signal(τ)))  →  0.95
    crossing_{τ}  −1 + c(α(τ))  →  τ_crit  ± grid step

T3 — Antisymmetric eigenvector labelling: at (s(α), s(α)) the leading
eigenvalue of J is −1 + c and its eigenvector is (1, −1)/sqrt(2).

Run:  pytest -q reproducibility/tests/test_lifted_truth_regression.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "code"))

import toggle_truth as gt                                              # noqa: E402
from T2_lifted_truth import lifted_analytic_readouts                    # noqa: E402


@pytest.mark.parametrize("seed", [42, 1, 7])
def test_at_fp_argmax_lands_at_right_edge(seed):
    """The brief's §4 prediction: monotone → argmax at upper interior edge."""
    r = lifted_analytic_readouts(seed=seed, n_cells=1500)
    # The interior is [0.05, 0.95] by construction; 0.95 is the upper clamp.
    assert r["tau_argmax_signal_fp"] == pytest.approx(0.95, abs=0.02), (
        f"[seed={seed}] at_fp argmax(signal) = {r['tau_argmax_signal_fp']} "
        "should saturate the right edge — argmax is scoring a monotone curve."
    )


@pytest.mark.parametrize("seed", [42, 1, 7])
def test_analytic_lambda_perp_crossing_is_tau_crit(seed):
    r = lifted_analytic_readouts(seed=seed, n_cells=1500)
    # Grid spacing is 1/199 ≈ 5e-3.
    assert abs(r["tau_cross_analytic"] - r["tau_crit"]) < 6e-3, (
        f"[seed={seed}] analytic crossing = {r['tau_cross_analytic']}, "
        f"τ_crit = {r['tau_crit']}."
    )


def test_antisymmetric_eigenvector_at_fixed_point():
    v_perp = np.array([1.0, -1.0]) / np.sqrt(2.0)
    v_par = np.array([1.0, 1.0]) / np.sqrt(2.0)
    for a in [0.5, 1.0, 1.05, 1.2, 1.5, 2.0, 3.0, 4.0]:
        s = gt.sym_fp(a)
        assert abs(s + s ** 5 - a) < 1e-12, f"fp iteration failed at α={a}"
        J = gt.toggle_jacobian_at(s, s, a)
        eigvals, eigvecs = np.linalg.eig(J)
        order = np.argsort(-eigvals.real)
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        c = 4.0 * a * s ** 3 / (1.0 + s ** 4) ** 2
        assert np.isclose(eigvals[0].real, -1.0 + c), f"leading eig at α={a}"
        assert np.isclose(eigvals[1].real, -1.0 - c), f"trailing eig at α={a}"
        v0 = eigvecs[:, 0].real
        v0 /= np.linalg.norm(v0)
        assert abs(abs(v0 @ v_perp) - 1.0) < 1e-8, f"leading vec at α={a}"
        v1 = eigvecs[:, 1].real
        v1 /= np.linalg.norm(v1)
        assert abs(abs(v1 @ v_par) - 1.0) < 1e-8, f"trailing vec at α={a}"


def test_plus_sign_symmetrizer_is_noop_on_analytic_drift():
    """T8 mandatory test: f_sym applied to the analytic drift must return it."""
    from T8_symmetrizer import (make_reflection_latent, symmetrize,
                                 lifted_analytic_drift_factory)
    z_true, _ = gt.simulate_v3(n_cells=1500, seed=42)
    W = gt.build_observation_map(seed=42)
    Z_lat, A, _ = gt.latent_axes(z_true, W, seed=42)
    sigma_z = make_reflection_latent(A)
    z_test = Z_lat[:20].astype(np.float64)

    drift_at_alpha = lifted_analytic_drift_factory(A)
    for alpha in [0.7, 1.0, 1.5, 2.5, 4.0]:
        f_true = drift_at_alpha(alpha)
        f_sym_plus = symmetrize(f_true, sigma_z, sign="+")
        f_sym_minus = symmetrize(f_true, sigma_z, sign="-")
        f_direct = f_true(z_test)
        f_p = f_sym_plus(z_test)
        f_m = f_sym_minus(z_test)
        norm = max(1.0, float(np.max(np.abs(f_direct))))
        assert np.max(np.abs(f_p - f_direct)) < 1e-4 * norm, f"+ symmetrizer α={alpha}"
        assert np.max(np.abs(f_m)) < 1e-4 * norm, f"− symmetrizer annihilation α={alpha}"


def test_transverse_flow_sign_matches_lambda_perp():
    eps = 1e-3
    for a in [0.5, 1.0, 1.05, 1.2, 2.0, 4.0]:
        s = gt.sym_fp(a)
        z = np.array([[s + eps, s - eps]])
        f = gt.toggle_drift(z, a)[0]
        w = (z[0, 0] - z[0, 1]) / np.sqrt(2.0)
        wdot = (f[0] - f[1]) / np.sqrt(2.0)
        c = 4.0 * a * s ** 3 / (1.0 + s ** 4) ** 2
        assert np.sign(wdot * w) == np.sign(-1.0 + c), (
            f"transverse flow sign mismatch at α={a}"
        )
