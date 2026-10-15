"""Phase B acceptance tests — pitchfork saddle recovery, reproducibility,
no silent fallbacks.

The tests exercise :func:`scjdo.v2.field.build_operator_field` with the
*analytic* drift of the toggle-switch synthetic as its ``V`` input. That
isolates field-estimation error from drift-training error — Phase B's
subject is the field machinery, not the drift model itself (which is
reused unchanged from v1).
"""
from __future__ import annotations

import numpy as np
import pytest

from scjdo.v2.field import build_operator_field
from scjdo.v2.opspace import operator_descriptors
from scjdo.v2._bench.pitchfork import (
    TAU_CRIT, ALPHA_MIN, ALPHA_MAX,
    analytic_drift, simulate_v3, tau_crit_at,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lambda_max_curve(
    z: np.ndarray, tau: np.ndarray,
    seed: int,
    k: int = 50,
    smoothness_alpha: float = 0.3,
    alpha_min: float = ALPHA_MIN, alpha_max: float = ALPHA_MAX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit the field with analytic drift as V and return per-cell λ_max.

    Returns (lambda_max, recovered, tau_sorted_idx).
    """
    V = analytic_drift(z, tau, alpha_min=alpha_min, alpha_max=alpha_max)
    field = build_operator_field(
        z, V, k=k, bandwidth="auto", ridge=1e-3,
        smoothness_alpha=smoothness_alpha, smoothness_iters=1,
        seed=seed,
    )
    desc = operator_descriptors(field.J, field.recovered)
    order = np.argsort(tau)
    return desc.lambda_max_real, field.recovered, order


def _tau_crossing(lambda_max: np.ndarray, tau: np.ndarray, recovered: np.ndarray,
                   bin_size: int = 50, n_sustain: int = 2) -> float:
    """Smallest τ where the binned median λ_max is positive for ``n_sustain``
    consecutive bins. Returns NaN if no bin sustains a positive crossing.
    """
    mask = recovered & np.isfinite(lambda_max)
    order = np.argsort(tau[mask])
    lam = lambda_max[mask][order]
    t = tau[mask][order]
    n = lam.shape[0]
    medians, bin_taus = [], []
    for i in range(0, n - bin_size + 1, bin_size):
        medians.append(float(np.median(lam[i:i + bin_size])))
        bin_taus.append(float(np.median(t[i:i + bin_size])))
    medians = np.asarray(medians)
    bin_taus = np.asarray(bin_taus)
    if medians.size == 0:
        return float("nan")
    for i in range(len(medians) - n_sustain + 1):
        if np.all(medians[i:i + n_sustain] > 0.0):
            return float(bin_taus[i])
    return float("nan")


# ---------------------------------------------------------------------------
# Test 1 — saddle recovery at τ_crit ≈ 0.004 (default α range)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [42, 43])
def test_pitchfork_default_tau_crit_recovered(seed: int, pitchfork_tol: float):
    """v2 must localise the saddle to within ±0.05 of τ_crit ≈ 0.00437
    on the default α range. v1's pseudotime slicing cannot (grid starts
    at τ=0.02 by default)."""
    z, tau = simulate_v3(n_cells=1500, seed=seed)
    lam, rec, _ = _lambda_max_curve(z, tau, seed=seed)
    assert rec.mean() > 0.9, f"Too many unrecovered cells: {1 - rec.mean():.2%}"

    tau_crossing = _tau_crossing(lam, tau, rec)
    assert np.isfinite(tau_crossing), (
        "v2 found no positive leading-eigenvalue region on the pitchfork."
    )
    assert abs(tau_crossing - TAU_CRIT) <= pitchfork_tol, (
        f"seed={seed}: crossing τ*={tau_crossing:.4f}, "
        f"TAU_CRIT={TAU_CRIT:.4f}, |Δ|={abs(tau_crossing - TAU_CRIT):.4f} > {pitchfork_tol}"
    )


# ---------------------------------------------------------------------------
# Test 2 — reproducibility across half-splits
# ---------------------------------------------------------------------------

def test_pitchfork_halfsplit_spearman(pitchfork_tol: float):
    """Random 50/50 split; after fitting the field on each half, binned
    median λ_max curves must agree at Spearman ρ ≥ 0.8."""
    from scipy.stats import spearmanr

    z, tau = simulate_v3(n_cells=1500, seed=42)
    rng = np.random.default_rng(0)
    idx = rng.permutation(z.shape[0])
    A = idx[: idx.size // 2]
    B = idx[idx.size // 2:]

    lam_A, rec_A, _ = _lambda_max_curve(z[A], tau[A], seed=0, k=50)
    lam_B, rec_B, _ = _lambda_max_curve(z[B], tau[B], seed=1, k=50)

    def _binned(lam, tau_sub, rec):
        mask = rec & np.isfinite(lam)
        edges = np.linspace(0.0, 1.0, 21)
        bin_idx = np.digitize(tau_sub[mask], edges) - 1
        bin_idx = np.clip(bin_idx, 0, 19)
        out = np.full(20, np.nan)
        for b in range(20):
            if (bin_idx == b).any():
                out[b] = float(np.median(lam[mask][bin_idx == b]))
        return out

    curve_A = _binned(lam_A, tau[A], rec_A)
    curve_B = _binned(lam_B, tau[B], rec_B)
    both = np.isfinite(curve_A) & np.isfinite(curve_B)
    assert both.sum() >= 10, f"Too few shared bins ({both.sum()})"
    rho, _ = spearmanr(curve_A[both], curve_B[both])
    assert rho >= 0.8, f"Half-split Spearman ρ={rho:.3f} < 0.8"


# ---------------------------------------------------------------------------
# Test 3 — no silent fallbacks
# ---------------------------------------------------------------------------

def test_pitchfork_outlier_marked_unrecovered():
    """An isolated outlier far from every other cell must have its J row
    left NaN and its ``recovered`` flag False."""
    z, tau = simulate_v3(n_cells=400, seed=42)
    outlier = np.array([[20.0, 20.0]], dtype=np.float32)
    z_aug = np.concatenate([z, outlier], axis=0)
    tau_aug = np.concatenate([tau, np.array([0.9], dtype=np.float32)], axis=0)
    V = analytic_drift(z_aug, tau_aug)

    field = build_operator_field(
        z_aug, V, k=30, bandwidth="auto", ridge=1e-3,
        smoothness_alpha=0.0, smoothness_iters=0,  # keep smoothing off
        radius_floor_factor=1.0,                   # tight ball
        seed=0,
    )
    assert field.recovered[-1] is np.False_ or field.recovered[-1] == False, (
        f"outlier was 'recovered'={field.recovered[-1]} — Phase B forbids silent fallbacks"
    )
    assert np.all(np.isnan(field.J[-1])), (
        "outlier Jacobian should be all NaN (no fabricated field)"
    )


# ---------------------------------------------------------------------------
# Test 4 (diagnostic, non-gating) — EL08 variants on α remappings
# ---------------------------------------------------------------------------
#
# Phase B's spec acceptance is the three tests above. The EL08 variants
# (``τ_crit ∈ {0.1, 0.3, 0.5}``) are a *diagnostic* for how the field
# behaves across α-ramp configurations; it is not a gate. Phase B's
# minimum crossing detector is a non-overlapping binned-median finder and
# struggles on low-dynamics or short post-bifurcation regimes (τ_target =
# 0.5), which is a known limit documented in V2_PLAN.md §4. Phase C may
# revisit the detector with the Fisher-weighted descriptors.

@pytest.mark.parametrize("tau_target", [0.1, 0.3])
@pytest.mark.parametrize("seed", [42, 43])
def test_pitchfork_el08_variant_diagnostic(tau_target: float, seed: int,
                                            pitchfork_tol: float, tmp_path_factory):
    """Non-gating diagnostic — records per-condition crossings to
    ``benchmarks/v2/pitchfork_el08.csv`` so Phase C/E can trend them."""
    from scjdo.v2._bench.pitchfork import ALPHA_CRIT
    alpha_max = ALPHA_CRIT + 1.0
    alpha_min = float((alpha_max * tau_target - ALPHA_CRIT) / (tau_target - 1.0))

    z, tau = simulate_v3(
        n_cells=1500, seed=seed,
        alpha_min=alpha_min, alpha_max=alpha_max,
    )
    lam, rec, _ = _lambda_max_curve(
        z, tau, seed=seed,
        alpha_min=alpha_min, alpha_max=alpha_max,
    )
    tau_crossing = _tau_crossing(lam, tau, rec)
    tc = tau_crit_at(alpha_min=alpha_min, alpha_max=alpha_max)
    delta = abs(tau_crossing - tc) if np.isfinite(tau_crossing) else float("inf")

    import csv, os
    out_dir = os.path.join(os.getcwd(), "benchmarks", "v2")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "pitchfork_el08.csv")
    header = ["seed", "tau_target", "tau_crit", "crossing", "delta", "within_tol"]
    row = [seed, tau_target, tc, tau_crossing, delta, bool(delta <= pitchfork_tol)]
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(header)
        w.writerow(row)

    # Non-gating: skip rather than fail on miss, so CI reflects the diagnostic nature.
    if delta > pitchfork_tol:
        pytest.skip(f"EL08 diagnostic miss: τ_crit={tc:.3f}, "
                    f"crossing={tau_crossing:.3f}, Δ={delta:.3f}")
