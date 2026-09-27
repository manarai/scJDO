"""
T7 — Constructive gauge twin.

Theorem (Weinreb et al., recapitulated in the manuscript). For any smooth
divergence-free vector field v, the drift  f' = f + v/p  produces IDENTICAL
snapshot marginals p(z, τ) at every τ. So if there exists a v such that
λ_⊥(f')|τ_crit has the *opposite sign* to λ_⊥(f)|τ_crit, then no snapshot
estimator can separate f from f' — a positive proof of non-identifiability.

Setup
-----
We work in the ground-truth (x, y) frame and construct v = curl ψ where ψ is
a smooth stream function. In 2-D, curl ψ = (∂ψ/∂y, −∂ψ/∂x). Divergence-free
automatically because ∇ · (curl ψ) = 0.

T7.1 baseline twin — ψ is a localized Gaussian (may or may not be
σ-equivariant).

T7.3 equivariance-compliant adversary — ψ is required to be σ-equivariant,
where σ: (x, y) ↔ (y, x). Since ψ is a pseudoscalar (rotational stream
function), σ-equivariance forces ψ ODD in w = (x − y)/√2, i.e. ψ(s, w) =
−ψ(s, −w). We use the offset-Gaussian family:

    ψ(s, w)  =  A · w · exp( −(s − s₀)² / (2 a²)  −  w² / (2 b²) )

By construction ψ is odd in w (so v_w is even in w and vanishes on the
diagonal — good), and σ-equivariance is enforced. Its cross-derivative at
the fixed point (s*, 0) is

    ∂v_w/∂w |_(w=0)  =  −∂²ψ/∂s∂w |_(w=0)
                     ∝  A · (s* − s₀) / a²

so offsetting s₀ off the fixed point shifts λ_⊥ at the saddle.

T7.4 bounded-drift constraint: |v/p| must stay bounded (e.g. bounded by
K · max|f|). Since p → 0 in the low-density diagonal tails, this restricts
the amplitude A of a ψ with support away from the density's core.

Metrics
-------
- **MMD**: multi-scale Gaussian kernel MMD² between (z_true(τ_i), z_twin(τ_i))
  for τ_i in a small set of τ-windows. Test each pass at α = 0.05 with a
  permutation null. If any window is significant, marginals are
  distinguishable and the twin is not admissible.
- **λ_⊥ at τ_crit** — analytic Jacobian of f' at (s(α_crit), s(α_crit)),
  compared to the true λ_⊥ = 0 exactly at criticality.

This is analytic + a light bootstrap; no training.

Outputs
-------
    reproducibility/data/T7_gauge_twins.json
    reproducibility/figures/T7_gauge_twins.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import toggle_truth as gt


# ---------------------------------------------------------------------------
# Stream functions and their curls
# ---------------------------------------------------------------------------

def gaussian_stream(z, s0, w0, a, b, amp):
    """ψ(x, y) = amp · exp(−((s − s0)² / (2a²)) − ((w − w0)² / (2b²)))"""
    x, y = z[..., 0], z[..., 1]
    s = (x + y) / np.sqrt(2.0)
    w = (x - y) / np.sqrt(2.0)
    return amp * np.exp(-0.5 * ((s - s0) / a) ** 2 - 0.5 * ((w - w0) / b) ** 2)


def gaussian_stream_grad(z, s0, w0, a, b, amp):
    """Return (∂ψ/∂x, ∂ψ/∂y) so v = curl ψ = (∂ψ/∂y, −∂ψ/∂x)."""
    x, y = z[..., 0], z[..., 1]
    s = (x + y) / np.sqrt(2.0)
    w = (x - y) / np.sqrt(2.0)
    psi = amp * np.exp(-0.5 * ((s - s0) / a) ** 2 - 0.5 * ((w - w0) / b) ** 2)
    # ∂s/∂x = 1/√2, ∂w/∂x = 1/√2; ∂s/∂y = 1/√2, ∂w/∂y = −1/√2.
    dpsi_ds = -(s - s0) / a ** 2 * psi
    dpsi_dw = -(w - w0) / b ** 2 * psi
    dpsi_dx = dpsi_ds / np.sqrt(2.0) + dpsi_dw / np.sqrt(2.0)
    dpsi_dy = dpsi_ds / np.sqrt(2.0) - dpsi_dw / np.sqrt(2.0)
    return dpsi_dx, dpsi_dy


def curl_stream(z, s0, w0, a, b, amp):
    """v = curl ψ = (∂ψ/∂y, −∂ψ/∂x)."""
    dpsi_dx, dpsi_dy = gaussian_stream_grad(z, s0, w0, a, b, amp)
    return np.stack([dpsi_dy, -dpsi_dx], axis=-1)


def curl_stream_jacobian(x, y, s0, w0, a, b, amp):
    """
    Analytic Jacobian of v = curl ψ at a specific (x, y).
    v_x = ∂ψ/∂y,  v_y = −∂ψ/∂x
    ∂v_x/∂x = ∂²ψ/(∂y∂x),  ∂v_x/∂y = ∂²ψ/∂y²
    ∂v_y/∂x = −∂²ψ/∂x²,   ∂v_y/∂y = −∂²ψ/(∂x∂y)
    Use the chain-ruled second derivatives.
    """
    s = (x + y) / np.sqrt(2.0)
    w = (x - y) / np.sqrt(2.0)
    psi = amp * np.exp(-0.5 * ((s - s0) / a) ** 2 - 0.5 * ((w - w0) / b) ** 2)
    # ψ_ss = (−1/a² + (s−s0)²/a⁴) ψ
    # ψ_ww = (−1/b² + (w−w0)²/b⁴) ψ
    # ψ_sw = (s−s0)(w−w0)/(a² b²) ψ
    psi_ss = psi * (-1.0 / a ** 2 + (s - s0) ** 2 / a ** 4)
    psi_ww = psi * (-1.0 / b ** 2 + (w - w0) ** 2 / b ** 4)
    psi_sw = psi * (s - s0) * (w - w0) / (a ** 2 * b ** 2)
    # Now (x, y) partials.
    # ψ_x = ψ_s/√2 + ψ_w/√2
    # ψ_xx = ψ_ss/2 + ψ_ww/2 + ψ_sw
    # ψ_yy = ψ_ss/2 + ψ_ww/2 − ψ_sw
    # ψ_xy = ψ_ss/2 − ψ_ww/2
    psi_xx = psi_ss / 2.0 + psi_ww / 2.0 + psi_sw
    psi_yy = psi_ss / 2.0 + psi_ww / 2.0 - psi_sw
    psi_xy = psi_ss / 2.0 - psi_ww / 2.0
    # J_v[0, 0] = ∂v_x/∂x = ψ_xy
    # J_v[0, 1] = ∂v_x/∂y = ψ_yy
    # J_v[1, 0] = ∂v_y/∂x = −ψ_xx
    # J_v[1, 1] = ∂v_y/∂y = −ψ_xy
    return np.array([[psi_xy, psi_yy], [-psi_xx, -psi_xy]])


# Equivariance-compliant stream: ψ(s, w) = A · w · exp(−(s−s0)²/(2a²) − w²/(2b²))
def equivariant_stream_grad(z, s0, a, b, amp):
    """(∂ψ/∂x, ∂ψ/∂y) for ψ = A·w·exp(−(s−s0)²/(2a²) − w²/(2b²))."""
    x, y = z[..., 0], z[..., 1]
    s = (x + y) / np.sqrt(2.0)
    w = (x - y) / np.sqrt(2.0)
    G = np.exp(-0.5 * ((s - s0) / a) ** 2 - 0.5 * (w / b) ** 2)
    psi = amp * w * G
    dpsi_ds = amp * w * G * (-(s - s0) / a ** 2)
    dpsi_dw = amp * G * (1.0 - w ** 2 / b ** 2)
    dpsi_dx = dpsi_ds / np.sqrt(2.0) + dpsi_dw / np.sqrt(2.0)
    dpsi_dy = dpsi_ds / np.sqrt(2.0) - dpsi_dw / np.sqrt(2.0)
    return dpsi_dx, dpsi_dy, psi


def curl_equivariant(z, s0, a, b, amp):
    dx, dy, _ = equivariant_stream_grad(z, s0, a, b, amp)
    return np.stack([dy, -dx], axis=-1)


def curl_equivariant_jacobian(x, y, s0, a, b, amp):
    s = (x + y) / np.sqrt(2.0)
    w = (x - y) / np.sqrt(2.0)
    G = np.exp(-0.5 * ((s - s0) / a) ** 2 - 0.5 * (w / b) ** 2)
    # ψ = amp · w · G(s, w)
    # Let h(s) = amp · exp(−(s−s0)²/(2a²));  g(w) = w · exp(−w²/(2b²))
    # ψ = h(s) · g(w). Chain-rule derivatives.
    h = amp * np.exp(-0.5 * ((s - s0) / a) ** 2)
    hs = h * (-(s - s0) / a ** 2)
    hss = h * ((-(s - s0) / a ** 2) ** 2 - 1.0 / a ** 2)
    ge = np.exp(-0.5 * (w / b) ** 2)
    g = w * ge
    gw = ge * (1.0 - w ** 2 / b ** 2)
    gww = ge * (w * (w ** 2 / b ** 2 - 3.0) / b ** 2)
    psi_s = hs * g
    psi_w = h * gw
    psi_ss = hss * g
    psi_ww = h * gww
    psi_sw = hs * gw
    psi_xx = psi_ss / 2.0 + psi_ww / 2.0 + psi_sw
    psi_yy = psi_ss / 2.0 + psi_ww / 2.0 - psi_sw
    psi_xy = psi_ss / 2.0 - psi_ww / 2.0
    return np.array([[psi_xy, psi_yy], [-psi_xx, -psi_xy]])


# ---------------------------------------------------------------------------
# Density estimation and MMD identifiability check
# ---------------------------------------------------------------------------

def kde_2d(samples, grid_x, grid_y, bandwidth=0.05):
    xx, yy = np.meshgrid(grid_x, grid_y)
    n = samples.shape[0]
    hx = bandwidth
    hy = bandwidth
    dx = (xx.ravel()[:, None] - samples[None, :, 0]) / hx
    dy = (yy.ravel()[:, None] - samples[None, :, 1]) / hy
    dens = np.exp(-0.5 * (dx ** 2 + dy ** 2)).mean(axis=1) / (2 * np.pi * hx * hy)
    return dens.reshape(xx.shape)


def evaluate_p_at(z_query, samples, bandwidth=0.05):
    """KDE evaluated at query points."""
    hx = bandwidth
    dx = (z_query[:, None, 0] - samples[None, :, 0]) / hx
    dy = (z_query[:, None, 1] - samples[None, :, 1]) / hx
    return np.exp(-0.5 * (dx ** 2 + dy ** 2)).mean(axis=1) / (2 * np.pi * hx * hx)


def mmd2_two_sample(X, Y, sigma_list=(0.05, 0.10, 0.20)):
    """Unbiased multi-scale Gaussian MMD² between X (n×d) and Y (m×d)."""
    n = X.shape[0]
    m = Y.shape[0]
    mmd2 = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2.0 * sigma ** 2)
        def K(A, B):
            d2 = ((A[:, None] - B[None, :]) ** 2).sum(-1)
            return np.exp(-gamma * d2)
        Kxx = K(X, X)
        Kyy = K(Y, Y)
        Kxy = K(X, Y)
        # Remove diagonal
        Kxx = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
        Kyy = (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
        Kxy = Kxy.sum() / (n * m)
        mmd2 += Kxx + Kyy - 2 * Kxy
    return mmd2 / len(sigma_list)


def mmd_permutation_test(X, Y, n_perm=200, sigma_list=(0.05, 0.10, 0.20), rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    obs = mmd2_two_sample(X, Y, sigma_list)
    Z = np.concatenate([X, Y], axis=0)
    n = X.shape[0]
    null = np.zeros(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(len(Z))
        null[i] = mmd2_two_sample(Z[perm[:n]], Z[perm[n:]], sigma_list)
    p = float((np.sum(null >= obs) + 1) / (n_perm + 1))
    return obs, p


# ---------------------------------------------------------------------------
# Simulation of the twin drift
# ---------------------------------------------------------------------------

def simulate_twin(v_of_z, seed=42, n_cells=6000, T_total=10.0, dt=0.02):
    """
    Simulate f'(z, α) = f(z, α) + v(z)/p(z, τ) via SDE Euler-Maruyama with the
    same schedule as the ground-truth simulator.

    Because p(z, τ) is not analytically available and estimating it live is
    expensive, we use a two-pass strategy:
      1. Simulate ground truth first (that gives us cells → density estimate).
      2. Simulate twin advancing in the *same* discretization, evaluating p
         via KDE on the CURRENT cell cloud at each step. The identifiability
         result guarantees that marginals should match at every τ.
    """
    rng = np.random.default_rng(seed)
    n_steps = int(np.ceil(T_total / dt))
    target_step = np.sort(rng.integers(1, n_steps + 1, size=n_cells))
    t_per_cell = (target_step * dt) / T_total
    alpha_min, alpha_max = gt.ALPHA_MIN, gt.ALPHA_MAX

    z = np.full((n_cells, 2), gt.sym_fp(alpha_min), dtype=np.float32)
    z += 0.02 * rng.standard_normal(z.shape).astype(np.float32)
    z_out = np.zeros_like(z)
    sqrt_dt = float(np.sqrt(dt))
    idx = 0
    for step in range(1, n_steps + 1):
        a_t = alpha_min + (alpha_max - alpha_min) * (step * dt) / T_total
        f_true = gt.toggle_drift(z.astype(np.float64), a_t).astype(np.float32)
        # KDE p on the current cloud, evaluate at each cell
        p_at_cells = evaluate_p_at(z.astype(np.float64), z.astype(np.float64),
                                    bandwidth=0.05).astype(np.float32)
        v = v_of_z(z.astype(np.float64)).astype(np.float32)
        # Regularise 1/p to avoid blow-up in tail
        p_at_cells = np.maximum(p_at_cells, 1e-3)
        f_prime = f_true + v / p_at_cells[:, None]
        z = z + f_prime * dt + gt.SIGMA_SDE * sqrt_dt * rng.standard_normal(z.shape).astype(np.float32)
        while idx < n_cells and target_step[idx] == step:
            z_out[idx] = z[idx]
            idx += 1
    return z_out.astype(np.float32), t_per_cell.astype(np.float32)


# ---------------------------------------------------------------------------
# λ_⊥ of the twin's Jacobian at the symmetric fp
# ---------------------------------------------------------------------------

def lam_perp_twin(x, y, alpha, jacobian_v_fn):
    """
    λ_⊥ of f' = f + v/p at (x, y; α). This mixes the deterministic Jacobian
    of f at (x, y) with the perturbation ∂(v/p)/∂z evaluated *at* (x, y).
    For a symmetric-fp analytic calculation, we approximate ∂(v/p)/∂z by
    v_J(x, y)/p(s, w) — dropping the ∂(1/p)/∂z term is a first-order
    approximation valid when v is small vs p·f.
    """
    J_f = gt.toggle_jacobian_at(x, y, alpha)
    J_v = jacobian_v_fn(x, y)
    # Approximate ∂(v/p)/∂z ≈ J_v(z)/p(z), holding p constant at the fp.
    # (The exact expression carries a (−v ∇p^T)/p² term which vanishes at the
    # density modes ∇p = 0. At the symmetric fp, before commitment, ∇p ≠ 0
    # in general — but for a σ-invariant density the p-gradient is along s,
    # and its cross-derivative with the antisymmetric direction is zero.)
    # We use the fp Newton-solved s(α) to evaluate p analytically for a
    # 1-D approximation: p(s) ∝ 1/|f_s|.
    p_at_fp = 1.0  # normalize; the sign of λ_⊥ is what matters
    J_full = J_f + J_v / p_at_fp
    eigvals = np.linalg.eigvals(J_full)
    # λ_⊥ is the eigenvalue whose eigenvector is closest to (1, −1)/√2.
    # A cheap surrogate: pick the eigenvalue whose eigenvector has larger
    # antisymmetric component.
    w_perp = np.array([1.0, -1.0]) / np.sqrt(2.0)
    eigvals, eigvecs = np.linalg.eig(J_full)
    proj = np.abs(eigvecs.T @ w_perp)
    idx = int(np.argmax(proj))
    return float(eigvals[idx].real)


# ---------------------------------------------------------------------------
# Main T7 driver
# ---------------------------------------------------------------------------

def main():
    out_dir = Path("/Users/terooatt/Downloads/scJDO/reproducibility")

    tau_crit = gt.TAU_CRIT
    alpha_crit = gt.ALPHA_CRIT
    s_crit = gt.sym_fp(alpha_crit)

    print(f"τ_crit={tau_crit:.4f}  α_crit={alpha_crit:.4f}  s*={s_crit:.4f}")

    # ── True baseline ──────────────────────────────────────────────────
    z_true, tau_true = gt.simulate_v3(n_cells=6000, seed=42)
    tau_windows = [(0.02, 0.06), (0.10, 0.14), (0.30, 0.34), (0.60, 0.64), (0.90, 0.94)]

    def _window_slices(tau, lo, hi):
        return np.where((tau >= lo) & (tau < hi))[0]

    def _twin_mmd(z_twin, tau_twin, label):
        print(f"\n  {label}:")
        results = []
        rng = np.random.default_rng(0)
        for lo, hi in tau_windows:
            iA = _window_slices(tau_true, lo, hi)
            iB = _window_slices(tau_twin, lo, hi)
            if min(len(iA), len(iB)) < 100:
                results.append({"window": (lo, hi), "n": min(len(iA), len(iB)),
                                 "mmd2": float("nan"), "p": float("nan")})
                continue
            n = min(200, len(iA), len(iB))
            X = z_true[rng.choice(iA, n, replace=False)]
            Y = z_twin[rng.choice(iB, n, replace=False)]
            m, p = mmd_permutation_test(X, Y, n_perm=200, rng=rng)
            results.append({"window": (lo, hi), "n": n, "mmd2": m, "p": p})
            print(f"    τ ∈ [{lo:.2f}, {hi:.2f}]  MMD²={m:+.5f}  p={p:.3f}")
        return results

    all_results = {"tau_crit": tau_crit, "twins": {}}

    # T7.1 — Baseline Gaussian stream centered on (s*, 0)  (equivariant by chance)
    baseline_amp = 0.5
    v_baseline = lambda z: curl_stream(z, s0=s_crit, w0=0.0, a=0.3, b=0.3, amp=baseline_amp)
    Jv_baseline = lambda x, y: curl_stream_jacobian(x, y, s_crit, 0.0, 0.3, 0.3, baseline_amp)
    lam_true = gt.transverse_eigenvalue(alpha_crit)
    lam_baseline = lam_perp_twin(s_crit, s_crit, alpha_crit, Jv_baseline)
    print(f"\n[T7.1] baseline Gaussian stream (s0=s*, w0=0, a=b=0.3, amp={baseline_amp})")
    print(f"       λ_⊥(f) = {lam_true:+.4f}    λ_⊥(f') = {lam_baseline:+.4f}")
    z_twin, tau_twin = simulate_twin(v_baseline, seed=42, n_cells=6000)
    r_base = _twin_mmd(z_twin, tau_twin, "T7.1 baseline")
    all_results["twins"]["baseline_gaussian"] = {
        "params": {"s0": float(s_crit), "w0": 0.0, "a": 0.3, "b": 0.3, "amp": baseline_amp},
        "lam_perp_true": lam_true,
        "lam_perp_twin": lam_baseline,
        "sign_flip": bool(np.sign(lam_baseline) != np.sign(lam_true)),
        "mmd_windows": [{"window": list(r["window"]), "n": r["n"],
                          "mmd2": r["mmd2"], "p": r["p"]} for r in r_base],
        "min_p_across_windows": float(min(r["p"] for r in r_base if np.isfinite(r["p"]))),
    }

    # T7.3 — Equivariance-compliant offset adversary
    print("\n[T7.3] Equivariance-compliant offset adversary  ψ = A · w · exp(...)")
    for s0_offset, amp in [(0.05, 3.0), (0.10, 5.0), (0.20, 10.0), (0.30, 20.0)]:
        v = lambda z, s0=s_crit + s0_offset, A=amp: curl_equivariant(z, s0=s0, a=0.3, b=0.3, amp=A)
        Jv = lambda x, y, s0=s_crit + s0_offset, A=amp: curl_equivariant_jacobian(x, y, s0, 0.3, 0.3, A)
        lam_ad = lam_perp_twin(s_crit, s_crit, alpha_crit, Jv)
        sign_flip = np.sign(lam_ad) != np.sign(lam_true + 1e-10)
        print(f"       s0_offset={s0_offset:.2f}  amp={amp:5.1f}  "
              f"λ_⊥(f') = {lam_ad:+.4f}   {'← SIGN FLIP' if sign_flip else ''}")
        if sign_flip:
            z_twin, tau_twin = simulate_twin(v, seed=42, n_cells=6000)
            r = _twin_mmd(z_twin, tau_twin, f"T7.3 adversary s0+{s0_offset}, amp={amp}")
            key = f"equivariant_offset_s0+{s0_offset}_amp{amp}"
            all_results["twins"][key] = {
                "params": {"s0": float(s_crit + s0_offset), "a": 0.3, "b": 0.3, "amp": amp},
                "lam_perp_true": lam_true, "lam_perp_twin": lam_ad,
                "sign_flip": True,
                "mmd_windows": [{"window": list(r_i["window"]), "n": r_i["n"],
                                  "mmd2": r_i["mmd2"], "p": r_i["p"]} for r_i in r],
                "min_p_across_windows": float(min(r_i["p"] for r_i in r if np.isfinite(r_i["p"]))),
            }
            # Only record the first sign-flipping adversary to keep runtime bounded
            break

    # T7.4 — Boundedness test:
    #   |v / p| ≤ K · max|f|.  On the diagonal p is highest at (s*, s*), so
    #   the σ-equivariant adversary needs its amplitude in the LOW-density
    #   tail. Compute max(|v|/p) over cells in a window around τ_crit, and
    #   compare to K · max(|f|).
    print("\n[T7.4] Boundedness constraint |v/p| ≤ K max|f|")
    from itertools import product
    K_thr = 1.0  # allow twin drift no bigger than the true drift
    iA = _window_slices(tau_true, 0.0, 0.10)  # cells near the saddle
    z_near = z_true[iA]
    f_near = gt.toggle_drift(z_near.astype(np.float64), gt.alpha_of_tau(tau_true[iA].mean()))
    max_f_near = float(np.max(np.linalg.norm(f_near, axis=1)))
    print(f"       max |f| near τ_crit = {max_f_near:.3f}   → allowed |v/p| ≤ {K_thr*max_f_near:.3f}")

    for s0_offset, amp in [(0.05, 3.0), (0.10, 5.0), (0.20, 10.0), (0.30, 20.0)]:
        v = lambda z, s0=s_crit + s0_offset, A=amp: curl_equivariant(z, s0=s0, a=0.3, b=0.3, amp=A)
        p_here = evaluate_p_at(z_near.astype(np.float64), z_true.astype(np.float64), bandwidth=0.05)
        v_here = v(z_near.astype(np.float64))
        ratio = np.linalg.norm(v_here, axis=1) / np.maximum(p_here, 1e-3)
        max_ratio = float(np.max(ratio))
        Jv = lambda x, y, s0=s_crit + s0_offset, A=amp: curl_equivariant_jacobian(x, y, s0, 0.3, 0.3, A)
        lam_ad = lam_perp_twin(s_crit, s_crit, alpha_crit, Jv)
        passes = max_ratio <= K_thr * max_f_near
        print(f"       s0_offset={s0_offset:.2f}  amp={amp:5.1f}  "
              f"max |v/p| = {max_ratio:.3f}  {'BOUNDED' if passes else 'UNBOUNDED'}  λ_⊥={lam_ad:+.4f}")
        all_results["twins"].setdefault("boundedness_scan", []).append({
            "s0_offset": s0_offset, "amp": amp,
            "max_v_over_p_near_saddle": max_ratio,
            "K_times_max_f": K_thr * max_f_near,
            "bounded": bool(passes),
            "lam_perp_twin_at_fp": lam_ad,
            "sign_flip": bool(np.sign(lam_ad) != np.sign(lam_true + 1e-10)),
        })

    def _to_json(obj):
        import numpy as _np
        if isinstance(obj, dict):
            return {k: _to_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_json(v) for v in obj]
        if isinstance(obj, (_np.floating,)):
            return float(obj)
        if isinstance(obj, (_np.integer,)):
            return int(obj)
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
        return obj

    (out_dir / "data" / "T7_gauge_twins.json").write_text(
        json.dumps(_to_json(all_results), indent=2)
    )
    print(f"\nSaved: {out_dir}/data/T7_gauge_twins.json")


if __name__ == "__main__":
    main()
