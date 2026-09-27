"""
FOLLOWUP5 (#2, #3) — F-hypothesis decisive tests.

Two experiments, both requiring a bit-reproducible fit. RUN ONLY AFTER
the determinism fix in FOLLOWUP5_determinism.py lands and standalone
== multi-seed output is verified.

(#2) cos(f_train, ∇pseudotime)
     If the trained drift encodes ∇pseudotime rather than the SDE
     drift, this cosine should be ≈ +1. `_pseudotime_velocity` in
     scjdo/tl/_drift.py:19-38 constructs V exactly as fit_drift feeds
     it in as prior — we reuse that.

(#3) Ablate the velocity prior
     Set vel_scale = 0 in DriftConfig. Refit and re-run the sign test
     and the cos test.

Both should be quick (~40 s per fit).
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

import toggle_truth as gt
from FOLLOWUP2_trained_probe import _build_adata
from T2_lifted_truth import kernel_matrix_curve
from crossing_robust import robust_crossing


def apply_determinism_fix():
    """Fix level F1 — call this at process start."""
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as e:
        print(f"(fix note) use_deterministic_algorithms failed: {e}")
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def fit_and_probe(seed, vel_scale=None):
    """Fit fresh, return trained-drift outputs plus the pseudotime-gradient
    velocity vector V that fit_drift built.  If vel_scale is provided,
    pass it explicitly (0 = ablate the prior)."""
    from scjdo.tl._drift import _pseudotime_velocity
    from scjdo.tl import fit_drift

    adata, Z_lat, tau = _build_adata(seed=seed)
    kwargs = dict(rep="X_pca", time_key="pseudotime", n_epochs=800,
                  n_archetypes=4, n_eff_min=20.0, n_boot=10, grid_size=200,
                  seed=seed, verbose=False)
    if vel_scale is not None:
        kwargs["vel_scale"] = float(vel_scale)
    t0 = time.time()
    model = fit_drift(adata, **kwargs)
    dt = time.time() - t0

    device = next(model.parameters()).device
    z_t = torch.tensor(Z_lat, dtype=torch.float32, device=device)
    t_t = torch.tensor(tau, dtype=torch.float32, device=device)
    with torch.no_grad():
        f_train = model(z_t, t_t).detach().cpu().numpy()

    # Rebuild the analytic lifted drift for comparison.
    W = gt.build_observation_map(seed=seed)
    # We need the exact (μ, V) used inside _build_adata. Re-derive.
    from sklearn.decomposition import PCA
    rng_g = np.random.default_rng(seed)
    X_obs = z_true_from_adata = None
    # Instead of reconstructing, take the pca-fit that _build_adata used
    # (which is discarded — but we can re-derive since the same seed gives
    # the same simulate_v3 output and the same W). Note this depends on
    # numpy's global RNG not being consumed between the calls.
    z2, tau2 = gt.simulate_v3(n_cells=len(tau), seed=seed)
    assert np.allclose(tau, tau2)
    rng_g = np.random.default_rng(seed)
    X_obs = z2 @ W + gt.NOISE_SIGMA * rng_g.standard_normal((z2.shape[0], W.shape[1])).astype(np.float32)
    pca = PCA(n_components=20, random_state=seed).fit(X_obs)
    V_pca = pca.components_.astype(np.float32)   # (20, 200)
    # Lifted analytic drift  f_lifted = f_2 @ (W @ V^T)
    alphas = gt.alpha_of_tau(tau)
    f2 = gt.toggle_drift(z2.astype(np.float64), alphas)
    A_lift = (W @ V_pca.T).astype(np.float64)   # (2, 20)
    f_lifted = (f2 @ A_lift).astype(np.float32)

    # Pseudotime-gradient velocity — exactly what fit_drift builds
    V_pt = _pseudotime_velocity(Z_lat, tau, k=15)   # (N, 20), unit-mean magnitude

    def _cos_matrix(a, b):
        na = np.linalg.norm(a, axis=1) + 1e-12
        nb = np.linalg.norm(b, axis=1) + 1e-12
        return (a * b).sum(1) / (na * nb)

    cos_train_vs_lifted = _cos_matrix(f_train, f_lifted)
    cos_train_vs_pt     = _cos_matrix(f_train, V_pt)
    cos_lifted_vs_pt    = _cos_matrix(f_lifted, V_pt)

    # Aggregate onto a τ-grid
    grid = np.linspace(0.0, 1.0, 400)

    def _kc(x):
        out = np.full(grid.size, np.nan)
        for k, tc in enumerate(grid):
            w = np.exp(-0.5 * ((tau - tc) / 0.02) ** 2)
            W_ = w.sum()
            if W_ < 1e-9: continue
            if W_ ** 2 / (w * w).sum() < 15.0: continue
            out[k] = float((w * x).sum() / W_)
        return out

    curve_cos_train_lifted = _kc(cos_train_vs_lifted)
    curve_cos_train_pt = _kc(cos_train_vs_pt)
    curve_cos_lifted_pt = _kc(cos_lifted_vs_pt)

    # Compute per-cell Jacobians to get λ
    from FOLLOWUP2_trained_probe import _per_cell_jacobians
    J_pc = _per_cell_jacobians(model, Z_lat, tau)
    Jbar, _ = kernel_matrix_curve(J_pc, tau, grid, bandwidth=0.02, n_eff_min=15.0)
    lam = np.array([float(np.real(np.linalg.eigvals(J)).max()) if not np.isnan(J).any() else np.nan for J in Jbar])
    tau_hat, code = robust_crossing(lam, grid, threshold=0.0, min_run=5)

    return {
        "seed": seed, "vel_scale": vel_scale, "fit_time_s": dt,
        "r2": float(adata.uns["scjdo"]["r2"]),
        "bandwidth_auto": adata.uns["scjdo"].get("bandwidth"),
        "cos_train_vs_lifted_curve": curve_cos_train_lifted,
        "cos_train_vs_pt_curve": curve_cos_train_pt,
        "cos_lifted_vs_pt_curve": curve_cos_lifted_pt,
        "cos_train_vs_lifted_mean": float(np.nanmean(cos_train_vs_lifted)),
        "cos_train_vs_pt_mean": float(np.nanmean(cos_train_vs_pt)),
        "cos_lifted_vs_pt_mean": float(np.nanmean(cos_lifted_vs_pt)),
        "min_lam": float(np.nanmin(lam)),
        "max_lam": float(np.nanmax(lam)),
        "tau_hat_cross": tau_hat, "cross_code": code,
        "grid": grid,
    }


def main():
    out_dir = REPO / "reproducibility"
    apply_determinism_fix()

    all_results = {"tau_crit": gt.TAU_CRIT}
    for vel_scale in [None, 0.0]:   # None = default (2.0); 0.0 = ablate
        label = "default (vel_scale=2.0)" if vel_scale is None else "ABLATED (vel_scale=0.0)"
        print(f"\n== {label} ==")
        results_list = []
        for seed in [42, 0, 1]:
            r = fit_and_probe(seed, vel_scale=vel_scale)
            results_list.append(r)
            print(f"  seed={seed}  cos(train, lifted)={r['cos_train_vs_lifted_mean']:+.3f}  "
                  f"cos(train, ∇τ)={r['cos_train_vs_pt_mean']:+.3f}  "
                  f"cos(lifted, ∇τ)={r['cos_lifted_vs_pt_mean']:+.3f}  "
                  f"min λ={r['min_lam']:+.4f}  code={r['cross_code']}  τ̂={r['tau_hat_cross']}")
        all_results[label] = results_list

    def _san(x):
        if isinstance(x, dict): return {k: _san(v) for k, v in x.items()}
        if isinstance(x, list): return [_san(v) for v in x]
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.floating): return float(x)
        if isinstance(x, np.integer): return int(x)
        return x
    (out_dir / "data" / "FOLLOWUP5_F_tests.json").write_text(
        json.dumps(_san(all_results), indent=2)
    )
    print(f"\nSaved: {out_dir}/data/FOLLOWUP5_F_tests.json")


if __name__ == "__main__":
    main()
