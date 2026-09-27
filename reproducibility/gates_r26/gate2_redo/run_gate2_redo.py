"""Gate 2 redo — velocity-matching-loss L arm + expression-only baseline.
Per PREREG_Gate2_redo.md."""
from __future__ import annotations
import json, sys, warnings, time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "reproducibility" / "gates_r26" / "gate2"))
warnings.filterwarnings("ignore")

from run_gate2 import (
    load_or_download, run_dynamo_reference, compute_pseudotime,
    build_kernel_operator, leading_eig,
    fit_scjdo_arm,  # unchanged G arm helper (V_ref=None uses monkey-patch inert)
    compute_per_cell_velocity_from_J, cos_per_row,
)
from scipy import stats

OUT = REPO / "reproducibility" / "gates_r26" / "gate2_redo"
DATA = OUT / "data"; DATA.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2]
GRID_SIZE = 150
N_ARCH = 5
N_EPOCHS = 3000
D_REP = 30
LAMBDA_MATCH = 1.0  # frozen per prereg


def fit_drift_vmatch(X_pca, tau, seed, V_ref_target, lambda_match=LAMBDA_MATCH,
                     n_epochs=N_EPOCHS, grid_size=GRID_SIZE, n_archetypes=N_ARCH):
    """Duplicate fit_drift's training loop with an added cosine-matching
    loss L_match = mean(1 - cos(model(x_i, t_i), V_ref_target[i])).
    Returns the same dict shape as fit_scjdo_arm."""
    from scjdo.models.drift import DriftField, DriftConfig
    from scjdo.losses import denoising_score_matching, control_energy, local_sigma

    torch.manual_seed(seed); np.random.seed(seed)
    device = "cpu"

    X = torch.tensor(X_pca.astype(np.float32), device=device)
    T = torch.tensor(tau.astype(np.float32), device=device)
    V = torch.tensor(V_ref_target.astype(np.float32), device=device)
    N, D = X.shape

    cfg = DriftConfig(dim=D, hidden=256, depth=4, beta=0.1,
                       use_spectral_norm=True, vel_scale=0.0,
                       vel_time_mode="flat")
    model = DriftField(cfg, X_ref=X, V_ref=V).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    sigma_per_cell = local_sigma(X).cpu()

    losses_total = []; losses_dsm = []; losses_match = []
    batch_size = 512
    model.train()
    for step in range(n_epochs):
        idx = torch.randint(0, N, (batch_size,), device=device)
        xb, tb, vb = X[idx], T[idx], V[idx]
        sig = sigma_per_cell[idx.cpu()].to(device)

        l_dsm = denoising_score_matching(model, xb, tb, sigma=sig)
        v_pred = model(xb, tb)
        l_ctrl = cfg.alpha_control * control_energy(v_pred)
        v_pred_n = v_pred / (v_pred.norm(dim=1, keepdim=True) + 1e-8)
        vb_n = vb / (vb.norm(dim=1, keepdim=True) + 1e-8)
        cos_batch = (v_pred_n * vb_n).sum(dim=1)  # (batch,)
        l_match = (1.0 - cos_batch).mean()
        loss = l_dsm + l_ctrl + lambda_match * l_match

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        losses_total.append(float(loss.item()))
        losses_dsm.append(float(l_dsm.item()))
        losses_match.append(float(l_match.item()))

    model.eval()

    # Compute per-cell Jacobians and kernel-aggregated tensor same way fit_drift does
    with torch.no_grad():
        # per-cell Jacobians via functorch
        from torch.func import jacrev, vmap
        def drift_fn(x_i, t_i):
            return model(x_i.unsqueeze(0), t_i.unsqueeze(0)).squeeze(0)
        J_per_cell = vmap(jacrev(drift_fn), in_dims=(0, 0))(X, T)  # (N, D, D)
    J_np = J_per_cell.cpu().numpy().astype(np.float32)

    # kernel windowing along tau to yield (grid_size, D, D)
    T_np = tau.astype(np.float32)
    grid = np.linspace(T_np.min(), T_np.max(), grid_size).astype(np.float32)
    # simple gaussian kernel with adaptive bandwidth: use a small default h
    # so this doesn't require the full build_temporal_operator machinery.
    # Match Gate 2 v1's kernel: h = 0.1 (dataset's typical bandwidth)
    h = 0.1
    n_grid = len(grid)
    Jt = np.zeros((n_grid, D, D), dtype=np.float32)
    max_re = np.zeros(n_grid, dtype=np.float32)
    for i, tc in enumerate(grid):
        w = np.exp(-((T_np - tc) ** 2) / (2.0 * h ** 2))
        W = w.sum()
        if W < 1e-9: continue
        Jt[i] = ((w[:, None, None] * J_np).sum(0) / W).astype(np.float32)
        ev = np.linalg.eigvals(Jt[i].astype(np.float64))
        max_re[i] = float(np.real(ev).max())
    return {"J_tensor": Jt, "t_centers": grid, "max_real_eig": max_re,
            "bandwidth": h,
            "loss_curve": {"total_final": losses_total[-1],
                            "dsm_final": losses_dsm[-1],
                            "match_final": losses_match[-1]}}


def per_cell_vel(arm, X_pca, tau):
    return compute_per_cell_velocity_from_J(arm["J_tensor"], arm["t_centers"],
                                              X_pca, tau)


def main():
    print("=" * 80); print("Gate 2 redo"); print("=" * 80)
    t_all = time.time()

    # Load + build R (reuse gate2/data/neuron_labeling.h5ad if present)
    print("[R] Dynamo reference (reuse Gate 2 pipeline)")
    src_ck = REPO / "reproducibility" / "gates_r26" / "gate2" / "data" / "neuron_labeling.h5ad"
    if src_ck.exists():
        import anndata as ad
        adata = ad.read_h5ad(src_ck)
    else:
        adata = load_or_download()
    R = run_dynamo_reference(adata)
    X_pca = R["X_pca"]
    tau = compute_pseudotime(adata)
    print(f"  X_pca={X_pca.shape}  tau=[{tau.min():.3f}, {tau.max():.3f}]  "
          f"|v_ref| median = {float(np.median(np.linalg.norm(R['v_ref'], axis=1))):.3f}")

    # G — bandwidth pilot
    print("\n[G] scJDO seed=0 (for bandwidth h*)")
    G0 = fit_scjdo_arm(X_pca, tau, seed=0, V_ref=None, vel_scale=0.0)
    bw = G0["bandwidth"]; t_centers = G0["t_centers"]
    print(f"  bandwidth h*={bw:.4f}, t_centers={t_centers.shape}")

    J_R = build_kernel_operator(X_pca, tau, t_centers, bw, V_per_cell=R["v_ref"])
    R_lead_re, R_lead_vec = leading_eig(J_R)
    print(f"  R Re(lambda_max) range=[{R_lead_re.min():.3f}, {R_lead_re.max():.3f}]")

    # G — three seeds (seed=0 already done)
    print("\n[G] geometry-only scJDO (vel_scale=0), 3 seeds")
    G_arms = {0: G0}
    for seed in SEEDS[1:]:
        t0 = time.time()
        G_arms[seed] = fit_scjdo_arm(X_pca, tau, seed=seed, V_ref=None, vel_scale=0.0)
        print(f"  seed {seed} done in {time.time()-t0:.1f}s")

    # L — three seeds with velocity-matching loss
    print(f"\n[L] velocity-matching-loss scJDO (lambda={LAMBDA_MATCH}), 3 seeds")
    L_arms = {}
    for seed in SEEDS:
        t0 = time.time()
        L_arms[seed] = fit_drift_vmatch(X_pca, tau, seed=seed, V_ref_target=R["v_ref"])
        print(f"  seed {seed} done in {time.time()-t0:.1f}s  "
              f"loss dsm={L_arms[seed]['loss_curve']['dsm_final']:.4f} "
              f"match={L_arms[seed]['loss_curve']['match_final']:.4f}")

    # Baseline_E — LinearRegression via 5-fold KFold
    print("\n[Baseline_E] LinearRegression X_pca -> v_ref, 5-fold KFold")
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    v_baseline = np.zeros_like(R["v_ref"])
    for tr, te in kf.split(X_pca):
        lr = LinearRegression()
        lr.fit(X_pca[tr], R["v_ref"][tr])
        v_baseline[te] = lr.predict(X_pca[te])
    baseline_cos = cos_per_row(v_baseline, R["v_ref"])
    print(f"  baseline_E mean cos(v_baseline, R) = {float(np.nanmean(baseline_cos)):+.4f}")

    # ── P1 — data-determined ──────────────────────────────────────────
    print("\n[P1] data-determined output alignment")
    p1 = {}
    for label, arms in [("G", G_arms), ("L", L_arms)]:
        cos_lead = []; spearman_re = []
        for seed in SEEDS:
            arm = arms[seed]
            l_re, l_vec = leading_eig(arm["J_tensor"])
            c = np.abs((l_vec * R_lead_vec).sum(axis=1)) / (
                np.linalg.norm(l_vec, axis=1) * np.linalg.norm(R_lead_vec, axis=1) + 1e-12
            )
            cos_lead.append(float(np.median(c)))
            spearman_re.append(float(stats.spearmanr(l_re, R_lead_re).correlation))
        p1[label] = {
            "median_cos_leading_eigvec_per_seed": cos_lead,
            "spearman_re_lmax_per_seed": spearman_re,
            "median_cos_mean": float(np.mean(cos_lead)),
            "spearman_re_mean": float(np.nanmean(spearman_re)),
        }
        print(f"  {label}: median cos(leading eigvec, R) = {np.mean(cos_lead):.3f}, "
              f"Spearman(Re lam_max, R) = {np.nanmean(spearman_re):.3f}")
    p1_pass = (p1["G"]["median_cos_mean"] >= 0.5 and p1["G"]["spearman_re_mean"] >= 0.5)
    print(f"  P1 pass? {p1_pass}")

    # ── P2 — prior-determined per-cell velocity ─────────────────────────
    print("\n[P2] prior-determined per-cell velocity cos(v_arm, R)")
    cos_G = np.stack([cos_per_row(per_cell_vel(G_arms[s], X_pca, tau), R["v_ref"])
                       for s in SEEDS], axis=1)  # (N, 3)
    cos_L = np.stack([cos_per_row(per_cell_vel(L_arms[s], X_pca, tau), R["v_ref"])
                       for s in SEEDS], axis=1)
    delta_per_cell = cos_L.mean(axis=1) - cos_G.mean(axis=1)
    rng = np.random.default_rng(0); N = len(delta_per_cell)
    boot = [float(np.nanmean(delta_per_cell[rng.integers(0, N, size=N)])) for _ in range(2000)]
    ci_p2 = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    p2_pass = (float(np.nanmean(cos_L)) > float(np.nanmean(cos_G))) and (ci_p2[0] > 0)
    print(f"  mean cos(vel_G, R) = {float(np.nanmean(cos_G)):+.4f}")
    print(f"  mean cos(vel_L, R) = {float(np.nanmean(cos_L)):+.4f}")
    print(f"  Δ (L-G) per-cell mean = {float(np.nanmean(delta_per_cell)):+.4f} "
          f"95%CI [{ci_p2[0]:+.4f}, {ci_p2[1]:+.4f}]")
    print(f"  P2 pass? {p2_pass}")

    # ── P2' — L vs Baseline_E ────────────────────────────────────────────
    delta_p2p = cos_L.mean(axis=1) - baseline_cos
    boot2 = [float(np.nanmean(delta_p2p[rng.integers(0, N, size=N)])) for _ in range(2000)]
    ci_p2p = (float(np.percentile(boot2, 2.5)), float(np.percentile(boot2, 97.5)))
    print(f"  Δ (L - Baseline_E) per-cell mean = {float(np.nanmean(delta_p2p)):+.4f} "
          f"95%CI [{ci_p2p[0]:+.4f}, {ci_p2p[1]:+.4f}]")

    # ── P3 — seed stability ─────────────────────────────────────────────
    print("\n[P3] seed stability of per-cell drift")
    vG_stack = np.stack([per_cell_vel(G_arms[s], X_pca, tau) for s in SEEDS], axis=0)
    vL_stack = np.stack([per_cell_vel(L_arms[s], X_pca, tau) for s in SEEDS], axis=0)
    var_G = ((vG_stack - vG_stack.mean(axis=0, keepdims=True)) ** 2).sum(axis=-1).mean(axis=0)
    var_L = ((vL_stack - vL_stack.mean(axis=0, keepdims=True)) ** 2).sum(axis=-1).mean(axis=0)
    ratio = var_L / (var_G + 1e-12)
    boot3 = [float(np.nanmedian(ratio[rng.integers(0, N, size=N)])) for _ in range(2000)]
    ci_p3 = (float(np.percentile(boot3, 2.5)), float(np.percentile(boot3, 97.5)))
    p3_pass = (float(np.nanmedian(ratio)) < 1.0) and (ci_p3[1] < 1.0)
    print(f"  var_G mean={float(np.nanmean(var_G)):.4f}  var_L mean={float(np.nanmean(var_L)):.4f}")
    print(f"  median var_L/var_G = {float(np.nanmedian(ratio)):.4f}  "
          f"95%CI [{ci_p3[0]:.4f}, {ci_p3[1]:.4f}]")
    print(f"  P3 pass? {p3_pass}")

    verdict = ("PASS Gate 2 redo" if (p1_pass and p2_pass and p3_pass)
                else "FAIL Gate 2 redo")
    print(f"\nVerdict: {verdict}  (P1={p1_pass}, P2={p2_pass}, P3={p3_pass})")

    def _tn(v):
        if isinstance(v, (np.floating,)): return float(v)
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, (list, tuple)): return [_tn(x) for x in v]
        if isinstance(v, dict): return {k: _tn(x) for k, x in v.items()}
        return v
    (OUT / "gate2_redo_summary.json").write_text(json.dumps(_tn({
        "shape": list(X_pca.shape), "bandwidth": bw,
        "lambda_match": LAMBDA_MATCH,
        "P1": p1, "P1_pass": bool(p1_pass),
        "P2": {"mean_cos_vG_R": float(np.nanmean(cos_G)),
                "mean_cos_vL_R": float(np.nanmean(cos_L)),
                "delta_mean_per_cell": float(np.nanmean(delta_per_cell)),
                "delta_95ci": ci_p2, "pass": bool(p2_pass)},
        "P2p": {"mean_cos_baseline_E_R": float(np.nanmean(baseline_cos)),
                 "delta_L_minus_baselineE_mean": float(np.nanmean(delta_p2p)),
                 "delta_95ci": ci_p2p},
        "P3": {"var_G_mean": float(np.nanmean(var_G)),
                "var_L_mean": float(np.nanmean(var_L)),
                "median_var_ratio": float(np.nanmedian(ratio)),
                "median_var_ratio_95ci": ci_p3, "pass": bool(p3_pass)},
        "L_loss_curve_finals": {s: L_arms[s]["loss_curve"] for s in SEEDS},
        "verdict": verdict,
    }), indent=2))
    np.savez(OUT / "gate2_redo_arms.npz",
              J_R=J_R, R_v_ref=R["v_ref"], X_pca=X_pca, tau=tau,
              t_centers=t_centers,
              baseline_v=v_baseline,
              **{f"G_seed{s}_J": G_arms[s]["J_tensor"] for s in SEEDS},
              **{f"L_seed{s}_J": L_arms[s]["J_tensor"] for s in SEEDS})
    print(f"\n[written] gate2_redo_summary.json + gate2_redo_arms.npz  "
          f"total {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
