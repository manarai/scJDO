"""Gate 2 — scNT-seq metabolic labelling per PREREG_Gate2.md.
Arms: R (Dynamo labelled reference), G (scJDO vel_scale=0), L (scJDO
with Dynamo velocities as V_ref, vel_scale=2). 3 seeds each for G, L."""
from __future__ import annotations
import json, sys, warnings, time, pickle
from pathlib import Path
import numpy as np, pandas as pd, torch
import scipy.stats as stats

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
OUT = REPO / "New_analysis" / "gates_r26" / "gate2"
DATA = OUT / "data"
DATA.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore")
SEEDS = [0, 1, 2]
GRID_SIZE = 150
N_ARCH = 5
N_EPOCHS = 3000
D_REP = 30


def load_or_download():
    import dynamo as dyn
    cache = DATA / "neuron_labeling.h5ad"
    if cache.exists():
        import anndata as ad
        return ad.read_h5ad(cache)
    import os
    cwd_backup = os.getcwd(); os.chdir(str(DATA.parent))
    try:
        a = dyn.sample_data.scNT_seq_neuron_labeling()
    finally:
        os.chdir(cwd_backup)
    # dynamo downloads to ./data/, which may be different from our DATA dir
    src = Path("data/neuron_labeling.h5ad")
    if src.exists() and not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        try:
            cache.write_bytes(src.read_bytes())
        except Exception:
            pass
    return a


def run_dynamo_reference(adata):
    """Arm R: labelling-based per-cell velocity in PCA-30 space, computed
    directly from the metabolic-labelling layer (new-transcript counts).
    We do not rely on dynamo's kinetic estimation (which failed on this
    dataset with `only 4 genes have finite velocity`). Instead: the
    labelling ratio X_new is projected into PCA space and taken as the
    reference velocity direction (up to a shared constant labelling
    duration)."""
    import dynamo as dyn
    import scipy.sparse as sp_sparse
    t0 = time.time()
    if adata.obs["time"].max() > 10:
        print("[R] converting time from minutes to hours...", flush=True)
        adata.obs["time"] = adata.obs["time"].astype(float) / 60.0
    print("[R] recipe_monocle (yields X_pca on total counts, PCs varm)...", flush=True)
    dyn.pp.recipe_monocle(adata, tkey="time", experiment_type="one-shot",
                           n_top_genes=2000, keep_filtered_cells=True)
    print(f"[R] moments (M_n, M_t)...", flush=True)
    dyn.tl.moments(adata)

    X_pca = np.asarray(adata.obsm["X_pca"]).astype(np.float32)      # (N, D_REP)
    # HVG-filtered PCA loadings for projecting per-gene signals into PCA
    hvg = adata.var["use_for_pca"].values if "use_for_pca" in adata.var else np.ones(adata.n_vars, dtype=bool)
    PCs = np.asarray(adata.varm["PCs"]).astype(np.float32)          # (n_hvg, D_REP)
    if PCs.shape[0] == adata.n_vars:
        PCs_hvg = PCs[hvg]
    else:
        PCs_hvg = PCs
    print(f"[R] X_pca={X_pca.shape}  PCs_hvg={PCs_hvg.shape}  HVG count={int(hvg.sum())}", flush=True)

    # M_n is the moment-smoothed new-transcript matrix
    M_n = adata.layers["M_n"] if "M_n" in adata.layers else adata.layers["X_new"]
    M_t = adata.layers["M_t"] if "M_t" in adata.layers else adata.layers["X_total"]
    if sp_sparse.issparse(M_n): M_n = M_n.toarray()
    if sp_sparse.issparse(M_t): M_t = M_t.toarray()
    M_n = M_n[:, hvg] if M_n.shape[1] == adata.n_vars else M_n
    M_t = M_t[:, hvg] if M_t.shape[1] == adata.n_vars else M_t

    # v_ref (per cell in PCA-30): log1p-normalise new counts, then project
    # (analogous to how X_pca is derived from log1p-normed total counts)
    v_ref_gene = np.log1p(M_n.astype(np.float32))    # (N, n_hvg)
    v_ref = v_ref_gene @ PCs_hvg                     # (N, D_REP)
    print(f"[R] v_ref (labelling-based, projected to PCA-30) = {v_ref.shape}, "
          f"|v_ref| median = {float(np.median(np.linalg.norm(v_ref, axis=1))):.3f}", flush=True)
    print(f"[R] done in {time.time()-t0:.1f}s", flush=True)
    return {"v_ref": v_ref, "X_pca": X_pca, "M_n": M_n, "M_t": M_t,
            "PCs_hvg": PCs_hvg, "adata": adata}


def compute_pseudotime(adata):
    """Cheap pseudotime: 4sU time (real time). scNT-seq neuron labelling has
    real experimental time — we use that as pseudotime directly, normalised to [0, 1]."""
    t = adata.obs["time"].values.astype(float)
    lo, hi = float(np.nanmin(t)), float(np.nanmax(t))
    if hi <= lo:
        return np.zeros_like(t, dtype=np.float32)
    return ((t - lo) / (hi - lo)).astype(np.float32)


def build_kernel_operator(X, tau, grid, bandwidth, V_per_cell=None):
    """Kernel-weighted local Jacobian tensor. If V_per_cell is None,
    fall back to local covariance (Gate 0b style). Otherwise, use
    kernel-weighted OLS estimate of J: at each grid, solve V ≈ J @ X_centered."""
    n_grid = len(grid); D = X.shape[1]
    J = np.zeros((n_grid, D, D), dtype=np.float32)
    Xf = X.astype(np.float64); tauf = tau.astype(np.float64)
    for i, tc in enumerate(grid):
        w = np.exp(-((tauf - tc) ** 2) / (2.0 * bandwidth ** 2))
        W = w.sum()
        if W < 1e-9: continue
        mu = (w[:, None] * Xf).sum(0) / W
        Xc = Xf - mu[None, :]
        if V_per_cell is not None:
            # ridge OLS: J = (V.T @ W @ Xc) @ inv(Xc.T @ W @ Xc + eps I)
            WXc = w[:, None] * Xc
            A = Xc.T @ WXc + 1e-3 * np.eye(D)
            B = V_per_cell.astype(np.float64).T @ WXc
            J[i] = (B @ np.linalg.inv(A)).astype(np.float32)
        else:
            # local covariance as "operator"
            J[i] = ((w[:, None, None] * Xc[:, :, None] * Xc[:, None, :]).sum(0) / W).astype(np.float32)
    return J


def leading_eig(J):
    """Per-slice leading real eigenvector + Re(lambda_max). J: (T, D, D)"""
    T = J.shape[0]
    lead_re = np.zeros(T, dtype=np.float32)
    lead_vec = np.zeros((T, J.shape[1]), dtype=np.float32)
    for i in range(T):
        eigval, eigvec = np.linalg.eig(J[i].astype(np.float64))
        k = int(np.argmax(np.real(eigval)))
        lead_re[i] = float(np.real(eigval[k]))
        lead_vec[i] = np.real(eigvec[:, k]).astype(np.float32)
        # sign convention: leading positive component of eigvec is positive
        if lead_vec[i, np.argmax(np.abs(lead_vec[i]))] < 0:
            lead_vec[i] = -lead_vec[i]
    return lead_re, lead_vec


def fit_scjdo_arm(X_pca, tau, seed, V_ref=None, vel_scale=0.0):
    """Fit scJDO on the bulk. If V_ref given, inject it via monkey-patch of
    _pseudotime_velocity so scJDO's DriftField uses Dynamo velocities as prior."""
    from scjdo.tl import fit_drift
    from scjdo.tl import _drift as _drift_mod
    torch.manual_seed(seed); np.random.seed(seed)
    import anndata as ad
    a = ad.AnnData(X=np.zeros((X_pca.shape[0], 1), dtype=np.float32))
    a.obsm["X_pca"] = X_pca.astype(np.float32)
    a.obs["pseudotime"] = tau.astype(np.float32)

    _orig = _drift_mod._pseudotime_velocity
    if V_ref is not None:
        def _inject(X_pca_arg, pt_arg, k=15):
            return V_ref.astype(np.float32)
        _drift_mod._pseudotime_velocity = _inject
    try:
        fit_drift(a, rep="X_pca", time_key="pseudotime",
                  n_archetypes=N_ARCH, n_epochs=N_EPOCHS,
                  vel_scale=vel_scale, hidden=256, depth=4, sigma=0.10,
                  windowing="kernel", bandwidth="auto", grid_size=GRID_SIZE,
                  seed=seed, verbose=False, key_added="scjdo_bulk")
    finally:
        _drift_mod._pseudotime_velocity = _orig

    res = a.uns["scjdo_bulk"]
    return {"J_tensor": np.asarray(res["J_tensor"]).astype(np.float32),
            "t_centers": np.asarray(res["t_centers"]).astype(np.float32),
            "max_real_eig": np.asarray(res["max_real_eig"]).astype(np.float32),
            "bandwidth": float(res.get("bandwidth", 0.05))}


def compute_per_cell_velocity_from_J(J_tensor, t_centers, X_pca, tau, mu_per_bin=None):
    """Approx per-cell velocity from a J_tensor along tau: v_i = J(τ_i) (x_i − μ(τ_i))."""
    idx = np.searchsorted(t_centers, tau).clip(0, len(t_centers) - 1)
    if mu_per_bin is None:
        mu_per_bin = np.zeros((len(t_centers), X_pca.shape[1]), dtype=np.float32)
        for i in range(len(t_centers)):
            in_bin = np.abs(tau - t_centers[i]) <= (t_centers[1] - t_centers[0])
            if in_bin.any():
                mu_per_bin[i] = X_pca[in_bin].mean(0)
    v_per_cell = np.zeros_like(X_pca, dtype=np.float32)
    for i in range(X_pca.shape[0]):
        b = int(idx[i])
        v_per_cell[i] = J_tensor[b] @ (X_pca[i] - mu_per_bin[b])
    return v_per_cell


def cos_per_row(A, B):
    a = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    b = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return (a * b).sum(axis=1)


def main():
    print("="*80); print("Gate 2: scNT-seq metabolic labelling"); print("="*80)
    t_all = time.time()

    # ---- Arm R ----
    print("\n[R] Dynamo reference pipeline")
    adata = load_or_download()
    print(f"  substrate: {adata.shape}")
    R = run_dynamo_reference(adata)
    X_pca = R["X_pca"]
    tau = compute_pseudotime(adata)
    print(f"  tau range: [{tau.min():.3f}, {tau.max():.3f}]  X_pca={X_pca.shape}")

    # Build reference Jacobian tensor by kernel-aggregating along pseudotime,
    # using Dynamo's per-cell velocities to fit local linear operator.
    print("[R] kernel-aggregated J_tensor from Dynamo v_ref...")
    # adaptive bandwidth: use scJDO's bandwidth from a quick preview fit
    from scjdo.archetypes.windowing import select_bandwidth
    grid = np.linspace(0, 1, GRID_SIZE).astype(np.float32)
    # first compute local covariance to estimate bandwidth
    J_dummy = None
    # use scJDO's select_bandwidth heuristic: fit at seed=0 vel_scale=0 first
    print("[G] scJDO seed=0 (for bandwidth h*)...")
    G_seed0 = fit_scjdo_arm(X_pca, tau, seed=0, V_ref=None, vel_scale=0.0)
    bw = G_seed0["bandwidth"]
    t_centers = G_seed0["t_centers"]
    print(f"  bandwidth h*={bw:.4f}, t_centers={t_centers.shape}")

    J_R = build_kernel_operator(X_pca, tau, t_centers, bw, V_per_cell=R["v_ref"])
    print(f"  J_R shape={J_R.shape}, |J_R|_F mean per slice={np.linalg.norm(J_R.reshape(len(J_R), -1), axis=1).mean():.3f}")

    R_lead_re, R_lead_vec = leading_eig(J_R)
    print(f"  R Re(lambda_max): [{R_lead_re.min():.3f}, {R_lead_re.max():.3f}]")

    # ---- Arm G — 3 seeds ----
    print("\n[G] scJDO geometry-only (vel_scale=0), 3 seeds")
    G_arms = {0: G_seed0}
    for seed in SEEDS[1:]:
        t0 = time.time()
        G_arms[seed] = fit_scjdo_arm(X_pca, tau, seed=seed, V_ref=None, vel_scale=0.0)
        print(f"  seed {seed} done in {time.time()-t0:.1f}s")

    # ---- Arm L — 3 seeds with V_ref = Dynamo velocities ----
    print("\n[L] scJDO labelling-supervised (vel_scale=2, V_ref=Dynamo velocities), 3 seeds")
    L_arms = {}
    for seed in SEEDS:
        t0 = time.time()
        L_arms[seed] = fit_scjdo_arm(X_pca, tau, seed=seed, V_ref=R["v_ref"], vel_scale=2.0)
        print(f"  seed {seed} done in {time.time()-t0:.1f}s")

    # ---- P1: data-determined agreement (cos of leading eigvec + Spearman on Re(lambda_max)) ----
    print("\n[metrics] P1 data-determined outputs")
    p1 = {}
    for label, arms in [("G", G_arms), ("L", L_arms)]:
        cos_lead = []
        spearman_re = []
        for seed in SEEDS:
            arm = arms[seed]
            l_re, l_vec = leading_eig(arm["J_tensor"])
            # cos leading eigvec vs R (sign-invariant abs)
            c = np.abs((l_vec * R_lead_vec).sum(axis=1)) / (
                np.linalg.norm(l_vec, axis=1) * np.linalg.norm(R_lead_vec, axis=1) + 1e-12
            )
            cos_lead.append(np.median(c))
            spearman_re.append(stats.spearmanr(l_re, R_lead_re).correlation)
        p1[label] = {"median_cos_leading_eigvec_per_seed": cos_lead,
                     "spearman_re_lmax_per_seed": spearman_re,
                     "median_cos_leading_eigvec_mean": float(np.mean(cos_lead)),
                     "spearman_re_lmax_mean": float(np.nanmean(spearman_re))}
        print(f"  {label}: median cos(leading eigvec, R) = "
              f"{np.mean(cos_lead):.3f}, Spearman(Re lam_max, R) = {np.nanmean(spearman_re):.3f}")

    # P1 pass: G median cos ≥ 0.5 AND G Spearman ≥ 0.5
    p1_pass = (p1["G"]["median_cos_leading_eigvec_mean"] >= 0.5 and
                p1["G"]["spearman_re_lmax_mean"] >= 0.5)
    print(f"  P1 pass? {p1_pass}")

    # ---- P2: prior-determined outputs (per-cell velocity cosine to R) ----
    print("\n[metrics] P2 prior-determined outputs")
    def per_cell_vel(arm):
        return compute_per_cell_velocity_from_J(arm["J_tensor"], arm["t_centers"],
                                                  X_pca, tau)
    cos_vG_R = []  # (N_cells, 3 seeds)
    cos_vL_R = []
    for seed in SEEDS:
        vG = per_cell_vel(G_arms[seed])
        vL = per_cell_vel(L_arms[seed])
        cos_vG_R.append(cos_per_row(vG, R["v_ref"]))
        cos_vL_R.append(cos_per_row(vL, R["v_ref"]))
    cos_vG_R = np.stack(cos_vG_R, axis=1)  # (N, 3)
    cos_vL_R = np.stack(cos_vL_R, axis=1)
    mean_cos_G = float(np.nanmean(cos_vG_R))
    mean_cos_L = float(np.nanmean(cos_vL_R))
    # Δ per cell per seed → paired bootstrap over cells (avg across seeds within cell)
    delta_per_cell = cos_vL_R.mean(axis=1) - cos_vG_R.mean(axis=1)
    rng = np.random.default_rng(0)
    N = len(delta_per_cell)
    boot_means = []
    for _ in range(2000):
        ii = rng.integers(0, N, size=N)
        boot_means.append(float(np.nanmean(delta_per_cell[ii])))
    ci_p2 = (float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5)))
    p2_pass = (mean_cos_L > mean_cos_G) and (ci_p2[0] > 0)
    print(f"  mean cos(vel_G, R) = {mean_cos_G:+.4f}")
    print(f"  mean cos(vel_L, R) = {mean_cos_L:+.4f}")
    print(f"  Δ (L-G) per-cell mean = {float(np.nanmean(delta_per_cell)):+.4f}  "
          f"95%CI [{ci_p2[0]:+.4f}, {ci_p2[1]:+.4f}]")
    print(f"  P2 pass? {p2_pass}")

    # ---- P3: seed stability (variance of per-cell drift across seeds) ----
    print("\n[metrics] P3 seed stability")
    vG_stack = np.stack([per_cell_vel(G_arms[s]) for s in SEEDS], axis=0)  # (3, N, D)
    vL_stack = np.stack([per_cell_vel(L_arms[s]) for s in SEEDS], axis=0)
    var_G_per_cell = ((vG_stack - vG_stack.mean(axis=0, keepdims=True)) ** 2).sum(axis=-1).mean(axis=0)  # (N,)
    var_L_per_cell = ((vL_stack - vL_stack.mean(axis=0, keepdims=True)) ** 2).sum(axis=-1).mean(axis=0)
    ratio = var_L_per_cell / (var_G_per_cell + 1e-12)
    boot_ratios = []
    for _ in range(2000):
        ii = rng.integers(0, N, size=N)
        boot_ratios.append(float(np.nanmedian(ratio[ii])))
    ci_p3 = (float(np.percentile(boot_ratios, 2.5)), float(np.percentile(boot_ratios, 97.5)))
    p3_pass = (np.nanmedian(ratio) < 1.0) and (ci_p3[1] < 1.0)
    print(f"  var_G per-cell mean = {float(np.nanmean(var_G_per_cell)):.4f}")
    print(f"  var_L per-cell mean = {float(np.nanmean(var_L_per_cell)):.4f}")
    print(f"  median var_L/var_G = {float(np.nanmedian(ratio)):.4f}  "
          f"95%CI [{ci_p3[0]:.4f}, {ci_p3[1]:.4f}]")
    print(f"  P3 pass? {p3_pass}")

    # ---- Verdict ----
    all_pass = p1_pass and p2_pass and p3_pass
    verdict = "PASS Gate 2" if all_pass else "FAIL Gate 2"
    print(f"\nVerdict: {verdict}  (P1={p1_pass}, P2={p2_pass}, P3={p3_pass})")

    (OUT / "gate2_summary.json").write_text(json.dumps({
        "shape": list(X_pca.shape), "bandwidth": bw,
        "P1": p1, "P1_pass": bool(p1_pass),
        "P2": {"mean_cos_vG_R": mean_cos_G, "mean_cos_vL_R": mean_cos_L,
                "delta_mean_per_cell": float(np.nanmean(delta_per_cell)),
                "delta_95ci": ci_p2, "pass": bool(p2_pass)},
        "P3": {"var_G_mean": float(np.nanmean(var_G_per_cell)),
                "var_L_mean": float(np.nanmean(var_L_per_cell)),
                "median_var_ratio": float(np.nanmedian(ratio)),
                "median_var_ratio_95ci": ci_p3, "pass": bool(p3_pass)},
        "verdict": verdict,
    }, indent=2))
    np.savez(OUT / "gate2_arms.npz",
              J_R=J_R, R_v_ref=R["v_ref"], X_pca=X_pca, tau=tau, t_centers=t_centers,
              **{f"G_seed{s}_J": G_arms[s]["J_tensor"] for s in SEEDS},
              **{f"L_seed{s}_J": L_arms[s]["J_tensor"] for s in SEEDS})
    print(f"\n[written] gate2_summary.json + gate2_arms.npz  total {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
