"""Gate 0d v2 per PREREG_Task3_Gate0d_v2.md.
Proper nulls (block permutation + circular shift), Hungarian 1-to-1
signed-cos stability, held-out gain K_eff, whitened precision comparison."""
from __future__ import annotations
import json, sys, warnings, time, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment, nnls

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "New_analysis" / "operator_claims_benchmark" / "scripts"))

from run_operator_claims import prepare_marrow_ery
from scjdo.tl import fit_drift_branches
from scjdo.archetypes.decompose import jacobian_modes

warnings.filterwarnings("ignore")
OUT = REPO / "New_analysis" / "gates_r26" / "gate0d_v2"
OUT.mkdir(parents=True, exist_ok=True)

K_ARCH = 5
N_EPOCHS = 3000
GRID = 150
N_FOLDS = 3
SEEDS = [0, 1]
STAB_TOL = 0.85
N_NULL_DRAWS = 10   # per null type per fold


def sign_normalise(P):
    """(K, D, D) → (K, D²) unit-L2 sign-normalised (leading-|magnitude| element +)."""
    K, D, _ = P.shape
    F = P.reshape(K, -1).astype(np.float64)
    F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)
    for k in range(K):
        i = int(np.argmax(np.abs(F[k])))
        if F[k, i] < 0:
            F[k] = -F[k]
    return F


def hungarian_signed(F_ref, F_other):
    K = F_ref.shape[0]
    S = F_ref @ F_other.T
    r, c = linear_sum_assignment(1.0 - S)
    return [(int(r[k]), int(c[k]), float(S[r[k], c[k]])) for k in range(K)]


def fit_one(ery, cell_mask, tau_used, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    b = ery[cell_mask].copy()
    b.obs["pseudotime"] = tau_used[cell_mask]
    b.obsm["branch_masks"] = pd.DataFrame({"Ery": np.ones(b.n_obs, dtype=bool)},
                                             index=b.obs_names)
    b.obs["cell_fate"] = "Other"
    pt = b.obs["pseudotime"].values
    b.obs.loc[pt <= np.quantile(pt, 0.3), "cell_fate"] = "Progenitor"
    b.obs.loc[pt >= np.quantile(pt, 0.7), "cell_fate"] = "Ery"
    fit_drift_branches(
        b, rep="X_fa", branch_key="branch_masks", branch_names=["Ery"],
        time_key="pseudotime", groupby="cell_fate",
        progenitor_cluster="Progenitor",
        terminal_clusters={"Ery": "Ery"},
        bias_strength=1.5, n_archetypes=K_ARCH, n_epochs=N_EPOCHS,
        vel_scale=0.0, hidden=256, depth=4, sigma=0.10,
        windowing="kernel", bandwidth="auto", grid_size=GRID,
        seed=seed, verbose=False,
    )
    res = b.uns["scjdo_Ery"]
    return {"patterns": np.asarray(res["patterns"]).astype(np.float32),
            "J_tensor": np.asarray(res["J_tensor"]).astype(np.float32),
            "t_centers": np.asarray(res["t_centers"]).astype(np.float32),
            "bandwidth": float(res.get("bandwidth", 0.05))}


def block_permutation(tau, bandwidth, rng):
    """Partition [0, 1] into blocks of width bandwidth; permute block order,
    keep within-block ordering."""
    n = len(tau)
    edges = np.arange(0.0, 1.0 + bandwidth, bandwidth)
    if edges[-1] < 1.0 + 1e-9:
        edges = np.append(edges, 1.0 + 1e-6)
    block_ids = np.digitize(tau, edges[1:-1])  # 0..len(edges)-2
    unique_blocks = np.unique(block_ids)
    perm = rng.permutation(unique_blocks)
    # remap block ids: block b → position of b in perm
    remap = {b: i for i, b in enumerate(perm)}
    new_block_pos = np.array([remap[b] for b in block_ids])
    # τ within block: preserve relative offset within the block
    # new τ = new_block_pos * bandwidth + within_block_offset
    within = tau - np.array([edges[b] for b in block_ids])
    new_tau = new_block_pos * bandwidth + within
    # renormalise to [0, 1]
    new_tau = np.clip(new_tau, 0.0, 1.0 - 1e-6)
    return new_tau.astype(np.float32)


def circular_shift(tau, delta):
    return ((tau + delta) % 1.0).astype(np.float32)


def whitened_sym(J, Sigma, ridge=1e-3):
    """W = Σ^{-1/2} J Σ^{1/2}, sym_W = 0.5 (W + W.T)."""
    T, D, _ = J.shape
    out_W = np.zeros_like(J, dtype=np.float64)
    out_sym = np.zeros_like(J, dtype=np.float64)
    for i in range(T):
        S = Sigma[i].astype(np.float64)
        S = S + ridge * np.trace(S) / D * np.eye(D)
        w, v = np.linalg.eigh(S)
        w = np.clip(w, 1e-9, None)
        Sinv_half = (v * (1.0 / np.sqrt(w))) @ v.T
        Shalf = (v * np.sqrt(w)) @ v.T
        Wm = Sinv_half @ J[i].astype(np.float64) @ Shalf
        out_W[i] = Wm
        out_sym[i] = 0.5 * (Wm + Wm.T)
    return out_W, out_sym


def build_covariance_tensor(X, tau, grid, bandwidth):
    """Kernel-aggregated local covariance (T, D, D) — same as Gate 0b."""
    T = len(grid); N, D = X.shape
    Xf = X.astype(np.float64); tauf = tau.astype(np.float64)
    C = np.zeros((T, D, D), dtype=np.float32)
    for i, tc in enumerate(grid):
        w = np.exp(-((tauf - tc) ** 2) / (2.0 * bandwidth ** 2))
        W = w.sum()
        if W < 1e-9: continue
        mu = (w[:, None] * Xf).sum(0) / W
        Xc = Xf - mu[None, :]
        C[i] = ((w[:, None, None] * Xc[:, :, None] * Xc[:, None, :]).sum(0) / W).astype(np.float32)
    return C


def leading_eig_cos(A, B):
    """Median-over-τ |cos(leading eigvec A, leading eigvec B)|. A,B: (T, D, D)."""
    T = A.shape[0]
    cs = np.zeros(T)
    for i in range(T):
        wa, va = np.linalg.eig(A[i].astype(np.float64))
        wb, vb = np.linalg.eig(B[i].astype(np.float64))
        ka = int(np.argmax(np.real(wa))); kb = int(np.argmax(np.real(wb)))
        u = np.real(va[:, ka]); v = np.real(vb[:, kb])
        cs[i] = abs(float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)))
    return cs


def frob_corr_profile(A, B):
    T = A.shape[0]
    out = np.zeros(T)
    for i in range(T):
        a = A[i].reshape(-1).astype(np.float64); b = B[i].reshape(-1).astype(np.float64)
        out[i] = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return out


def main():
    print("=" * 80); print("Gate 0d v2"); print("=" * 80)
    t_all = time.time()
    ery = prepare_marrow_ery()
    tau = ery.obs["pseudotime"].values.astype(np.float32)
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
    fold_masks = []
    for tr, te in kf.split(np.arange(ery.n_obs)):
        m = np.zeros(ery.n_obs, dtype=bool); m[tr] = True
        fold_masks.append(m)
    print(f"folds train counts: {[int(m.sum()) for m in fold_masks]}")

    # ── REAL fits ─────────────────────────────────────────────────────
    print(f"\n[REAL] fitting {N_FOLDS} folds × {len(SEEDS)} seeds ...")
    real_fits = []
    for f_idx in range(N_FOLDS):
        for seed in SEEDS:
            ck = OUT / f"fit_REAL_fold{f_idx}_seed{seed}.pkl"
            if ck.exists():
                real_fits.append((f_idx, seed, pickle.loads(ck.read_bytes())))
                print(f"  REAL fold={f_idx} seed={seed}: cache hit"); continue
            t0 = time.time()
            fit = fit_one(ery, fold_masks[f_idx], tau, seed)
            ck.write_bytes(pickle.dumps(fit))
            real_fits.append((f_idx, seed, fit))
            print(f"  REAL fold={f_idx} seed={seed}: {time.time()-t0:.1f}s bw={fit['bandwidth']:.4f}")

    # bandwidth from REAL fold-0 seed-0 for null block sizing
    bw = real_fits[0][2]["bandwidth"]
    print(f"  bandwidth (fold-0 seed-0 REAL): {bw:.4f}")

    # ── NULLs ─────────────────────────────────────────────────────────
    null_fits = {"block": [], "circular": []}
    for null_type in ("block", "circular"):
        print(f"\n[{null_type}] {N_NULL_DRAWS} draws × {N_FOLDS} folds ...")
        for f_idx in range(N_FOLDS):
            for draw in range(N_NULL_DRAWS):
                ck = OUT / f"fit_NULL_{null_type}_fold{f_idx}_draw{draw}.pkl"
                if ck.exists():
                    null_fits[null_type].append((f_idx, draw, pickle.loads(ck.read_bytes())))
                    print(f"  {null_type} fold={f_idx} draw={draw}: cache hit"); continue
                rng = np.random.default_rng(f_idx * 100 + draw)
                if null_type == "block":
                    tau_null = block_permutation(tau, bw, rng)
                else:
                    delta = float(rng.uniform(0.0, 1.0))
                    tau_null = circular_shift(tau, delta)
                t0 = time.time()
                fit = fit_one(ery, fold_masks[f_idx], tau_null, seed=0)
                ck.write_bytes(pickle.dumps(fit))
                null_fits[null_type].append((f_idx, draw, fit))
                print(f"  {null_type} fold={f_idx} draw={draw}: {time.time()-t0:.1f}s")

    # ── Stability under REAL and each null ────────────────────────────
    # Reference per fold: fold seed-0 REAL
    fold_refs = {f: sign_normalise(fit["patterns"])
                  for f, s, fit in real_fits if s == SEEDS[0]}
    def match_to_fold_ref(patterns_np, fold_idx):
        return hungarian_signed(fold_refs[fold_idx], sign_normalise(patterns_np))
    # REAL non-ref matches: seed-1 fits + medoid analysis across folds
    real_non_ref_signed = []  # per fold list of 5 signed-cos rows
    for f, s, fit in real_fits:
        if s == SEEDS[0]: continue
        pairs = match_to_fold_ref(fit["patterns"], f)
        row = [p[2] for p in sorted(pairs, key=lambda p: p[0])]
        real_non_ref_signed.append({"fold": f, "seed": s, "row": row})
    # Cross-fold: match each seed-0 fold≠0 REAL to fold-0's ref? No — the
    # prereg says reference per fold, not cross-fold. Instead, medoid:
    # We report per-fold, per-archetype `min signed cos` over 1 non-ref REAL fit
    # (seed=1). Since there is only 1 non-ref per fold, min-over-replicates is
    # the raw match. Report as 5 per fold.
    print("\n[stability REAL per fold]")
    for r in real_non_ref_signed:
        print(f"  fold={r['fold']} seed={r['seed']}: signed cos per arch = "
              f"{[f'{v:+.3f}' for v in r['row']]}")
    # Fraction of REAL matches ≥ 0.85 across all 5 archs × 3 folds = 15
    real_matches = np.array([[v for v in r["row"]] for r in real_non_ref_signed])
    real_frac_stable = float((real_matches >= STAB_TOL).sum() / real_matches.size)
    print(f"  fraction REAL matches at signed cos ≥ {STAB_TOL} "
          f"= {real_frac_stable:.3f}  ({int((real_matches >= STAB_TOL).sum())}/{real_matches.size})")

    # NULL fraction stable per null type per fold, per draw
    null_frac = {}
    for null_type, fits in null_fits.items():
        rows = []
        for f, d, fit in fits:
            pairs = match_to_fold_ref(fit["patterns"], f)
            row = [p[2] for p in sorted(pairs, key=lambda p: p[0])]
            rows.append(row)
        arr = np.array(rows)  # (N_draws * N_folds, 5)
        null_frac[null_type] = {
            "matched_signed_cos": arr,
            "fraction_stable": float((arr >= STAB_TOL).sum() / arr.size),
        }
        print(f"  fraction {null_type}-null at signed cos ≥ {STAB_TOL} = "
              f"{null_frac[null_type]['fraction_stable']:.3f}")

    # ── Held-out gain K_eff ───────────────────────────────────────────
    print("\n[K_eff via held-out NNLS gain]")
    # Discovery = fold-0 seed-0 REAL. Validation = folds 1,2 seed-0 REAL.
    disc_fit = [f for (fi, s, f) in real_fits if fi == 0 and s == 0][0]
    val_fits = [f for (fi, s, f) in real_fits if fi != 0 and s == 0]
    J_disc = disc_fit["J_tensor"].astype(np.float64)
    D = J_disc.shape[1]

    def compute_err(J_val, dictionary):
        """Reconstruct J_val (T, D, D) via NNLS onto (K, D²) dictionary."""
        T = J_val.shape[0]
        errs_num = 0.0; errs_den = 0.0
        for i in range(T):
            v = J_val[i].reshape(-1)
            act, _ = nnls(dictionary.T, v)
            recon = dictionary.T @ act
            errs_num += float(np.linalg.norm(v - recon) ** 2)
            errs_den += float(np.linalg.norm(v) ** 2)
        return float(errs_num / (errs_den + 1e-12))

    err_by_K = {0: 1.0}
    for K in range(1, K_ARCH + 1):
        with torch.no_grad():
            pats, _, _ = jacobian_modes(torch.tensor(J_disc.astype(np.float32)),
                                          rank=K, n_restarts=5, seed=0)
        pats = pats.numpy().reshape(K, -1).astype(np.float64)  # (K, D²)
        # average error across val folds
        errs = [compute_err(vf["J_tensor"].astype(np.float64), pats) for vf in val_fits]
        err_by_K[K] = float(np.mean(errs))
        print(f"  K={K}: err = {err_by_K[K]:.4f}  gain(K) = {err_by_K[K-1] - err_by_K[K]:+.4f}")

    # null gain distribution — 10 block-null draws' dictionaries
    print("\n[null gain distribution]")
    null_gains = {K: [] for K in range(1, K_ARCH + 1)}
    for (fi, di, nf) in null_fits["block"][:10]:  # first 10 block-null fits
        prev_err = 1.0
        for K in range(1, K_ARCH + 1):
            with torch.no_grad():
                pats, _, _ = jacobian_modes(torch.tensor(nf["J_tensor"].astype(np.float32)),
                                              rank=K, n_restarts=3, seed=0)
            pats = pats.numpy().reshape(K, -1).astype(np.float64)
            errs = [compute_err(vf["J_tensor"].astype(np.float64), pats) for vf in val_fits]
            e_K = float(np.mean(errs))
            null_gains[K].append(prev_err - e_K)
            prev_err = e_K
    K_eff = 0
    K_gain_null_95 = {}
    for K in range(1, K_ARCH + 1):
        p95 = float(np.percentile(null_gains[K], 95))
        gK = err_by_K[K-1] - err_by_K[K]
        K_gain_null_95[K] = p95
        print(f"  K={K}: real gain = {gK:+.4f}, block-null 95th = {p95:+.4f}")
        if gK > p95:
            K_eff = K
    print(f"\n  K_eff = {K_eff}")

    # ── Whitened precision comparison ─────────────────────────────────
    print("\n[whitened precision comparison]")
    # local covariance tensor at same grid + bandwidth as REAL fold-0 seed-0
    X_full = ery.obsm["X_fa"].astype(np.float64)
    grid = disc_fit["t_centers"]
    Sigma = build_covariance_tensor(X_full, tau, grid, bw)
    print(f"  Sigma tensor {Sigma.shape}")
    Wm, sym_W = whitened_sym(disc_fit["J_tensor"], Sigma)

    # D_hat: gene-wise Poisson variance in gene space, projected to FA-30 via loadings
    from anndata import AnnData
    fa_load = ery.varm["fa_loadings"].astype(np.float64)  # (n_genes, 30)
    counts = ery.X
    if hasattr(counts, "toarray"): counts = counts.toarray()
    gene_var = counts.astype(np.float64).mean(axis=0)  # Poisson: variance = mean
    D_hat_FA = fa_load.T @ np.diag(gene_var) @ fa_load  # (30, 30)
    print(f"  D_hat_FA shape {D_hat_FA.shape}, |D_hat|_F = {np.linalg.norm(D_hat_FA):.3f}")

    T = sym_W.shape[0]
    predicted_sym = np.zeros_like(sym_W)
    local_prec = np.zeros_like(sym_W)
    for i in range(T):
        S = Sigma[i].astype(np.float64)
        S = S + 1e-3 * np.trace(S) / S.shape[0] * np.eye(S.shape[0])
        w, v = np.linalg.eigh(S)
        w = np.clip(w, 1e-9, None)
        Sinv_half = (v * (1.0 / np.sqrt(w))) @ v.T
        predicted_sym[i] = -0.5 * Sinv_half @ D_hat_FA @ Sinv_half
        local_prec[i] = np.linalg.inv(S)

    cos_ps_pred = leading_eig_cos(sym_W, predicted_sym)
    cos_ps_prec = leading_eig_cos(sym_W, local_prec)
    frob_pred = frob_corr_profile(sym_W, predicted_sym)
    frob_prec = frob_corr_profile(sym_W, local_prec)
    print(f"  sym_W vs predicted_sym: median |cos| = {np.median(cos_ps_pred):.4f}, "
          f"mean Frob corr = {frob_pred.mean():.4f}")
    print(f"  sym_W vs local precision: median |cos| = {np.median(cos_ps_prec):.4f}, "
          f"mean Frob corr = {frob_prec.mean():.4f}")

    # ── Save ────────────────────────────────────────────────────────────
    def _tn(v):
        if isinstance(v, (np.floating,)): return float(v)
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, (list, tuple)): return [_tn(x) for x in v]
        if isinstance(v, dict): return {k: _tn(x) for k, x in v.items()}
        return v
    payload = _tn({
        "n_folds": N_FOLDS, "seeds": SEEDS, "K": K_ARCH,
        "n_null_draws_per_fold": N_NULL_DRAWS,
        "stab_tol": STAB_TOL,
        "bandwidth": bw,
        "REAL_signed_cos_rows": [r["row"] for r in real_non_ref_signed],
        "REAL_fraction_stable": real_frac_stable,
        "NULL_fraction_stable": {t: null_frac[t]["fraction_stable"] for t in null_frac},
        "K_eff_gain": {"err_by_K": err_by_K,
                        "real_gain": {K: err_by_K[K-1] - err_by_K[K] for K in range(1, K_ARCH + 1)},
                        "block_null_95pct": K_gain_null_95,
                        "K_eff": K_eff},
        "precision_comparison": {
            "sym_W_vs_predicted_D_hat": {"median_cos": float(np.median(cos_ps_pred)),
                                          "mean_frob_corr": float(frob_pred.mean())},
            "sym_W_vs_local_precision": {"median_cos": float(np.median(cos_ps_prec)),
                                          "mean_frob_corr": float(frob_prec.mean())},
        },
    })
    (OUT / "gate0d_v2_summary.json").write_text(json.dumps(payload, indent=2))
    print(f"\n[written] gate0d_v2_summary.json  total {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
