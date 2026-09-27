"""Gate 0d — sign-preserving one-to-one consensus + fold refits + shuffled-time
null + precision-matrix baseline. Per PREREG_Gate0d.md."""
from __future__ import annotations
import json, sys, warnings, time, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "reproducibility" / "operator_claims_benchmark" / "scripts"))

from run_operator_claims import prepare_marrow_ery
from scjdo.tl import fit_drift_branches
from scjdo.archetypes.decompose import jacobian_modes

warnings.filterwarnings("ignore")
OUT = REPO / "reproducibility" / "gates_r26" / "gate0d"
OUT.mkdir(parents=True, exist_ok=True)

K_ARCH = 5
N_EPOCHS = 3000
GRID = 150
N_FOLDS = 3
SEEDS = [0, 1]
TOL = 0.7


def sign_normalise(P):
    """(K, D, D) → (K, D*D) sign-normalised (leading-magnitude element +)."""
    K, D, _ = P.shape
    F = P.reshape(K, -1).astype(np.float64)
    F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)
    # flip so leading-|magnitude| element is positive
    for k in range(K):
        i = int(np.argmax(np.abs(F[k])))
        if F[k, i] < 0:
            F[k] = -F[k]
    return F


def hungarian_signed_cos(F_ref, F_other):
    """Both (K, D²) sign-normalised. Returns list of (ref_i, other_j, signed_cos)."""
    K = F_ref.shape[0]
    S = F_ref @ F_other.T  # signed cosine matrix (K, K)
    cost = 1.0 - S
    r, c = linear_sum_assignment(cost)
    return [(int(r[k]), int(c[k]), float(S[r[k], c[k]])) for k in range(K)]


def fit_one(ery, cell_mask, tau_used, seed):
    """Fit fit_drift_branches on cells[cell_mask], using tau_used as pseudotime."""
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


def build_precision_tensor(X_full, tau_full, grid, bandwidth):
    """Kernel-aggregated local precision tensor (T, D, D)."""
    T = len(grid); N, D = X_full.shape
    Xf = X_full.astype(np.float64); tauf = tau_full.astype(np.float64)
    P = np.zeros((T, D, D), dtype=np.float32)
    for i, tc in enumerate(grid):
        w = np.exp(-((tauf - tc) ** 2) / (2.0 * bandwidth ** 2))
        W = w.sum()
        if W < 1e-9: continue
        mu = (w[:, None] * Xf).sum(0) / W
        Xc = Xf - mu[None, :]
        Sigma = ((w[:, None, None] * Xc[:, :, None] * Xc[:, None, :]).sum(0) / W)
        lam = 1e-3 * np.trace(Sigma) / D
        P[i] = np.linalg.pinv(Sigma + lam * np.eye(D)).astype(np.float32)
    return P


def main():
    print("=" * 80); print("Gate 0d"); print("=" * 80)
    t_all = time.time()
    ery = prepare_marrow_ery()
    tau = ery.obs["pseudotime"].values.astype(np.float32)
    print(f"substrate: {ery.shape}, tau=[{tau.min():.3f}, {tau.max():.3f}]")

    # shuffled-time null (fixed permutation)
    rng = np.random.default_rng(0)
    tau_shuffled = tau.copy()
    rng.shuffle(tau_shuffled)
    print(f"tau_shuffled first 5: {tau_shuffled[:5]}")

    # folds
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
    fold_train_masks = []
    for tr, te in kf.split(np.arange(ery.n_obs)):
        mask = np.zeros(ery.n_obs, dtype=bool); mask[tr] = True
        fold_train_masks.append(mask)
    print(f"fold train counts: {[int(m.sum()) for m in fold_train_masks]}")

    # Fits: 3 folds × 2 seeds × 2 conditions = 12
    fits = {"REAL": [], "NULL": []}
    for cond_name, tau_used in [("REAL", tau), ("NULL", tau_shuffled)]:
        print(f"\n[{cond_name}] fitting {N_FOLDS} folds × {len(SEEDS)} seeds ...")
        for f_idx in range(N_FOLDS):
            for seed in SEEDS:
                ck = OUT / f"fit_{cond_name}_fold{f_idx}_seed{seed}.pkl"
                if ck.exists():
                    print(f"  {cond_name} fold={f_idx} seed={seed}: cache hit")
                    fits[cond_name].append(pickle.loads(ck.read_bytes()))
                    continue
                t0 = time.time()
                res = fit_one(ery, fold_train_masks[f_idx], tau_used, seed)
                ck.write_bytes(pickle.dumps(res))
                fits[cond_name].append(res)
                print(f"  {cond_name} fold={f_idx} seed={seed}: {time.time()-t0:.1f}s "
                      f"bw={res['bandwidth']:.4f}")

    # Consensus per condition
    def consensus(condition_fits):
        F_all = [sign_normalise(f["patterns"]) for f in condition_fits]  # 6 x (5, 900)
        F_ref = F_all[0]
        matched = []  # 5 per non-ref replicate
        for j in range(1, len(F_all)):
            pairs = hungarian_signed_cos(F_ref, F_all[j])
            row_cos = [p[2] for p in sorted(pairs, key=lambda p: p[0])]  # ordered by ref arch
            matched.append(row_cos)
        matched = np.array(matched)  # (5, 5) matched signed cosines
        min_per_arch = matched.min(axis=0)  # (5,) worst match across 5 replicates
        n_stable = int((min_per_arch >= TOL).sum())
        return matched, min_per_arch, n_stable

    print("\n[consensus] REAL")
    matched_R, minc_R, n_stable_R = consensus(fits["REAL"])
    print(f"  matched (5 non-ref replicates × 5 archs):\n{matched_R}")
    print(f"  min signed cos per arch: {minc_R}")
    print(f"  n_stable_REAL (min ≥ {TOL}): {n_stable_R} / 5")

    print("\n[consensus] NULL")
    matched_N, minc_N, n_stable_N = consensus(fits["NULL"])
    print(f"  matched:\n{matched_N}")
    print(f"  min signed cos per arch: {minc_N}")
    print(f"  n_stable_NULL: {n_stable_N} / 5")

    print(f"\n[real vs null] Δ n_stable = {n_stable_R - n_stable_N}")

    # Precision-matrix baseline
    print("\n[precision baseline]")
    ref = fits["REAL"][0]  # fold-0 seed-0 real
    bw = ref["bandwidth"]
    tc = ref["t_centers"]
    # rebuild ery for full-cohort precision tensor
    X_full = ery.obsm["X_fa"].astype(np.float64)
    P_tensor = build_precision_tensor(X_full, tau, tc, bw)
    print(f"  P_tensor {P_tensor.shape}  bw={bw:.4f}")
    p_patterns, _, err = jacobian_modes(torch.tensor(P_tensor), rank=K_ARCH,
                                          n_restarts=5, seed=0)
    p_patterns = p_patterns.numpy()
    F_prec = sign_normalise(p_patterns)
    pairs = hungarian_signed_cos(sign_normalise(ref["patterns"]), F_prec)
    prec_cos = [p[2] for p in sorted(pairs, key=lambda p: p[0])]
    med_prec = float(np.median(prec_cos))
    print(f"  matched signed cos (J ref vs P): {prec_cos}")
    print(f"  median: {med_prec:.4f}")

    # Verdict
    cond1 = n_stable_R >= 2
    cond2 = (n_stable_R - n_stable_N) >= 2
    cond3 = med_prec < TOL
    all_pass = cond1 and cond2 and cond3
    verdict = "PASS Gate 0d" if all_pass else "FAIL Gate 0d"
    print(f"\nVerdict: {verdict}")
    print(f"  cond1 (n_stable_REAL ≥ 2): {cond1}  ({n_stable_R})")
    print(f"  cond2 (Δ ≥ 2): {cond2}  ({n_stable_R - n_stable_N})")
    print(f"  cond3 (J ≠ precision, median < 0.7): {cond3}  (median={med_prec:.3f})")

    def _tn(v):
        if isinstance(v, (np.floating,)): return float(v)
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, (list, tuple)): return [_tn(x) for x in v]
        if isinstance(v, dict): return {k: _tn(x) for k, x in v.items()}
        return v

    (OUT / "gate0d_summary.json").write_text(json.dumps(_tn({
        "n_folds": N_FOLDS, "seeds": SEEDS, "K": K_ARCH,
        "tolerance": TOL,
        "REAL": {"matched": matched_R, "min_per_arch": minc_R,
                  "n_stable": n_stable_R},
        "NULL": {"matched": matched_N, "min_per_arch": minc_N,
                  "n_stable": n_stable_N},
        "precision_baseline": {"matched_signed_cos": prec_cos,
                                 "median": med_prec},
        "verdict": verdict,
        "conditions": {"cond1_n_stable_REAL_ge_2": cond1,
                        "cond2_delta_ge_2": cond2,
                        "cond3_J_not_precision": cond3},
    }), indent=2))
    print(f"\n[written] gate0d_summary.json  total {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
