"""Task B — recompute EL03 (whitened symmetric part vs Lyapunov D_hat and
vs local precision) on ALL SIX real Gate 0d v2 fits, with across-fit spread.
Reads existing fit_REAL_fold{0,1,2}_seed{0,1}.pkl. No new fits."""
from __future__ import annotations
import json, sys, pickle, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "reproducibility" / "operator_claims_benchmark" / "scripts"))
sys.path.insert(0, str(REPO / "reproducibility" / "gates_r26" / "gate0d_v2"))

from run_operator_claims import prepare_marrow_ery
from run_gate0d_v2 import (
    build_covariance_tensor, whitened_sym, leading_eig_cos, frob_corr_profile,
    N_FOLDS, SEEDS,
)

OUT = REPO / "reproducibility" / "gates_r26" / "gate0d_v2"


def main():
    print("=" * 80); print("Task B — EL03 across all 6 REAL Gate 0d v2 fits"); print("=" * 80)
    t0 = time.time()
    ery = prepare_marrow_ery()
    tau = ery.obs["pseudotime"].values.astype(np.float32)
    X_full = ery.obsm["X_fa"].astype(np.float64)

    # Fold masks (same seed=0 KFold as Gate 0d v2)
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
    fold_masks = []
    for tr, te in kf.split(np.arange(ery.n_obs)):
        m = np.zeros(ery.n_obs, dtype=bool); m[tr] = True
        fold_masks.append(m)

    # D_hat_FA (constant across fits)
    fa_load = ery.varm["fa_loadings"].astype(np.float64)
    counts = ery.X
    if hasattr(counts, "toarray"): counts = counts.toarray()
    gene_var = counts.astype(np.float64).mean(axis=0)   # Poisson: var = mean
    D_hat_FA = fa_load.T @ np.diag(gene_var) @ fa_load  # (30, 30)
    D = D_hat_FA.shape[0]

    per_fit = []
    for fold_idx in range(N_FOLDS):
        for seed in SEEDS:
            ck = OUT / f"fit_REAL_fold{fold_idx}_seed{seed}.pkl"
            fit = pickle.loads(ck.read_bytes())
            bw = fit["bandwidth"]
            J = fit["J_tensor"].astype(np.float32)
            grid = fit["t_centers"]
            # Sigma tensor on the fold's train cells at same grid + bandwidth
            X_tr = X_full[fold_masks[fold_idx]]
            tau_tr = tau[fold_masks[fold_idx]]
            Sigma = build_covariance_tensor(X_tr, tau_tr, grid, bw)

            # Whitened sym
            _, sym_W = whitened_sym(J, Sigma)

            # Predicted sym
            T = sym_W.shape[0]
            predicted_sym = np.zeros_like(sym_W)
            local_prec = np.zeros_like(sym_W)
            for i in range(T):
                S = Sigma[i].astype(np.float64)
                S = S + 1e-3 * np.trace(S) / D * np.eye(D)
                w, v = np.linalg.eigh(S)
                w = np.clip(w, 1e-9, None)
                Sinv_half = (v * (1.0 / np.sqrt(w))) @ v.T
                predicted_sym[i] = -0.5 * Sinv_half @ D_hat_FA @ Sinv_half
                local_prec[i] = np.linalg.inv(S)

            cos_pred = leading_eig_cos(sym_W, predicted_sym)
            cos_prec = leading_eig_cos(sym_W, local_prec)
            frob_pred = frob_corr_profile(sym_W, predicted_sym)
            frob_prec = frob_corr_profile(sym_W, local_prec)

            row = {
                "fold": fold_idx, "seed": seed, "bandwidth": bw,
                "median_cos_vs_predicted": float(np.median(cos_pred)),
                "mean_frob_vs_predicted": float(frob_pred.mean()),
                "median_cos_vs_precision": float(np.median(cos_prec)),
                "mean_frob_vs_precision": float(frob_prec.mean()),
            }
            per_fit.append(row)
            print(f"  fold={fold_idx} seed={seed} bw={bw:.4f}  "
                  f"|cos|_pred={row['median_cos_vs_predicted']:.4f}  "
                  f"Frob_pred={row['mean_frob_vs_predicted']:+.4f}  "
                  f"|cos|_prec={row['median_cos_vs_precision']:.4f}  "
                  f"Frob_prec={row['mean_frob_vs_precision']:+.4f}")

    # Aggregate: mean ± SD across 6 fits per metric
    def agg(key):
        vals = np.array([r[key] for r in per_fit])
        return {"mean": float(vals.mean()),
                "sd": float(vals.std(ddof=1)),
                "median": float(np.median(vals)),
                "min": float(vals.min()),
                "max": float(vals.max())}

    summary = {
        "n_fits": len(per_fit),
        "per_fit": per_fit,
        "across_fit": {
            "median_cos_vs_predicted": agg("median_cos_vs_predicted"),
            "mean_frob_vs_predicted": agg("mean_frob_vs_predicted"),
            "median_cos_vs_precision": agg("median_cos_vs_precision"),
            "mean_frob_vs_precision": agg("mean_frob_vs_precision"),
        }
    }
    print("\n=== ACROSS-FIT SUMMARY (n=6 REAL fits) ===")
    for k, v in summary["across_fit"].items():
        print(f"  {k}: mean {v['mean']:+.4f} sd {v['sd']:.4f}  "
              f"[{v['min']:+.4f}, {v['max']:+.4f}]")

    (OUT / "task_B_precision_allfits.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[written] task_B_precision_allfits.json  total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
