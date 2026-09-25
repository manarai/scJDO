"""Gate 0b — local-covariance archetype baseline. Per PREREG_Gate0b.md,
compare scJDO Jacobian archetypes against a covariance-tensor baseline
built with the same kernel grid and bandwidth."""
from __future__ import annotations
import json, sys, warnings, time
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "New_analysis" / "operator_claims_benchmark" / "scripts"))

from run_operator_claims import prepare_marrow_ery
from scjdo.tl import fit_drift_branches
from scjdo.archetypes.decompose import jacobian_modes

OUT = REPO / "New_analysis" / "gates_r26" / "gate0b"
OUT.mkdir(parents=True, exist_ok=True)


def build_covariance_tensor(X, tau, grid, bandwidth):
    """(T, D, D) kernel-weighted local sample covariance tensor.
    Same grid and bandwidth as scJDO's kernel windowing."""
    n_grid = len(grid)
    N, D = X.shape
    tau64 = tau.astype(np.float64); X64 = X.astype(np.float64)
    C = np.zeros((n_grid, D, D), dtype=np.float32)
    for i, tc in enumerate(grid):
        w = np.exp(-((tau64 - tc) ** 2) / (2.0 * bandwidth ** 2))
        W = w.sum()
        if W < 1e-9: continue
        mu = (w[:, None] * X64).sum(0) / W
        Xc = X64 - mu[None, :]
        C[i] = ((w[:, None, None] * Xc[:, :, None] * Xc[:, None, :]).sum(0) / W).astype(np.float32)
    return C


def fit_scjdo_reference(ery, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    b = ery.copy()
    b.obsm["branch_masks"] = pd.DataFrame({"Ery": np.ones(b.n_obs, dtype=bool)},
                                             index=b.obs_names)
    pt = b.obs["pseudotime"].values
    b.obs["cell_fate"] = "Other"
    b.obs.loc[b.obs["pseudotime"] <= np.quantile(pt, 0.3), "cell_fate"] = "Progenitor"
    b.obs.loc[b.obs["pseudotime"] >= np.quantile(pt, 0.7), "cell_fate"] = "Ery"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_drift_branches(
            b, rep="X_fa", branch_key="branch_masks", branch_names=["Ery"],
            time_key="pseudotime", groupby="cell_fate",
            progenitor_cluster="Progenitor",
            terminal_clusters={"Ery": "Ery"},
            bias_strength=1.5, n_archetypes=5, n_epochs=5000,
            vel_scale=2.0, hidden=256, depth=4, sigma=0.10,
            windowing="kernel", bandwidth="auto", grid_size=200,
            seed=seed, verbose=False,
        )
    res = b.uns["scjdo_Ery"]
    return {
        "J_tensor": np.asarray(res["J_tensor"]),
        "patterns": np.asarray(res["patterns"]),
        "activations": np.asarray(res["activations"]),
        "t_centers": np.asarray(res["t_centers"]),
        "max_eig": np.asarray(res["max_real_eig"]),
        "bandwidth": float(res.get("bandwidth", 0.05)),
    }


def pattern_cos(A, B):
    a = A.reshape(-1).astype(np.float64); b = B.reshape(-1).astype(np.float64)
    n = np.linalg.norm(a) * np.linalg.norm(b) + 1e-9
    return float(np.dot(a, b) / n)


def hungarian_match(P_ref, P_cov):
    from scipy.optimize import linear_sum_assignment
    K = P_ref.shape[0]
    cost = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            cost[i, j] = -abs(pattern_cos(P_ref[i], P_cov[j]))
    r, c = linear_sum_assignment(cost)
    matched_cos = [-cost[ri, ci] for ri, ci in zip(r, c)]
    return list(zip(r, c, matched_cos))


def leading_gene_top15(pat, loadings):
    eigvals, eigvecs = np.linalg.eig(pat.astype(np.float64))
    k_top = int(np.argmax(np.real(eigvals)))
    v = np.real(eigvecs[:, k_top])
    gene_load = np.abs(loadings @ v)
    return np.argsort(-gene_load)[:15]


def main():
    ery = prepare_marrow_ery()
    print(f"Substrate: Ery {ery.n_obs} × {ery.n_vars}")
    L = ery.varm["fa_loadings"]

    t0 = time.time()
    print("\n[scJDO reference fit (seed=42, ~4 min)] ...")
    ref = fit_scjdo_reference(ery, seed=42)
    print(f"  fit {time.time()-t0:.1f}s | bandwidth h*={ref['bandwidth']:.4f}  "
          f"J_tensor {ref['J_tensor'].shape}")

    print("\n[Covariance tensor build] ...")
    X = ery.obsm["X_fa"]
    tau = ery.obs["pseudotime"].values.astype(np.float32)
    C_tensor = build_covariance_tensor(X, tau, ref["t_centers"], ref["bandwidth"])
    print(f"  cov tensor: {C_tensor.shape}  |C|_F mean per slice: "
          f"{np.linalg.norm(C_tensor.reshape(len(C_tensor), -1), axis=1).mean():.3f}")

    # Semi-NMF on covariance tensor with 3 seeds
    print("\n[Semi-NMF on covariance tensor, 3 seeds] ...")
    from scjdo.archetypes.decompose import jacobian_modes as _jm
    cov_seed_results = []
    for seed in (0, 1, 2):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p, a, err = _jm(torch.tensor(C_tensor), rank=5, n_restarts=5, seed=seed)
        cov_seed_results.append({"patterns": p.numpy(), "activations": a.numpy(),
                                   "err": err, "seed": seed})
        print(f"  seed {seed}: err={err:.4f}")

    # Also run 3 seeds on the scJDO J_tensor for direct comparison
    print("\n[Semi-NMF on scJDO J_tensor, 3 seeds] ...")
    j_seed_results = []
    for seed in (0, 1, 2):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p, a, err = _jm(torch.tensor(ref["J_tensor"]), rank=5, n_restarts=5, seed=seed)
        j_seed_results.append({"patterns": p.numpy(), "activations": a.numpy(),
                                  "err": err, "seed": seed})
        print(f"  seed {seed}: err={err:.4f}")

    # Match cov[seed=s] archetypes to scJDO[seed=s] archetypes for each s
    print("\n" + "="*80)
    print("Matched-pattern cosine and gene-loading Jaccard (5 pairs × 3 seeds)")
    print("="*80)
    per_seed_matches = []
    for s in range(3):
        P_ref = j_seed_results[s]["patterns"]
        P_cov = cov_seed_results[s]["patterns"]
        matches = hungarian_match(P_ref, P_cov)
        row_cos = [m[2] for m in matches]
        # gene-loading Jaccard per pair
        row_jacc = []
        for ri, ci, cos in matches:
            g_ref = set(leading_gene_top15(P_ref[ri], L).tolist())
            g_cov = set(leading_gene_top15(P_cov[ci], L).tolist())
            row_jacc.append(len(g_ref & g_cov) / len(g_ref | g_cov))
        per_seed_matches.append({"cos_per_pair": row_cos, "jacc_per_pair": row_jacc})
        cos_str = " ".join(f"{c:.3f}" for c in row_cos)
        jacc_str = " ".join(f"{j:.2f}" for j in row_jacc)
        print(f"  seed {s}: matched cos [{cos_str}]  jacc [{jacc_str}]")

    # Aggregate: per-pair mean across seeds
    cos_matrix = np.array([r["cos_per_pair"] for r in per_seed_matches])
    jacc_matrix = np.array([r["jacc_per_pair"] for r in per_seed_matches])
    per_pair_mean_cos = cos_matrix.mean(axis=0)  # (5,)
    per_pair_mean_jacc = jacc_matrix.mean(axis=0)
    n_pairs_ge_09 = int((per_pair_mean_cos >= 0.9).sum())
    overall_mean_cos = float(cos_matrix.mean())
    print(f"\nPer-pair mean cos (across 3 seeds): "
          f"{[f'{c:.3f}' for c in per_pair_mean_cos]}")
    print(f"Per-pair mean top-15 Jaccard:       "
          f"{[f'{j:.2f}' for j in per_pair_mean_jacc]}")
    print(f"Overall mean matched cos: {overall_mean_cos:.3f}")
    print(f"Number of pairs with mean cos >= 0.9: {n_pairs_ge_09} / 5")

    # Stop-rule verdict (per PREREG_Gate0b.md)
    if n_pairs_ge_09 >= 3:
        verdict = ("STOP: >= 3 of 5 pairs have matched cos >= 0.9. "
                    "Jacobian archetypes do not carry more than local covariance. "
                    "Calibration paper only.")
    else:
        verdict = (f"PROCEED: only {n_pairs_ge_09} of 5 pairs have cos >= 0.9. "
                    "Jacobian adds distinctive structure vs local covariance. "
                    "Proceed to Gate 0c.")

    print("\n" + "="*80)
    print(f"VERDICT: {verdict}")
    print("="*80)

    (OUT / "gate0b_summary.json").write_text(json.dumps({
        "scjdo_bandwidth_h_star": ref["bandwidth"],
        "per_seed_matches": [
            {"seed": s, "cos_per_pair": per_seed_matches[s]["cos_per_pair"],
             "jacc_per_pair": per_seed_matches[s]["jacc_per_pair"]}
            for s in range(3)],
        "per_pair_mean_cos_across_seeds": per_pair_mean_cos.tolist(),
        "per_pair_mean_top15_jaccard_across_seeds": per_pair_mean_jacc.tolist(),
        "overall_mean_matched_cos": overall_mean_cos,
        "n_pairs_with_mean_cos_ge_0.9": n_pairs_ge_09,
        "stop_rule_threshold": "n_pairs_ge_0.9 >= 3 → stop",
        "verdict": verdict,
    }, indent=2))
    print(f"\nWritten: {OUT / 'gate0b_summary.json'}")


if __name__ == "__main__":
    main()
