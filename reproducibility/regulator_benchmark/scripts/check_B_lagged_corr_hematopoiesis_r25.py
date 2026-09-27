"""r25 Check B — Does lagged_corr_hub recover GATA2 and IRF8 on real
hematopoiesis? If yes, the "unique scJDO wins" from Round 3b aren't
scJDO-specific either.

Uses the pinned Palantir 1.4.4 + MAGIC pipeline (r24 Task C — Fig 3's
cell counts 1151/1903/2008). No scJDO fit needed — this is a
baseline-only check. Ranks canonical TFs by lagged_corr_hub score on
each branch's cells.
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "reproducibility" / "regulator_benchmark" / "scripts"))

OUT = REPO / "reproducibility" / "regulator_benchmark" / "outputs_r25_check_B"
OUT.mkdir(parents=True, exist_ok=True)

BRANCH_TFS = {
    "Ery":  {"GATA1", "GATA2", "KLF1", "TAL1"},
    "DC":   {"IRF7", "IRF8"},
    "Mono": {"CEBPA", "CEBPE", "IRF8"},
}


def preprocess_marrow_fig3():
    """Exact Fig 3 preprocessing incl MAGIC — r24 Task C recipe."""
    import scanpy as sc, palantir, anndata as ad
    from sklearn.decomposition import FactorAnalysis
    a = ad.read_h5ad(REPO / "examples" / "marrow_sample_scseq_counts.h5ad")
    sc.pp.normalize_per_cell(a); palantir.preprocess.log_transform(a)
    sc.pp.highly_variable_genes(a, n_top_genes=1000, flavor="cell_ranger")
    hvg = a.var["highly_variable"].values
    Xhvg = a[:, hvg].X
    if sp.issparse(Xhvg): Xhvg = Xhvg.toarray()
    fa = FactorAnalysis(n_components=30, random_state=42)
    a.obsm["X_fa"] = fa.fit_transform(Xhvg).astype(np.float32)
    palantir.utils.run_diffusion_maps(a, n_components=20, pca_key="X_fa")
    palantir.utils.determine_multiscale_space(a)
    sc.pp.neighbors(a, use_rep="X_fa")
    palantir.utils.run_magic_imputation(a)   # ← the Fig 3 step
    START = "Run5_164698952452459"
    TERMS = pd.Series(["DC", "Mono", "Ery"],
                       index=["Run5_131097901611291", "Run5_134936662236454",
                                "Run4_200562869397916"])
    tsp = TERMS[TERMS.index.isin(a.obs_names)]
    palantir.core.run_palantir(a, START, num_waypoints=500,
                                 terminal_states=tsp)
    palantir.presults.select_branch_cells(a, q=0.01, eps=0.01,
                                            masks_key="branch_masks",
                                            save_as_df=True)
    counts = {b: int(a.obsm["branch_masks"][b].sum())
              for b in a.obsm["branch_masks"].columns}
    print(f"branch counts: {counts}")
    return a


def lagged_corr_hub(X, pt, lag_bins=5, n_bins=25):
    """For each gene j, sum |corr(x_i(τ), x_j(τ-lag))| over i."""
    order = np.argsort(pt)
    Xs = X[order]
    edges = np.linspace(0, len(order), n_bins + 1, dtype=int)
    Xbin = np.array([Xs[edges[b]:edges[b+1]].mean(axis=0) for b in range(n_bins)])
    x_now, x_past = Xbin[lag_bins:], Xbin[:-lag_bins]
    G = X.shape[1]; score = np.zeros(G)
    # Vectorized: corr matrix between all pairs
    xn = (x_now - x_now.mean(0)) / (x_now.std(0) + 1e-9)
    xp = (x_past - x_past.mean(0)) / (x_past.std(0) + 1e-9)
    corr_matrix = (xn.T @ xp) / (x_now.shape[0] - 1)   # (G, G) rows=now-i, cols=past-j
    return np.abs(corr_matrix).sum(axis=0)    # sum over i, per-j score


def main():
    a = preprocess_marrow_fig3()
    Xall = a.X.toarray() if sp.issparse(a.X) else a.X
    print(f"\nMarrow: {a.n_obs} × {a.n_vars}")

    results = {}
    for branch, tfs in BRANCH_TFS.items():
        mask = a.obsm["branch_masks"][branch].values.astype(bool)
        Xb = Xall[mask]
        ptb = a.obs["palantir_pseudotime"].values[mask].astype(np.float32)
        print(f"\n[{branch}] {mask.sum()} cells")

        score = lagged_corr_hub(Xb, ptb, lag_bins=5, n_bins=25)
        order = np.argsort(-score)
        ranked = [a.var_names[i] for i in order]
        ranks = {tf: (ranked.index(tf) + 1) if tf in ranked else None for tf in tfs}
        top10 = ranked[:10]
        top5 = ranked[:5]
        for tf in sorted(tfs):
            r = ranks[tf]
            in100 = "✓ top-100" if r is not None and r <= 100 else "✗"
            print(f"  {tf}: rank = {r}   {in100}")
        print(f"  top-5 lagged_corr_hub: {top5}")
        results[branch] = {"tf_ranks": ranks, "top10": top10,
                            "n_cells": int(mask.sum())}

    print("\n" + "="*66)
    print("Comparison to Round 3b scJDO's unique wins")
    print("="*66)
    print("Round 3b claim: GATA2 on Ery = scJDO_instab #10 (vs baselines #793+)")
    print(f"  → lagged_corr_hub GATA2 (Ery) rank: {results['Ery']['tf_ranks']['GATA2']}")
    print("Round 3b claim: IRF8 on Mono = scJDO_Jcol #7 (vs baselines #74+)")
    print(f"  → lagged_corr_hub IRF8 (Mono) rank: {results['Mono']['tf_ranks']['IRF8']}")

    (OUT / "check_B_lagged_corr_hematopoiesis.json").write_text(
        json.dumps(results, indent=2, default=str))
    print(f"\nWritten: {OUT / 'check_B_lagged_corr_hematopoiesis.json'}")


if __name__ == "__main__":
    main()
