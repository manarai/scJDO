"""r25 Check A — Does vel_scale=0 (A2) add anything beyond PCA on real Ery?

Compare A2's top-15 leading-direction loadings against:
  - Top-15 highly-variable genes (Seurat HVG rank)
  - Top-15 |PC1 loadings|
  - Top-15 |FA factor-1 loadings|

If A2's top-15 ≈ top-15 HVG or PC1 loadings, then scJDO under vel_scale=0
recovers the branch's main axis of variance (HBB dominates erythroid
because it's the top HVG), not any dynamics-specific signal.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "reproducibility" / "operator_claims_benchmark" / "scripts"))

from run_operator_claims import prepare_marrow_ery
from scjdo.tl import fit_drift_branches

OUT = REPO / "reproducibility" / "operator_claims_benchmark" / "outputs" / "r25_check_A"
OUT.mkdir(parents=True, exist_ok=True)


def get_a2_top15_gene_load(ery, seed=42):
    """Reproduce A2 (vel_scale=0, bias=1.5) and return top-15 gene loading."""
    import torch
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
            bias_strength=1.5,       # A2 keeps bias on
            n_archetypes=5, n_epochs=5000,
            vel_scale=0.0,            # ← A2's defining knob
            hidden=256, depth=4, sigma=0.10,
            windowing="kernel", bandwidth="auto", grid_size=200,
            seed=seed, verbose=False,
        )
    res = b.uns["scjdo_Ery"]
    J = np.asarray(res["J_tensor"])
    max_eig = np.asarray(res["max_real_eig"])
    t_peak = int(np.argmax(max_eig))
    eigvals, eigvecs = np.linalg.eig(J[t_peak])
    k_top = int(np.argmax(np.real(eigvals)))
    v_peak = np.real(eigvecs[:, k_top])
    L = ery.varm["fa_loadings"]
    gene_load = np.abs(L @ v_peak)
    top15 = np.argsort(-gene_load)[:15]
    return [ery.var_names[i] for i in top15]


def main():
    from sklearn.decomposition import PCA
    ery = prepare_marrow_ery()

    # Restrict to Ery cells only (the A2 fit uses only these cells; comparison
    # baselines should use the same cell set)
    # Wait — ery from prepare_marrow_ery IS already the Ery-branch subset
    print(f"Ery subset: {ery.shape}")

    # ── Baseline 1: top-15 HVGs by variance on Ery cells ──────────────────
    Xdense = ery.X.toarray() if sp.issparse(ery.X) else ery.X
    var = Xdense.var(axis=0)
    top15_var = np.argsort(-var)[:15]
    hvg_names = [ery.var_names[i] for i in top15_var]
    print(f"\nTop-15 highest-variance genes on Ery: {hvg_names}")

    # ── Baseline 2: PC1 (top-15 |loadings|) on Ery cells ───────────────────
    # Compute PCA on Ery HVGs to mirror A2's substrate
    hvg_mask = ery.var["highly_variable"].values if "highly_variable" in ery.var.columns else np.ones(ery.n_vars, dtype=bool)
    Xhvg = Xdense[:, hvg_mask]
    pca = PCA(n_components=5, random_state=42)
    pca.fit(Xhvg)
    # Project PC1 loadings back to full gene index
    pc1_load_hvg = np.abs(pca.components_[0])   # (n_hvg,)
    pc1_full = np.zeros(ery.n_vars)
    pc1_full[hvg_mask] = pc1_load_hvg
    top15_pc1 = np.argsort(-pc1_full)[:15]
    pc1_names = [ery.var_names[i] for i in top15_pc1]
    print(f"Top-15 |PC1 loading| on Ery: {pc1_names}")

    # ── Baseline 3: FA factor-1 (top-15 |loadings|) on Ery cells ────────────
    # Use the fa_loadings computed in prepare_marrow_ery (which was fit on
    # ALL cells, not just Ery). But FA factor 1 is the first component of
    # the FA embedding — for A2's leading eigenvector to "just be" FA-1,
    # the eigenvector at peak-τ would need to be a canonical basis vector.
    # A more honest baseline is to redo FA on Ery-only.
    from sklearn.decomposition import FactorAnalysis
    fa_ery = FactorAnalysis(n_components=5, random_state=42)
    fa_ery.fit(Xhvg)
    fa1_load_hvg = np.abs(fa_ery.components_[0])
    fa1_full = np.zeros(ery.n_vars)
    fa1_full[hvg_mask] = fa1_load_hvg
    top15_fa1 = np.argsort(-fa1_full)[:15]
    fa1_names = [ery.var_names[i] for i in top15_fa1]
    print(f"Top-15 |FA factor-1| (fit on Ery) loadings: {fa1_names}")

    # ── A2 top-15 (compute or reload) ──────────────────────────────────────
    print(f"\n[A2] Fitting scJDO with vel_scale=0.0, bias=1.5, seed=42, 5000 epochs...")
    a2_names = get_a2_top15_gene_load(ery, seed=42)
    print(f"A2 top-15 leading-direction loadings: {a2_names}")

    # ── Compare ────────────────────────────────────────────────────────────
    def _jacc(a, b):
        A, B = set(a), set(b)
        return len(A & B) / len(A | B) if A | B else 0.0

    print("\n" + "="*70)
    print("Jaccard vs A2 top-15")
    print("="*70)
    print(f"  top-15 HVG        vs A2: Jaccard = {_jacc(hvg_names, a2_names):.3f}"
          f"  |  overlap: {sorted(set(hvg_names) & set(a2_names))}")
    print(f"  top-15 |PC1|      vs A2: Jaccard = {_jacc(pc1_names, a2_names):.3f}"
          f"  |  overlap: {sorted(set(pc1_names) & set(a2_names))}")
    print(f"  top-15 |FA1-Ery|  vs A2: Jaccard = {_jacc(fa1_names, a2_names):.3f}"
          f"  |  overlap: {sorted(set(fa1_names) & set(a2_names))}")

    import json
    (OUT / "check_A_summary.json").write_text(json.dumps({
        "a2_top15":  a2_names,
        "hvg_top15": hvg_names,
        "pc1_top15": pc1_names,
        "fa1_ery_top15": fa1_names,
        "jaccards": {
            "hvg_vs_a2": _jacc(hvg_names, a2_names),
            "pc1_vs_a2": _jacc(pc1_names, a2_names),
            "fa1_ery_vs_a2": _jacc(fa1_names, a2_names),
        },
    }, indent=2))
    print(f"\nWritten: {OUT / 'check_A_summary.json'}")


if __name__ == "__main__":
    main()
