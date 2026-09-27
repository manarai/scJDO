"""Gate 1 v2 build — per PREREG_Gate1_v2.md.
Reuses the 12,550-cell subsample; computes E_PCA / E_FA / E_scVI reps,
recomputes palantir + scJDO on X_FA, and saves gate1_v2_features.npz."""
from __future__ import annotations
import json, sys, warnings, time, pickle, subprocess
from pathlib import Path
from collections import Counter

import numpy as np, pandas as pd, torch
import scipy.sparse as sp
import scanpy as sc

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "New_analysis" / "gates_r26" / "gate1"))

from run_gate1_build import (
    H5AD_PATH_RAW, MYELOID_CELLTYPES,
    palantir_pseudotime, fit_scjdo_features,
    consensus_archetype_activation, classification_cohort,
    SCJDO_SEEDS, K_ARCH,
)

OUT = REPO / "New_analysis" / "gates_r26" / "gate1"
DATA = OUT / "data"
H5AD_PP_V2 = DATA / "larry_myeloid_preproc_v2.h5ad"
warnings.filterwarnings("ignore")


def prepare_substrate():
    """Load raw h5ad, apply same subsample (seed=0), preprocess, compute
    X_PCA + X_FA + X_scVI reps. Save + return AnnData."""
    if H5AD_PP_V2.exists():
        print(f"[build] loading v2 preproc cache {H5AD_PP_V2.name}")
        return sc.read_h5ad(H5AD_PP_V2)
    print(f"[build] loading raw {H5AD_PATH_RAW.name}")
    mye = sc.read_h5ad(H5AD_PATH_RAW)
    # SAME subsample as v1 (seed=0, same rules)
    rng = np.random.default_rng(0)
    has_clone = np.asarray(mye.obsm["X_clone"].sum(axis=1)).flatten() > 0
    d = mye.obs["Time point"].astype(int).values
    ct = mye.obs["Cell type annotation"].astype(str).values
    keep_d2_clone = (d == 2) & has_clone
    neut_idx = np.where((d != 2) & (ct == "Neutrophil"))[0]
    mono_idx = np.where((d != 2) & (ct == "Monocyte"))[0]
    neut_sample = rng.choice(neut_idx, size=min(2000, len(neut_idx)), replace=False)
    mono_sample = rng.choice(mono_idx, size=min(2000, len(mono_idx)), replace=False)
    keep_terminals = np.zeros(mye.n_obs, dtype=bool)
    keep_terminals[neut_sample] = True; keep_terminals[mono_sample] = True
    late_undiff_idx = np.where((d != 2) & (ct == "Undifferentiated"))[0]
    late_undiff_sample = rng.choice(late_undiff_idx,
                                     size=min(2000, len(late_undiff_idx)),
                                     replace=False)
    d2_undiff_idx = np.where((d == 2) & (ct == "Undifferentiated") & ~has_clone)[0]
    d2_undiff_sample = rng.choice(d2_undiff_idx,
                                   size=min(2000, len(d2_undiff_idx)),
                                   replace=False)
    keep_intermediates = np.zeros(mye.n_obs, dtype=bool)
    keep_intermediates[late_undiff_sample] = True
    keep_intermediates[d2_undiff_sample] = True
    keep = keep_d2_clone | keep_terminals | keep_intermediates
    print(f"[build] subsample: total={int(keep.sum())}")
    mye = mye[keep].copy()
    mye.obs["cell_type"] = mye.obs["Cell type annotation"].astype(str)
    mye.obs["day"] = mye.obs["Time point"].astype(int)

    # Keep raw counts layer (for scVI); log-normalise mye.X
    print("[build] normalize + log + HVG(2000)...")
    mye.layers["counts"] = mye.X.copy()
    sc.pp.normalize_total(mye, target_sum=1e4)
    sc.pp.log1p(mye)
    sc.pp.highly_variable_genes(mye, n_top_genes=2000, flavor="seurat",
                                 batch_key=None, subset=False)
    hvg = mye.var["highly_variable"].values
    mye_hvg = mye[:, hvg].copy()
    sc.pp.scale(mye_hvg, max_value=10, zero_center=True)
    X_hvg_scaled = mye_hvg.X
    if sp.issparse(X_hvg_scaled):
        X_hvg_scaled = X_hvg_scaled.toarray()
    X_hvg_scaled = X_hvg_scaled.astype(np.float32)
    print(f"[build] X_hvg_scaled shape: {X_hvg_scaled.shape} dtype={X_hvg_scaled.dtype}")

    # PCA30
    from sklearn.decomposition import PCA, FactorAnalysis
    print("[build] PCA(30) on scaled HVG...")
    pca = PCA(n_components=30, random_state=0)
    X_PCA = pca.fit_transform(X_hvg_scaled).astype(np.float32)
    print(f"  X_PCA={X_PCA.shape}")

    # FA30
    print("[build] FactorAnalysis(30) on scaled HVG...")
    fa = FactorAnalysis(n_components=30, random_state=0, max_iter=200)
    X_FA = fa.fit_transform(X_hvg_scaled).astype(np.float32)
    print(f"  X_FA={X_FA.shape}")
    # gene loadings for FA (n_hvg, 30) — for scJDO consumption
    fa_loadings_hvg = fa.components_.T.astype(np.float32)

    # scVI30 (on raw counts, NOT scaled/logged)
    print("[build] scVI setup + train (max_epochs=200)...")
    import scvi
    mye_vi = mye.copy()  # keep all genes; scVI can select HVG internally
    # scvi setup needs raw counts in .X or specific layer
    mye_vi.X = mye_vi.layers["counts"]
    scvi.model.SCVI.setup_anndata(mye_vi, layer=None)  # uses .X (raw)
    model = scvi.model.SCVI(mye_vi, n_latent=30, n_layers=2, n_hidden=128)
    t0 = time.time()
    model.train(max_epochs=200, early_stopping=True, batch_size=1024,
                accelerator="cpu", enable_progress_bar=False)
    print(f"  scVI train: {time.time()-t0:.1f}s")
    X_scVI = model.get_latent_representation().astype(np.float32)
    print(f"  X_scVI={X_scVI.shape}")

    mye.obsm["X_PCA"] = X_PCA
    mye.obsm["X_FA"] = X_FA
    mye.obsm["X_scVI"] = X_scVI
    # store fa_loadings on full var index (zero-pad non-HVG rows)
    fa_loadings_full = np.zeros((mye.n_vars, 30), dtype=np.float32)
    fa_loadings_full[hvg] = fa_loadings_hvg
    mye.varm["fa_loadings"] = fa_loadings_full
    # also mirror as "X_fa" (lowercase) so run_gate1_build.fit_scjdo_features works
    mye.obsm["X_fa"] = X_FA

    mye.write_h5ad(H5AD_PP_V2)
    print(f"[build] wrote {H5AD_PP_V2}  {mye.shape}")
    return mye


def main():
    print("=" * 80); print("Gate 1 v2 build"); print("=" * 80)
    t_all = time.time()
    mye = prepare_substrate()
    print(f"substrate: {mye.shape}")

    # Palantir on X_FA
    if "pseudotime" not in mye.obs or np.isnan(mye.obs["pseudotime"].values).all():
        palantir_pseudotime(mye, seed=0)
    else:
        print("[palantir] cached pseudotime present")
    tau = mye.obs["pseudotime"].values.astype(np.float32)
    print(f"  tau range: [{tau.min():.3f}, {tau.max():.3f}]")

    # Save Palantir + reps to preproc cache (subprocess reads it)
    mye.write_h5ad(H5AD_PP_V2)

    # scJDO features × 3 seeds on X_FA via subprocess
    HELPER = OUT / "_scjdo_one_seed_v2.py"
    per_seed_features = {}
    for seed in SCJDO_SEEDS:
        ckpt = OUT / f"gate1_v2_scjdo_seed{seed}.pkl"
        if ckpt.exists():
            print(f"\n[scjdo v2] seed={seed} loading cache")
            per_seed_features[seed] = pickle.loads(ckpt.read_bytes())
            continue
        print(f"\n[scjdo v2] seed={seed} → subprocess")
        t_s = time.time()
        try:
            subprocess.run([sys.executable, "-u", str(HELPER), str(seed)],
                            cwd=str(REPO), check=True, timeout=2700)
        except subprocess.TimeoutExpired:
            print(f"  seed {seed} TIMEOUT — skipping"); continue
        except subprocess.CalledProcessError as e:
            print(f"  seed {seed} FAILED: {e}"); continue
        if ckpt.exists():
            per_seed_features[seed] = pickle.loads(ckpt.read_bytes())
            print(f"  seed {seed} done in {time.time()-t_s:.1f}s")
    if not per_seed_features:
        raise RuntimeError("All scJDO seeds failed on v2")

    available_seeds = sorted(per_seed_features.keys())
    branches = list(per_seed_features[available_seeds[0]].keys())
    print(f"branches: {branches}, seeds: {available_seeds}")

    # Consensus per branch (within-LARRY across seeds)
    consensus_result = {}
    for branch in branches:
        per_seed = [per_seed_features[s][branch] for s in available_seeds]
        res = consensus_archetype_activation(per_seed, branch, mye.n_obs, tau)
        consensus_result[branch] = res
        print(f"[consensus] {branch}: "
              f"{'passing R=' + str(res['R_cluster']) if res else 'no passing cluster (dropped)'}")

    # S_FA per seed
    S_per_seed = {}
    for s in available_seeds:
        cols = []
        for branch in branches:
            f_re = per_seed_features[s][branch]["re_lambda_max"]
            f_lp = per_seed_features[s][branch]["lead_proj"]
            cols.append(f_re[:, None]); cols.append(f_lp[:, None])
            if consensus_result[branch] is not None:
                cols.append(consensus_result[branch]["activation_per_cell"][:, None])
        S_per_seed[s] = np.concatenate(cols, axis=1).astype(np.float32)

    # Cohort
    mask, cell_label, cell_group = classification_cohort(mye)
    print(f"[cohort] {mask.sum()} cells, {len(np.unique(cell_group[mask]))} clones")

    # Save
    save_kw = {
        "X_PCA": mye.obsm["X_PCA"].astype(np.float32),
        "X_FA": mye.obsm["X_FA"].astype(np.float32),
        "X_scVI": mye.obsm["X_scVI"].astype(np.float32),
        "cohort_mask": mask,
        "cohort_label": cell_label,
        "cohort_group": cell_group,
        "pseudotime": tau,
        "day": mye.obs["day"].values.astype(np.int32),
        "branches": np.array(branches, dtype=object),
        "available_seeds": np.array(available_seeds, dtype=np.int32),
    }
    for s in available_seeds:
        save_kw[f"S_FA_seed{s}"] = S_per_seed[s]
    np.savez(OUT / "gate1_v2_features.npz", **save_kw)
    print(f"\n[written] gate1_v2_features.npz  total {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
