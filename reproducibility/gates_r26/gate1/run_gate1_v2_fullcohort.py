"""Gate 1 v2 full-cohort rerun (Task A) per PREREG_TaskA_Gate1_v2_fullcohort.md.
Day-2-only pipeline; ALL 28,249 day-2 LARRY cells; arms E_PCA, E_FA,
E_scVI, S_FA, E_FA + S_FA; clone-grouped 5x5 CV; 3 scJDO seeds."""
from __future__ import annotations
import json, sys, warnings, time, pickle, subprocess
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np, pandas as pd, torch
import scanpy as sc, anndata as ad
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "reproducibility" / "gates_r26" / "gate1"))
sys.path.insert(0, str(REPO / "reproducibility" / "gates_r26" / "gate3"))
warnings.filterwarnings("ignore")

# Import label-building and cohort logic from Gate 3 strict
from run_gate3_strict import (
    build_day2_h5ad,   # yields (adata_day2, clone_late) — 28,249 cells + clone fate matrix
    K_NEIGHBOURS, N_FA, N_HVG,
    FATE_MAP, MAJOR_GROUPS,
)
# Import consensus procedure + feature extraction from Gate 1 v2
from run_gate1_v2_build import (
    consensus_archetype_activation,
)
from run_gate1_build import fit_scjdo_features

OUT = REPO / "reproducibility" / "gates_r26" / "gate1"
DATA = OUT / "data"
DATA.mkdir(parents=True, exist_ok=True)
H5AD_PP_TA = DATA / "larry_day2_preproc_taskA.h5ad"
SCJDO_SEEDS = [0, 1, 2]


def build_features_taskA():
    """Return the day-2 AnnData with X_PCA, X_FA, X_scVI, X_fa alias,
    fa_loadings varm, palantir pseudotime, X_clone."""
    if H5AD_PP_TA.exists():
        print(f"[preproc] loading cached {H5AD_PP_TA.name}")
        return sc.read_h5ad(H5AD_PP_TA)

    adata_day2, _ = build_day2_h5ad()
    print(f"[preproc] day-2 raw AnnData: {adata_day2.shape}")

    # Preserve raw counts for scVI
    a = adata_day2.copy()
    a.layers["counts"] = a.X.copy()

    # normalize + log1p on day-2
    print("[preproc] normalize_total + log1p on day-2 ...")
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)

    # HVG on day-2
    print("[preproc] HVG(2000, seurat) ...")
    sc.pp.highly_variable_genes(a, n_top_genes=N_HVG, flavor="seurat",
                                 batch_key=None, subset=False)
    hvg = a.var["highly_variable"].values
    a_hvg = a[:, hvg].copy()

    # scale in HVG subspace
    print("[preproc] scale HVG ...")
    sc.pp.scale(a_hvg, max_value=10, zero_center=True)
    X_scaled = a_hvg.X
    if sp.issparse(X_scaled):
        X_scaled = X_scaled.toarray()
    X_scaled = X_scaled.astype(np.float32)
    print(f"  X_scaled: {X_scaled.shape}")

    # PCA30
    print("[preproc] PCA(30) ...")
    from sklearn.decomposition import PCA, FactorAnalysis
    pca = PCA(n_components=30, random_state=0)
    X_PCA = pca.fit_transform(X_scaled).astype(np.float32)
    print(f"  X_PCA: {X_PCA.shape}")

    # FA30 (X_FA)
    print("[preproc] FactorAnalysis(30) ...")
    fa = FactorAnalysis(n_components=30, random_state=0, max_iter=200)
    X_FA = fa.fit_transform(X_scaled).astype(np.float32)
    fa_load_hvg = fa.components_.T.astype(np.float32)
    print(f"  X_FA: {X_FA.shape}")

    # scVI30
    print("[preproc] scVI(30) on raw counts (max_epochs=200) ...")
    import scvi
    a_vi = adata_day2.copy()
    a_vi.X = a_vi.layers.get("counts", a_vi.X) if "counts" in a_vi.layers else adata_day2.X
    scvi.model.SCVI.setup_anndata(a_vi, layer=None)
    m = scvi.model.SCVI(a_vi, n_latent=30, n_layers=2, n_hidden=128)
    t0 = time.time()
    m.train(max_epochs=200, early_stopping=True, batch_size=1024,
            accelerator="cpu", enable_progress_bar=False)
    print(f"  scVI train: {time.time()-t0:.1f}s")
    X_scVI = m.get_latent_representation().astype(np.float32)
    print(f"  X_scVI: {X_scVI.shape}")

    # Assemble the preprocessed AnnData at HVG-subset shape
    # (so varm["fa_loadings"] has correct n_vars = n_hvg for scJDO)
    a = a[:, hvg].copy()
    a.obsm["X_PCA"] = X_PCA
    a.obsm["X_FA"] = X_FA
    a.obsm["X_scVI"] = X_scVI
    a.obsm["X_fa"] = X_FA   # lowercase alias for fit_scjdo_features
    a.varm["fa_loadings"] = fa_load_hvg
    # clone matrix is preserved from raw substrate load
    if "X_clone" in adata_day2.obsm:
        a.obsm["X_clone"] = adata_day2.obsm["X_clone"]

    # ── Amendment (revised): scJDO fit substrate = all 4,638 barcoded day-2
    #    cells + 5,000 random day-2 Undiff without clone = ~9,638 total.
    #    Prior 14K attempt OOM'd; this preserves 100% of the cohort while
    #    shrinking the fill to fit compute budget.
    has_clone = np.asarray(a.obsm["X_clone"].sum(axis=1)).flatten() > 0
    rng = np.random.default_rng(0)
    all_no_clone_idx = np.where(~has_clone)[0]
    n_fill = min(5000, len(all_no_clone_idx))
    fill_sample = rng.choice(all_no_clone_idx, size=n_fill, replace=False)
    keep_fit = has_clone.copy()
    keep_fit[fill_sample] = True
    print(f"[preproc] scJDO fit substrate subsample: total={int(keep_fit.sum())} "
          f"(has_clone={int(has_clone.sum())}, fill={n_fill})")
    a_fit = a[keep_fit].copy()

    # Palantir on the ~14K fit substrate's X_FA
    import palantir
    print("[preproc] palantir pseudotime on fit substrate X_FA ...")
    dm = palantir.utils.run_diffusion_maps(
        pd.DataFrame(a_fit.obsm["X_FA"], index=a_fit.obs_names), n_components=15,
    )
    ms = palantir.utils.determine_multiscale_space(dm)
    obs_meta = a_fit.obs
    spring_x = obs_meta["SPRING-x"].astype(float).values if "SPRING-x" in obs_meta \
        else np.zeros(a_fit.n_obs)
    start_cell = a_fit.obs_names[int(np.argmin(spring_x))]
    pr = palantir.core.run_palantir(
        ms, start_cell, num_waypoints=1000, seed=0,
        use_early_cell_as_start=False,
    )
    a_fit.obs["pseudotime"] = pr.pseudotime.reindex(a_fit.obs_names).values.astype(np.float32)
    a_fit.obsm["branch_probs"] = np.ones((a_fit.n_obs, 1), dtype=np.float32)
    a_fit.uns["branch_names"] = ["Ery"]
    print(f"  fit substrate pseudotime range: [{np.nanmin(a_fit.obs['pseudotime']):.3f}, "
          f"{np.nanmax(a_fit.obs['pseudotime']):.3f}]")

    # Write TWO AnnDatas: the full 28k baseline (for E arms) and the fit substrate
    a.write_h5ad(H5AD_PP_TA)
    a_fit.write_h5ad(H5AD_PP_TA.parent / "larry_day2_preproc_taskA_fit.h5ad")
    print(f"[preproc] wrote {H5AD_PP_TA}  {a.shape}  (full-cohort baseline)")
    print(f"[preproc] wrote {H5AD_PP_TA.parent / 'larry_day2_preproc_taskA_fit.h5ad'}  "
          f"{a_fit.shape}  (scJDO fit substrate)")
    return a


def cohort_labels(adata_day2, clone_late):
    """Build the 759-cell / 494-clone classification cohort per Gate 1 v2
    purity rule: >= 2 late Neu union Mono cells per clone, purity >= 80%."""
    # clone_late is not passed in this signature; refetch clone_late from raw
    # We rebuild it here for full transparency + independence.
    C = adata_day2.obsm["X_clone"]  # sparse (n_day2, n_clones)
    n_clones = C.shape[1]
    # We need late-time fate outcomes per clone — pull from the raw h5ad
    # loaded once by build_day2_h5ad (it writes clone_late.parquet)
    clone_late_path = REPO / "reproducibility" / "gates_r26" / "gate3" / "data" / "clone_late_fate.parquet"
    clone_late_df = pd.read_parquet(clone_late_path)
    neut_clones, mono_clones = set(), set()
    for _, row in clone_late_df.iterrows():
        cid = int(row["clone_id"])
        n_ct = int(row.get("Neutrophil", 0))
        m_ct = int(row.get("Monocyte", 0))
        if n_ct + m_ct < 2: continue
        frac_n = n_ct / (n_ct + m_ct)
        if frac_n >= 0.8:
            neut_clones.add(cid)
        elif frac_n <= 0.2:
            mono_clones.add(cid)
    print(f"[cohort] clones: neut-biased={len(neut_clones)}, mono-biased={len(mono_clones)}")

    Ccsr = C.tocsr()
    cell_label = np.full(adata_day2.n_obs, -1, dtype=np.int8)
    cell_group = np.full(adata_day2.n_obs, -1, dtype=np.int64)
    for cell_idx in range(adata_day2.n_obs):
        cols = Ccsr.indices[Ccsr.indptr[cell_idx]:Ccsr.indptr[cell_idx + 1]]
        for cid in cols:
            if cid in neut_clones:
                cell_label[cell_idx] = 1; cell_group[cell_idx] = cid; break
            elif cid in mono_clones:
                cell_label[cell_idx] = 0; cell_group[cell_idx] = cid; break
    mask = cell_label >= 0
    print(f"[cohort] labelled cells: {mask.sum()} (neut={int((cell_label == 1).sum())}, "
          f"mono={int((cell_label == 0).sum())}), unique clones: {len(np.unique(cell_group[mask]))}")
    return mask, cell_label, cell_group


def main():
    print("=" * 80); print("Task A — Gate 1 v2 full-cohort rerun"); print("=" * 80)
    t_all = time.time()
    a = build_features_taskA()
    print(f"substrate: {a.shape}")

    # scJDO features x 3 seeds via subprocess helper (from Gate 1 v2)
    HELPER = OUT / "_scjdo_one_seed_taskA.py"
    per_seed = {}
    for seed in SCJDO_SEEDS:
        ck = OUT / f"gate1_v2_taskA_scjdo_seed{seed}.pkl"
        if ck.exists():
            print(f"\n[scjdo] seed={seed} cache hit")
            per_seed[seed] = pickle.loads(ck.read_bytes())
            continue
        print(f"\n[scjdo] seed={seed} -> subprocess")
        t_s = time.time()
        try:
            subprocess.run([sys.executable, "-u", str(HELPER), str(seed)],
                            cwd=str(REPO), check=True, timeout=3600)  # 60 min cap
        except subprocess.TimeoutExpired:
            print(f"  seed {seed} TIMEOUT — skipping"); continue
        except subprocess.CalledProcessError as e:
            print(f"  seed {seed} FAILED: {e}"); continue
        if ck.exists():
            per_seed[seed] = pickle.loads(ck.read_bytes())
            print(f"  seed {seed}: {time.time()-t_s:.1f}s")
    if not per_seed:
        raise RuntimeError("All scJDO seeds failed")
    seeds = sorted(per_seed.keys())
    branches = list(per_seed[seeds[0]].keys())
    print(f"[scjdo] branches={branches}, seeds={seeds}")

    # Consensus (single-branch, within-substrate across seeds)
    tau = a.obs["pseudotime"].values.astype(np.float32)
    consensus_result = {}
    for br in branches:
        ps = [per_seed[s][br] for s in seeds]
        r = consensus_archetype_activation(ps, br, a.n_obs, tau)
        consensus_result[br] = r
        print(f"[consensus] {br}: {('passing R=' + str(r['R_cluster'])) if r else 'no passing cluster'}")

    # S_FA per seed
    S_per_seed = {}
    for s in seeds:
        cols = []
        for br in branches:
            cols.append(per_seed[s][br]["re_lambda_max"][:, None])
            cols.append(per_seed[s][br]["lead_proj"][:, None])
            if consensus_result[br] is not None:
                cols.append(consensus_result[br]["activation_per_cell"][:, None])
        S_per_seed[s] = np.concatenate(cols, axis=1).astype(np.float32)
    if seeds:
        print(f"S_FA per-seed shape: {S_per_seed[seeds[0]].shape}")

    # Cohort
    mask, cell_label, cell_group = cohort_labels(a, None)

    # Save features (mask-aligned + full-array versions)
    save_kw = {
        "X_PCA": a.obsm["X_PCA"].astype(np.float32),
        "X_FA": a.obsm["X_FA"].astype(np.float32),
        "X_scVI": a.obsm["X_scVI"].astype(np.float32),
        "cohort_mask": mask,
        "cohort_label": cell_label,
        "cohort_group": cell_group,
        "pseudotime": tau,
        "branches": np.array(branches, dtype=object),
        "available_seeds": np.array(seeds, dtype=np.int32),
    }
    for s in seeds:
        save_kw[f"S_FA_seed{s}"] = S_per_seed[s]
    np.savez(OUT / "gate1_v2_taskA_features.npz", **save_kw)
    print(f"\n[written] gate1_v2_taskA_features.npz  total {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
