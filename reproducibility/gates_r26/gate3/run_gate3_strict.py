"""Gate 3 strict — single entry point, deterministic per PREREG_Gate3_strict.md.
Tests whether local-covariance softening at day 2 predicts clone commitment
BEYOND expression on the LARRY in vitro release. NO scJDO fitted-field feature.
Day-2 cells only for feature construction; day-4/6 used only for label."""
from __future__ import annotations
import gzip, json, sys, warnings, time
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmread
from scipy.stats import spearmanr
import scanpy as sc
import anndata as ad

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
OUT = REPO / "reproducibility" / "gates_r26" / "gate3"
DATA = OUT / "data"
DATA.mkdir(parents=True, exist_ok=True)
LARRY_DIR = Path("/tmp/larry_data")

warnings.filterwarnings("ignore")

# ---------- frozen constants ----------
K_NEIGHBOURS = 50
N_FA = 30
N_HVG = 2000
N_RAREFY = 200
MIN_DESCENDANTS = 5
CV_FOLDS = 5
CV_REPEATS = 5
CV_SEEDS = [42, 43, 44, 45, 46]
N_BOOT = 2000

FATE_MAP = {
    "Neutrophil":     "Myeloid_Neu",
    "Monocyte":       "Myeloid_Mono",
    "Erythroid":      "ErythroMeg",
    "Meg":            "ErythroMeg",
    "Baso":           "MastGranu",
    "Mast":           "MastGranu",
    "Eos":            "MastGranu",
    "Lymphoid":       "LymphoDC",
    "Ccr7_DC":        "LymphoDC",
    "pDC":            "LymphoDC",
}
MAJOR_GROUPS = sorted(set(FATE_MAP.values()))


# ============================================================================
#  Phase 1 — day-2 subset build (cached in day2.h5ad)
# ============================================================================
def build_day2_h5ad():
    ck = DATA / "larry_day2.h5ad"
    ck_late = DATA / "clone_late_fate.parquet"
    if ck.exists() and ck_late.exists():
        print(f"[phase1] loading cached {ck.name} + {ck_late.name}")
        return ad.read_h5ad(ck), pd.read_parquet(ck_late)

    print("[phase1] loading Klein LARRY raw files ...")
    t0 = time.time()
    plain = LARRY_DIR / "stateFate_inVitro_normed_counts.mtx"
    if plain.exists():
        X = mmread(str(plain)).tocsr().astype(np.float32)
    else:
        with gzip.open(LARRY_DIR / "stateFate_inVitro_normed_counts.mtx.gz", "rt") as f:
            X = mmread(f).tocsr().astype(np.float32)
    print(f"  counts {X.shape} in {time.time()-t0:.1f}s")
    genes = pd.read_csv(LARRY_DIR / "stateFate_inVitro_gene_names.txt.gz",
                         sep="\t", header=None)[0].values
    meta = pd.read_csv(LARRY_DIR / "stateFate_inVitro_metadata.txt.gz", sep="\t")
    with gzip.open(LARRY_DIR / "stateFate_inVitro_clone_matrix.mtx.gz", "rt") as f:
        C = mmread(f).tocsr()
    print(f"  meta {meta.shape}, genes {len(genes)}, clone matrix {C.shape}")

    # day-2 selection
    day = meta["Time point"].astype(float).values
    day2_mask = np.isclose(day, 2.0)
    late_mask = np.isin(day, [4.0, 6.0])
    n_total = int(len(meta))
    n_day2 = int(day2_mask.sum())
    n_late = int(late_mask.sum())
    print(f"  n_total={n_total}, n_day2={n_day2}, n_late={n_late}")

    # clone → late-fate outcome matrix (n_clones × 10 raw categories)
    fate = meta["Cell type annotation"].astype(str).values
    Ccoo = C.tocoo()
    clone_fate_counts = defaultdict(lambda: Counter())
    for r, cc in zip(Ccoo.row, Ccoo.col):
        if late_mask[r] and fate[r] in FATE_MAP:
            clone_fate_counts[int(cc)][fate[r]] += 1
    n_clones = C.shape[1]
    fate_rows = []
    for clone_id in range(n_clones):
        row = {"clone_id": clone_id}
        for cat in FATE_MAP:
            row[cat] = int(clone_fate_counts.get(clone_id, {}).get(cat, 0))
        for grp in MAJOR_GROUPS:
            row[grp] = 0
        for cat, cnt in clone_fate_counts.get(clone_id, {}).items():
            row[FATE_MAP[cat]] += cnt
        row["n_descendants_mature"] = int(sum(clone_fate_counts.get(clone_id, {}).values()))
        fate_rows.append(row)
    clone_late = pd.DataFrame(fate_rows)
    clone_late.to_parquet(ck_late)
    print(f"  wrote {ck_late}  clones with any mature desc: "
          f"{(clone_late['n_descendants_mature'] > 0).sum()}")

    # subset day-2 X + clone matrix
    X_d2 = X[day2_mask].tocsr()
    C_d2 = C[day2_mask].tocsr()
    del X, C
    import gc; gc.collect()

    obs = meta[day2_mask].reset_index(drop=True).copy()
    obs.index = pd.Index([f"cell_{i}" for i in np.where(day2_mask)[0]])
    obs["cell_type"] = obs["Cell type annotation"].astype(str)
    obs["day"] = 2

    a = ad.AnnData(X=X_d2, obs=obs,
                    var=pd.DataFrame(index=pd.Index(genes, name="gene")))
    a.obsm["X_clone"] = C_d2
    a.uns["cohort_report"] = {
        "n_total": n_total, "n_day2": n_day2, "n_late": n_late,
        "n_day2_with_clone": int((np.asarray(C_d2.sum(axis=1)).flatten() > 0).sum()),
    }
    a.write_h5ad(ck)
    print(f"  wrote {ck}  {a.shape}  ({time.time()-t0:.1f}s total)")
    return a, clone_late


# ============================================================================
#  Phase 2 — labels per clone
# ============================================================================
def build_labels(clone_late):
    print("\n[phase2] labels — rarefy 5 descendants x N_RAREFY iterations ...")
    rows = []
    for _, row in clone_late.iterrows():
        counts_by_grp = {g: int(row[g]) for g in MAJOR_GROUPS}
        n_desc = int(row["n_descendants_mature"])
        if n_desc < MIN_DESCENDANTS:
            continue
        # multiset of major-group labels
        pool = []
        for g in MAJOR_GROUPS:
            pool.extend([g] * counts_by_grp[g])
        assert len(pool) == n_desc
        rng = np.random.default_rng(int(row["clone_id"]) * 7919)
        mixed_iters = np.zeros(N_RAREFY, dtype=np.float32)
        entropy_iters = np.zeros(N_RAREFY, dtype=np.float32)
        pool_arr = np.array(pool)
        for it in range(N_RAREFY):
            idx = rng.choice(n_desc, size=MIN_DESCENDANTS, replace=False)
            sampled = pool_arr[idx]
            uniq, cts = np.unique(sampled, return_counts=True)
            mixed_iters[it] = 1.0 if len(uniq) >= 2 else 0.0
            p = cts / cts.sum()
            entropy_iters[it] = -float((p * np.log(p + 1e-12)).sum())
        p_mixed = float(mixed_iters.mean())
        H = float(entropy_iters.mean())
        rows.append({"clone_id": int(row["clone_id"]),
                     "n_descendants": n_desc,
                     "p_mixed": p_mixed,
                     "y_mixed": int(p_mixed >= 0.5),
                     "H_entropy": H})
    df = pd.DataFrame(rows)
    print(f"  eligible clones: {len(df)}; positive fraction: {df['y_mixed'].mean():.3f}")
    print(f"  descendants: mean={df['n_descendants'].mean():.1f} "
          f"median={df['n_descendants'].median():.0f} "
          f"min={df['n_descendants'].min()} max={df['n_descendants'].max()}")
    return df


# ============================================================================
#  Phase 3 — features per cell (E, C, Ic) + aggregate per clone
# ============================================================================
def compute_features(adata_day2):
    ck_feat_cell = DATA / "features_per_cell.parquet"
    if ck_feat_cell.exists():
        print(f"[phase3] loading cached {ck_feat_cell.name}")
        return pd.read_parquet(ck_feat_cell)

    a = adata_day2.copy()
    print(f"\n[phase3] E preprocessing on day-2 substrate {a.shape} ...")
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=N_HVG, flavor="seurat",
                                 batch_key=None, subset=False)
    hvg_mask = a.var["highly_variable"].values
    a = a[:, hvg_mask].copy()
    print(f"  after HVG: {a.shape}")
    sc.pp.scale(a, max_value=10, zero_center=True)
    X_scaled = a.X.astype(np.float32)
    if sp.issparse(X_scaled):
        X_scaled = X_scaled.toarray()
    print(f"  X_scaled dense: {X_scaled.shape}  dtype={X_scaled.dtype}")

    # FA30
    print("[phase3] FactorAnalysis(30) ...")
    t0 = time.time()
    from sklearn.decomposition import FactorAnalysis
    fa = FactorAnalysis(n_components=N_FA, random_state=0, max_iter=200)
    X_FA = fa.fit_transform(X_scaled).astype(np.float32)
    print(f"  X_FA={X_FA.shape}  fit time {time.time()-t0:.1f}s")

    # k-NN k=50 in FA space
    print(f"[phase3] k-NN k={K_NEIGHBOURS} in FA{N_FA} space ...")
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=K_NEIGHBOURS + 1, algorithm="brute")
    nn.fit(X_FA)
    _, nn_idx = nn.kneighbors(X_FA)
    nn_idx = nn_idx[:, 1:]  # drop self
    print(f"  nn_idx={nn_idx.shape}")

    # C — softness ratio + top-3 eigenvalues of local covariance
    # Gram trick: for X_nbr shape (k, d), the non-zero eigvals of
    # (X_nbr^T X_nbr)/(k-1) equal those of (X_nbr X_nbr^T)/(k-1),
    # a (k, k) matrix that is cheap to eigendecompose.
    # Ridge on the (d, d) form adds lambda to every eigval; we track ratio.
    print("[phase3] Σ_local top-3 eigenvalues via Gram trick ...")
    t0 = time.time()
    N_cell, D = X_scaled.shape
    eig_top3 = np.zeros((N_cell, 3), dtype=np.float32)
    for i in range(N_cell):
        Xn = X_scaled[nn_idx[i]]  # (k, d)
        # ridge lambda from raw diagonal trace
        Gram = (Xn @ Xn.T).astype(np.float64) / (K_NEIGHBOURS - 1)
        # equivalent (d, d) trace: sum(diag(X_nbr^T @ X_nbr))/n = sum(x^2)/(n-1)
        raw_trace_per_gene = float(np.sum(Xn * Xn)) / (K_NEIGHBOURS - 1)
        lam = 1e-3 * raw_trace_per_gene / D
        # eigvals of the (k, k) Gram are the same as top-k eigvals of the
        # (d, d) unregularised covariance. Adding ridge lambda*I_d shifts
        # ALL d eigvals of the (d, d) form by lambda. Store the top-3 of
        # the (d, d) form: top-3 of Gram (unregularised) + lam
        w = np.linalg.eigvalsh(Gram)
        top3 = w[-3:][::-1] + lam
        eig_top3[i] = top3.astype(np.float32)
        if (i + 1) % 5000 == 0:
            print(f"    {i+1}/{N_cell}  elapsed {time.time()-t0:.1f}s")
    print(f"  Σ_local top-3 done in {time.time()-t0:.1f}s")

    # Σ_global top-3 via TruncatedSVD
    from sklearn.decomposition import TruncatedSVD
    print("[phase3] Σ_global top-3 via TruncatedSVD ...")
    svd = TruncatedSVD(n_components=3, random_state=0)
    svd.fit(X_scaled)
    # eigvals of X.T X / (n-1) = singular_values^2 / (n-1)
    sig = svd.singular_values_
    global_top3 = (sig ** 2) / (N_cell - 1)
    # add ridge (same lambda scheme applied to global)
    global_raw_trace = float((X_scaled ** 2).sum()) / (N_cell - 1) / D
    global_lam = 1e-3 * global_raw_trace
    global_top3_reg = global_top3 + global_lam
    print(f"  Σ_global top-3 (regularised): {global_top3_reg}")

    softness_ratio = (eig_top3[:, 0] / global_top3_reg[0]).astype(np.float32)

    # Ic — Mojtahedi index over 50-NN, per cell, on HVG COUNTS (unnormalised)
    print(f"[phase3] Ic Mojtahedi index over {K_NEIGHBOURS}-NN ...")
    # HVG counts: pull from the original ORIGINAL day-2 counts (pre-normalisation)
    # since a.X now holds scaled data. Re-load from adata_day2 subsetted by HVG.
    hvg_gene_idx = np.where(hvg_mask)[0]
    X_counts_hvg = adata_day2.X[:, hvg_gene_idx]
    if sp.issparse(X_counts_hvg):
        X_counts_hvg = X_counts_hvg.toarray()
    X_counts_hvg = X_counts_hvg.astype(np.float32)
    print(f"  X_counts_hvg={X_counts_hvg.shape}")
    Ic = np.zeros(N_cell, dtype=np.float32)
    t0 = time.time()
    for i in range(N_cell):
        Z = X_counts_hvg[nn_idx[i]]  # (k=50, d=2000)
        # column-centre + scale to unit std per gene → correlation via Gram
        mu_g = Z.mean(axis=0, keepdims=True)
        Zc = Z - mu_g
        sd_g = np.sqrt((Zc * Zc).sum(axis=0) / max(K_NEIGHBOURS - 1, 1)) + 1e-12
        Zg = Zc / sd_g[None, :]  # (k, d) unit-var columns
        # gene-gene corr matrix: Zg.T @ Zg / (k-1)
        # We only need mean of |off-diagonal|
        Cgene = (Zg.T @ Zg) / (K_NEIGHBOURS - 1)  # (d, d)
        # exclude diagonal (should be ~1)
        np.fill_diagonal(Cgene, 0.0)
        R_gene = float(np.abs(Cgene).sum() / (D * (D - 1)))
        # cell-cell corr on same rows (transpose): (k, k) via row-normalised Z
        mu_c = Z.mean(axis=1, keepdims=True)
        Zc2 = Z - mu_c
        sd_c = np.sqrt((Zc2 * Zc2).sum(axis=1) / max(D - 1, 1)) + 1e-12
        Zc_n = Zc2 / sd_c[:, None]
        Ccell = (Zc_n @ Zc_n.T) / (D - 1)
        np.fill_diagonal(Ccell, 0.0)
        R_cell = float(Ccell.sum() / (K_NEIGHBOURS * (K_NEIGHBOURS - 1)))
        Ic[i] = R_gene / (R_cell + 1e-12) if abs(R_cell) > 1e-12 else np.nan
        if (i + 1) % 2000 == 0:
            print(f"    {i+1}/{N_cell}  Ic elapsed {time.time()-t0:.1f}s")
    print(f"  Ic done in {time.time()-t0:.1f}s")

    # Palantir pseudotime for sanity check (day-2 only)
    try:
        import palantir
        print("[phase3] palantir pseudotime on day-2 FA space (sanity control) ...")
        pt_seed = 0
        a2 = a.copy()
        a2.obsm["X_fa"] = X_FA
        # simplest anchor: min-SPRING-x cell
        obs_meta = adata_day2.obs
        # SPRING-x is in the metadata
        spring_x = obs_meta["SPRING-x"].values.astype(float) \
            if "SPRING-x" in obs_meta else np.zeros(N_cell)
        start_cell = a2.obs_names[int(np.argmin(spring_x))]
        import scanpy as sc2
        sc2.pp.neighbors(a2, use_rep="X_fa", n_neighbors=30, random_state=pt_seed)
        dm = palantir.utils.run_diffusion_maps(
            pd.DataFrame(X_FA, index=a2.obs_names), n_components=15,
        )
        ms = palantir.utils.determine_multiscale_space(dm)
        pr = palantir.core.run_palantir(
            ms, start_cell, num_waypoints=800, seed=pt_seed,
            use_early_cell_as_start=False,
        )
        pseudotime = pr.pseudotime.reindex(a2.obs_names).values.astype(np.float32)
        print(f"  pseudotime range=[{np.nanmin(pseudotime):.3f}, {np.nanmax(pseudotime):.3f}]")
    except Exception as e:
        print(f"  palantir failed: {e} — pseudotime set to NaN")
        pseudotime = np.full(N_cell, np.nan, dtype=np.float32)

    # Cell-cycle score
    try:
        import scanpy as sc2
        print("[phase3] cell-cycle score (Tirosh 2016) ...")
        a2 = adata_day2.copy()
        sc2.pp.normalize_total(a2, target_sum=1e4); sc2.pp.log1p(a2)
        s_genes = ("MCM5,PCNA,TYMS,FEN1,MCM2,MCM4,RRM1,UNG,GINS2,MCM6,CDCA7,DTL,PRIM1,"
                    "UHRF1,MLF1IP,HELLS,RFC2,RPA2,NASP,RAD51AP1,GMNN,WDR76,SLBP,"
                    "CCNE2,UBR7,POLD3,MSH2,ATAD2,RAD51,RRM2,CDC45,CDC6,EXO1,TIPIN,"
                    "DSCC1,BLM,CASP8AP2,USP1,CLSPN,POLA1,CHAF1B,BRIP1,E2F8").split(",")
        g2m_genes = ("HMGB2,CDK1,NUSAP1,UBE2C,BIRC5,TPX2,TOP2A,NDC80,CKS2,NUF2,CKS1B,"
                     "MKI67,TMPO,CENPF,TACC3,FAM64A,SMC4,CCNB2,CKAP2L,CKAP2,AURKB,"
                     "BUB1,KIF11,ANP32E,TUBB4B,GTSE1,KIF20B,HJURP,CDCA3,HN1,CDC20,"
                     "TTK,CDC25C,KIF2C,RANGAP1,NCAPD2,DLGAP5,CDCA2,CDCA8,ECT2,KIF23,"
                     "HMMR,AURKA,PSRC1,ANLN,LBR,CKAP5,CENPE,CTCF,NEK2,G2E3,GAS2L3,"
                     "CBX5,CENPA").split(",")
        # match case-insensitively
        var_upper = a2.var_names.str.upper()
        s_present = [g for g in s_genes if g in set(var_upper)]
        g2m_present = [g for g in g2m_genes if g in set(var_upper)]
        # rename lowercase to uppercase temporarily
        a2.var_names = var_upper
        sc2.tl.score_genes_cell_cycle(a2, s_genes=s_present, g2m_genes=g2m_present)
        cc_score = (a2.obs["S_score"].values + a2.obs["G2M_score"].values).astype(np.float32)
        print(f"  cc_score summary: mean={float(cc_score.mean()):.3f}  "
              f"n_S_genes={len(s_present)}  n_G2M_genes={len(g2m_present)}")
    except Exception as e:
        print(f"  cell-cycle scoring failed: {e}")
        cc_score = np.full(N_cell, np.nan, dtype=np.float32)

    # Assemble per-cell dataframe
    E_cols = {f"E_{k}": X_FA[:, k] for k in range(N_FA)}
    df = pd.DataFrame({
        "cell_index": np.arange(N_cell),
        "softness_ratio": softness_ratio,
        "sigma_local_top1": eig_top3[:, 0],
        "sigma_local_top2": eig_top3[:, 1],
        "sigma_local_top3": eig_top3[:, 2],
        "Ic": Ic,
        "pseudotime": pseudotime,
        "cc_score": cc_score,
        **E_cols,
    })
    df.to_parquet(ck_feat_cell)
    print(f"[phase3] wrote {ck_feat_cell}  {df.shape}")
    return df


# ============================================================================
#  Phase 4 — aggregate per clone
# ============================================================================
def aggregate_per_clone(features_cell, adata_day2, labels_df):
    print("\n[phase4] aggregating features per clone (mean over day-2 cells) ...")
    C_day2 = adata_day2.obsm["X_clone"].tocsc()  # (n_day2, n_clones)
    per_cell = features_cell.set_index("cell_index").sort_index()
    n_cells = per_cell.shape[0]
    rows = []
    for _, lab in labels_df.iterrows():
        clone_id = int(lab["clone_id"])
        # day-2 cells carrying this clone
        col = C_day2[:, clone_id]
        cell_idx = col.nonzero()[0]
        if len(cell_idx) == 0:
            continue
        agg = per_cell.iloc[cell_idx].mean(numeric_only=True)
        row = {"clone_id": clone_id,
                "n_day2_cells_in_clone": len(cell_idx),
                "n_descendants": int(lab["n_descendants"]),
                "p_mixed": float(lab["p_mixed"]),
                "y_mixed": int(lab["y_mixed"]),
                "H_entropy": float(lab["H_entropy"]),
                "log_n_desc": float(np.log1p(lab["n_descendants"]))}
        for k in per_cell.columns:
            row[k] = float(agg[k])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_parquet(DATA / "features_per_clone.parquet")
    print(f"[phase4] wrote features_per_clone.parquet {df.shape}")
    return df


# ============================================================================
#  Phase 5 — sanity checks
# ============================================================================
def sanity_checks(features_cell, per_clone):
    print("\n[phase5] SANITY CHECKS (before primary evaluation)")
    checks = {}

    # 1. ID alignment
    assert per_clone["clone_id"].is_unique
    print(f"  [ID align] n_clones={len(per_clone)}  unique clone_ids OK")
    checks["n_clones"] = int(len(per_clone))
    checks["clone_ids_unique"] = True

    # 2. Positive control: C, Ic vs pseudotime (day-2 pseudotime, per cell)
    pt = features_cell["pseudotime"].values
    if not np.isnan(pt).all():
        rho_c = float(spearmanr(features_cell["softness_ratio"].values, pt,
                                 nan_policy="omit").correlation)
        rho_ic = float(spearmanr(features_cell["Ic"].values, pt,
                                  nan_policy="omit").correlation)
        print(f"  [pt control] Spearman(softness_ratio, pt) = {rho_c:+.3f}")
        print(f"  [pt control] Spearman(Ic, pt) = {rho_ic:+.3f}")
        checks["spearman_softness_vs_pt"] = rho_c
        checks["spearman_Ic_vs_pt"] = rho_ic
    else:
        print("  [pt control] pseudotime unavailable")
        checks["spearman_softness_vs_pt"] = None; checks["spearman_Ic_vs_pt"] = None

    # 3. Non-trivial variance
    for c in ["softness_ratio", "Ic"]:
        v = per_clone[c].values
        checks[f"{c}_clone_dist"] = {"mean": float(np.nanmean(v)),
                                       "std": float(np.nanstd(v)),
                                       "min": float(np.nanmin(v)),
                                       "max": float(np.nanmax(v)),
                                       "q05": float(np.nanpercentile(v, 5)),
                                       "q95": float(np.nanpercentile(v, 95))}
        print(f"  [{c} across clones] "
              f"mean={checks[f'{c}_clone_dist']['mean']:+.3f} "
              f"std={checks[f'{c}_clone_dist']['std']:.3f} "
              f"range=[{checks[f'{c}_clone_dist']['min']:.3f}, "
              f"{checks[f'{c}_clone_dist']['max']:.3f}]")

    # 4. Confound with N
    N = per_clone["log_n_desc"].values
    for c in ["softness_ratio", "Ic", "cc_score"]:
        v = per_clone[c].values
        rho = float(spearmanr(v, N, nan_policy="omit").correlation)
        checks[f"spearman_{c}_vs_log_n_desc"] = rho
        print(f"  [{c} vs log_n_desc] Spearman = {rho:+.3f}")

    return checks


# ============================================================================
#  Phase 6 — model evaluation
# ============================================================================
def evaluate_arms(per_clone):
    print("\n[phase6] arm evaluation (5-fold GroupKFold x 5 fold seeds) ...")
    from sklearn.linear_model import LogisticRegressionCV, RidgeCV
    from sklearn.model_selection import GroupKFold
    from sklearn.utils import shuffle
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, log_loss

    E_cols = [c for c in per_clone.columns if c.startswith("E_")]
    C_cols = ["softness_ratio", "sigma_local_top1",
              "sigma_local_top2", "sigma_local_top3"]
    Ic_cols = ["Ic"]
    N_cols = ["log_n_desc"]

    arms = {
        "arm1_E":     E_cols + N_cols,
        "arm2_E+C":   E_cols + C_cols + N_cols,
        "arm3_E+Ic":  E_cols + Ic_cols + N_cols,
        "arm4_E+C+Ic": E_cols + C_cols + Ic_cols + N_cols,
    }
    y = per_clone["y_mixed"].values
    groups = per_clone["clone_id"].values
    p_mixed = per_clone["p_mixed"].values
    H_ent = per_clone["H_entropy"].values

    results = {}  # arm → dict with per-fold predictions + metrics
    for arm_name, cols in arms.items():
        X = per_clone[cols].values.astype(np.float32)
        # Per-clone predictions accumulated across 5 seeds x 5 folds
        preds_all = []  # (n_seeds, n_clones) probability
        auroc_list = []; logloss_list = []; rho_p_list = []; rho_H_list = []
        for cv_seed in CV_SEEDS:
            uniq = np.unique(groups)
            perm = shuffle(np.arange(len(uniq)), random_state=cv_seed)
            reordered = uniq[perm]
            mp = {g: i for i, g in enumerate(reordered)}
            g2 = np.array([mp[g] for g in groups])
            gkf = GroupKFold(n_splits=CV_FOLDS)
            preds_this_seed = np.zeros(len(y))
            preds_p = np.zeros(len(y)); preds_H = np.zeros(len(y))
            for tr, te in gkf.split(X, y, g2):
                sc = StandardScaler().fit(X[tr])
                Xtr = sc.transform(X[tr]); Xte = sc.transform(X[te])
                inner_gkf = GroupKFold(n_splits=5)
                inner_groups = g2[tr]
                clf = LogisticRegressionCV(
                    Cs=np.logspace(-3, 3, 13), penalty="l2",
                    solver="liblinear", scoring="neg_log_loss",
                    cv=inner_gkf.split(Xtr, y[tr], inner_groups),
                    max_iter=1000,
                )
                clf.fit(Xtr, y[tr])
                proba = clf.predict_proba(Xte)[:, 1]
                preds_this_seed[te] = proba
                # regression on p_mixed
                rr = RidgeCV(alphas=np.logspace(-3, 3, 13),
                              cv=inner_gkf.split(Xtr, p_mixed[tr], inner_groups),
                              scoring="neg_mean_squared_error")
                rr.fit(Xtr, p_mixed[tr])
                preds_p[te] = rr.predict(Xte)
                # regression on H
                rr2 = RidgeCV(alphas=np.logspace(-3, 3, 13),
                               cv=inner_gkf.split(Xtr, H_ent[tr], inner_groups),
                               scoring="neg_mean_squared_error")
                rr2.fit(Xtr, H_ent[tr])
                preds_H[te] = rr2.predict(Xte)
            auroc_list.append(float(roc_auc_score(y, preds_this_seed)))
            ll = float(log_loss(y, np.clip(preds_this_seed, 1e-6, 1 - 1e-6)))
            logloss_list.append(ll)
            rho_p_list.append(float(spearmanr(preds_p, p_mixed).correlation))
            rho_H_list.append(float(spearmanr(preds_H, H_ent).correlation))
            preds_all.append(preds_this_seed)
        preds_arr = np.stack(preds_all, axis=0)  # (n_seeds, n_clones)
        preds_mean = preds_arr.mean(axis=0)
        results[arm_name] = {
            "cols": cols,
            "auroc_per_seed": auroc_list,
            "logloss_per_seed": logloss_list,
            "spearman_p_per_seed": rho_p_list,
            "spearman_H_per_seed": rho_H_list,
            "preds_per_seed_mean": preds_mean,
            "auroc_mean": float(np.mean(auroc_list)),
            "logloss_mean": float(np.mean(logloss_list)),
            "spearman_p_mean": float(np.mean(rho_p_list)),
            "spearman_H_mean": float(np.mean(rho_H_list)),
        }
        print(f"  {arm_name}: AUROC = {results[arm_name]['auroc_mean']:.4f}  "
              f"logloss = {results[arm_name]['logloss_mean']:.4f}  "
              f"rho_p = {results[arm_name]['spearman_p_mean']:+.3f}  "
              f"rho_H = {results[arm_name]['spearman_H_mean']:+.3f}")
    return results, y, p_mixed, H_ent


# ============================================================================
#  Phase 7 — paired clone-level bootstrap
# ============================================================================
def paired_bootstrap(results, y, p_mixed, H_ent):
    from sklearn.metrics import roc_auc_score, log_loss
    print("\n[phase7] paired clone-level bootstrap (2000 resamples) ...")
    n = len(y)
    rng = np.random.default_rng(0)
    keys = list(results.keys())
    arm_preds = {k: results[k]["preds_per_seed_mean"] for k in keys}
    # metrics per bootstrap
    metric_boot = {k: {"auroc": [], "logloss": []} for k in keys}
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:  # degenerate resample
            for k in keys:
                metric_boot[k]["auroc"].append(np.nan)
                metric_boot[k]["logloss"].append(np.nan)
            continue
        for k in keys:
            pb = np.clip(arm_preds[k][idx], 1e-6, 1 - 1e-6)
            metric_boot[k]["auroc"].append(float(roc_auc_score(yb, pb)))
            metric_boot[k]["logloss"].append(float(log_loss(yb, pb)))

    for k in keys:
        for m in ("auroc", "logloss"):
            arr = np.array(metric_boot[k][m])
            results[k][f"boot_{m}_mean"] = float(np.nanmean(arr))
            results[k][f"boot_{m}_ci"] = (
                float(np.nanpercentile(arr, 2.5)),
                float(np.nanpercentile(arr, 97.5)),
            )
    # paired diffs vs arm1
    diffs = {}
    for k in keys:
        if k == "arm1_E": continue
        auroc_diffs = np.array(metric_boot[k]["auroc"]) - np.array(metric_boot["arm1_E"]["auroc"])
        ll_diffs = np.array(metric_boot[k]["logloss"]) - np.array(metric_boot["arm1_E"]["logloss"])
        diffs[k] = {
            "auroc_delta_mean": float(np.nanmean(auroc_diffs)),
            "auroc_delta_ci": (float(np.nanpercentile(auroc_diffs, 2.5)),
                                float(np.nanpercentile(auroc_diffs, 97.5))),
            "logloss_delta_mean": float(np.nanmean(ll_diffs)),
            "logloss_delta_ci": (float(np.nanpercentile(ll_diffs, 2.5)),
                                  float(np.nanpercentile(ll_diffs, 97.5))),
        }
        print(f"  {k} vs arm1: ΔAUROC = {diffs[k]['auroc_delta_mean']:+.4f}  "
              f"95%CI [{diffs[k]['auroc_delta_ci'][0]:+.4f}, "
              f"{diffs[k]['auroc_delta_ci'][1]:+.4f}]  "
              f"Δlogloss = {diffs[k]['logloss_delta_mean']:+.4f}")
    return results, diffs


# ============================================================================
#  Phase 8 — verdict per prereg
# ============================================================================
def verdict(results, diffs):
    r_a1 = results["arm1_E"]
    pass_arms = []
    for k in ("arm2_E+C", "arm3_E+Ic"):
        d = diffs[k]
        auroc_ok = d["auroc_delta_mean"] >= 0.02 and d["auroc_delta_ci"][0] > 0
        ll_ok = results[k]["auroc_mean"] > 0 and results[k]["logloss_mean"] < r_a1["logloss_mean"]
        if auroc_ok and ll_ok:
            pass_arms.append(k)
    verdict_str = f"PASS Gate 3 strict (winners: {pass_arms})" if pass_arms else "FAIL Gate 3 strict"
    print(f"\nVerdict: {verdict_str}")
    return verdict_str, pass_arms


# ============================================================================
#  main
# ============================================================================
def _tn(v):
    if isinstance(v, (np.floating,)): return float(v)
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, (list, tuple)): return [_tn(x) for x in v]
    if isinstance(v, dict): return {k: _tn(x) for k, x in v.items()}
    return v


def main():
    print("=" * 80); print("Gate 3 strict"); print("=" * 80)
    t_all = time.time()
    adata_day2, clone_late = build_day2_h5ad()
    labels_df = build_labels(clone_late)
    features_cell = compute_features(adata_day2)
    per_clone = aggregate_per_clone(features_cell, adata_day2, labels_df)
    print(f"\nEligible clones after aggregation: {len(per_clone)}")

    sanity = sanity_checks(features_cell, per_clone)
    results, y, p_mixed, H_ent = evaluate_arms(per_clone)
    results, diffs = paired_bootstrap(results, y, p_mixed, H_ent)
    verdict_str, pass_arms = verdict(results, diffs)

    summary = {
        "verdict": verdict_str,
        "cohort": {**adata_day2.uns.get("cohort_report", {}),
                    "n_clones_eligible": int(len(per_clone)),
                    "positive_fraction": float(per_clone["y_mixed"].mean()),
                    "descendant_count_stats": {
                        "mean": float(per_clone["n_descendants"].mean()),
                        "median": float(per_clone["n_descendants"].median()),
                        "min": int(per_clone["n_descendants"].min()),
                        "max": int(per_clone["n_descendants"].max()),
                        "q25": float(per_clone["n_descendants"].quantile(0.25)),
                        "q75": float(per_clone["n_descendants"].quantile(0.75)),
                    }},
        "sanity": sanity,
        "arms": {k: {kk: v for kk, v in vv.items() if kk != "preds_per_seed_mean"}
                  for k, vv in results.items()},
        "pair_diffs_vs_arm1": diffs,
    }
    (OUT / "gate3_strict_summary.json").write_text(json.dumps(_tn(summary), indent=2))
    # save fold assignment as CSV — deterministic reconstruction
    fold_rows = []
    from sklearn.model_selection import GroupKFold
    from sklearn.utils import shuffle
    groups = per_clone["clone_id"].values
    for cv_seed in CV_SEEDS:
        uniq = np.unique(groups)
        perm = shuffle(np.arange(len(uniq)), random_state=cv_seed)
        reordered = uniq[perm]
        mp = {g: i for i, g in enumerate(reordered)}
        g2 = np.array([mp[g] for g in groups])
        gkf = GroupKFold(n_splits=CV_FOLDS)
        for f_idx, (tr, te) in enumerate(gkf.split(np.zeros((len(y), 1)), np.zeros(len(y)), g2)):
            for i in te:
                fold_rows.append({"cv_seed": cv_seed, "fold": f_idx,
                                   "clone_id": int(per_clone.iloc[i]["clone_id"]),
                                   "held_out": True})
    pd.DataFrame(fold_rows).to_csv(OUT / "gate3_strict_folds.csv", index=False)
    print(f"\n[written] gate3_strict_summary.json + gate3_strict_folds.csv "
          f"+ features_per_cell.parquet + features_per_clone.parquet  "
          f"total {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
