"""Gate 1 — LARRY feature build (heavy compute, run once).
Assembles h5ad from Klein-lab raw files, preprocesses, computes:
  - Palantir pseudotime + branch probabilities on myeloid subset
  - scJDO features per branch × 3 seeds
  - Consensus archetype procedure on the LARRY substrate (per Gate 1 amendment)
  - Local-covariance features (Gate 0b baseline)
  - CellRank fate probabilities (no clone labels)
  - Fate labels from clone matrix (759 day-2 cells target)
Saves all per-cell feature blocks + labels + groups into gate1_features.npz."""
from __future__ import annotations
import gzip, json, sys, warnings, time
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmread
import scanpy as sc
import anndata as ad
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

OUT = REPO / "New_analysis" / "gates_r26" / "gate1"
DATA = OUT / "data"
DATA.mkdir(parents=True, exist_ok=True)

LARRY_DIR = Path("/tmp/larry_data")
H5AD_PATH_RAW = DATA / "larry_myeloid.h5ad"        # written by _build_h5ad.py
H5AD_PATH_PP = DATA / "larry_myeloid_preproc.h5ad" # normalized + HVG + PCA + FA

MYELOID_CELLTYPES = {"Undifferentiated", "Neutrophil", "Monocyte"}
SCJDO_SEEDS = [0, 1, 2]
K_ARCH = 5


def build_myeloid_h5ad():
    if H5AD_PATH_PP.exists():
        print(f"[build] loading preprocessed cache {H5AD_PATH_PP}")
        return sc.read_h5ad(H5AD_PATH_PP)
    if H5AD_PATH_RAW.exists():
        print(f"[build] loading raw cache {H5AD_PATH_RAW} — will preprocess...")
        mye = sc.read_h5ad(H5AD_PATH_RAW)
        return _preprocess_and_save(mye)
    print("[build] reading Klein LARRY raw files...")
    t0 = time.time()
    with gzip.open(LARRY_DIR / "stateFate_inVitro_normed_counts.mtx.gz", "rt") as f:
        X = mmread(f).tocsr().astype(np.float32)
    print(f"  counts {X.shape} in {time.time()-t0:.1f}s")
    genes = pd.read_csv(LARRY_DIR / "stateFate_inVitro_gene_names.txt.gz",
                        sep="\t", header=None)[0].values
    meta = pd.read_csv(LARRY_DIR / "stateFate_inVitro_metadata.txt.gz", sep="\t")
    with gzip.open(LARRY_DIR / "stateFate_inVitro_clone_matrix.mtx.gz", "rt") as f:
        C = mmread(f).tocsr()  # (n_cells, n_clones)
    print(f"  meta {meta.shape} genes {len(genes)} clone {C.shape}")

    obs = meta.copy()
    obs.index = pd.Index([f"cell_{i}" for i in range(len(obs))])
    obs["cell_type"] = obs["Cell type annotation"].astype(str)
    obs["day"] = obs["Time point"].astype(int)
    mye_mask = obs["cell_type"].isin(MYELOID_CELLTYPES).values
    print(f"[build] myeloid subset: {int(mye_mask.sum())} cells")

    # subset counts + clone in sparse form BEFORE building AnnData; free raw X
    X_mye = X[mye_mask].tocsr()
    C_mye = C[mye_mask].tocsr()
    del X, C
    import gc; gc.collect()

    mye = ad.AnnData(X=X_mye, obs=obs[mye_mask].copy(),
                      var=pd.DataFrame(index=pd.Index(genes, name="gene")))
    mye.obsm["X_clone"] = C_mye
    print(f"[build] AnnData shape {mye.shape}")

    return _preprocess_and_save(mye)


def _preprocess_and_save(mye):
    print("[build] subsampling for tractable pseudotime + scJDO fit (compute-scale)...")
    t0 = time.time()
    rng = np.random.default_rng(0)
    has_clone = np.asarray(mye.obsm["X_clone"].sum(axis=1)).flatten() > 0
    d = mye.obs["day"].values
    ct = mye.obs["cell_type"].values

    keep_d2_clone = (d == 2) & has_clone                            # ~4.6K
    # subsample Neut/Mono at day-4/6 — 2000 each for terminal anchors
    neut_idx = np.where((d != 2) & (ct == "Neutrophil"))[0]
    mono_idx = np.where((d != 2) & (ct == "Monocyte"))[0]
    neut_sample = rng.choice(neut_idx, size=min(2000, len(neut_idx)), replace=False)
    mono_sample = rng.choice(mono_idx, size=min(2000, len(mono_idx)), replace=False)
    keep_terminals = np.zeros(mye.n_obs, dtype=bool)
    keep_terminals[neut_sample] = True; keep_terminals[mono_sample] = True
    # subsample late Undiff and day-2 Undiff without clone
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
    print(f"[build] subsample: total={int(keep.sum())} "
          f"(d2_clone={int(keep_d2_clone.sum())}, "
          f"terminals_neut={len(neut_sample)}, terminals_mono={len(mono_sample)}, "
          f"late_undiff={len(late_undiff_sample)}, d2_undiff={len(d2_undiff_sample)})")
    mye = mye[keep].copy()

    print("[build] normalize + log + HVG(2000) + PCA(30) as X_fa rep (memory-safe)...")
    sc.pp.normalize_total(mye, target_sum=1e4)
    sc.pp.log1p(mye)
    sc.pp.highly_variable_genes(mye, n_top_genes=2000, flavor="seurat",
                                 batch_key=None, subset=False)
    mye = mye[:, mye.var["highly_variable"]].copy()
    print(f"[build] after HVG: {mye.shape}")
    # arpack on sparse: implicit zero-centering, no densification
    sc.tl.pca(mye, n_comps=30, random_state=0, svd_solver="arpack",
              use_highly_variable=False)
    mye.obsm["X_fa"] = mye.obsm["X_pca"].astype(np.float32)
    mye.varm["fa_loadings"] = mye.varm["PCs"].astype(np.float32)
    mye.write_h5ad(H5AD_PATH_PP)
    print(f"[build] wrote {H5AD_PATH_PP}  {mye.shape}  preproc time {time.time()-t0:.1f}s")
    return mye


def palantir_pseudotime(mye, seed=0):
    """Run Palantir with a day-2 Undifferentiated start cell chosen by SPRING coords.
    Terminal cells = medoid Neut + medoid Mono (day-6 cells)."""
    import palantir
    t0 = time.time()
    # k-NN + diffusion maps on X_fa
    print("[palantir] neighbors + diffusion maps...")
    sc.pp.neighbors(mye, use_rep="X_fa", n_neighbors=30, random_state=seed)
    dm = palantir.utils.run_diffusion_maps(pd.DataFrame(mye.obsm["X_fa"],
                                                          index=mye.obs_names),
                                              n_components=15)
    ms = palantir.utils.determine_multiscale_space(dm)
    # start: earliest day-2 Undiff cell (min SPRING-x among day-2 Undiff, arbitrary but reproducible)
    d2u = mye.obs[(mye.obs["day"] == 2) & (mye.obs["cell_type"] == "Undifferentiated")]
    start_cell = d2u.sort_values("SPRING-x").index[0]
    print(f"[palantir] start cell: {start_cell}")
    # terminal states: medoid Neut, medoid Mono on day 6
    d6n = mye.obs[(mye.obs["day"] == 6) & (mye.obs["cell_type"] == "Neutrophil")]
    d6m = mye.obs[(mye.obs["day"] == 6) & (mye.obs["cell_type"] == "Monocyte")]
    # pick median-SPRING cell of each
    tn = d6n.sort_values("SPRING-x").index[len(d6n) // 2]
    tm = d6m.sort_values("SPRING-x").index[len(d6m) // 2]
    terminals = pd.Series(["Neut", "Mono"], index=[tn, tm])
    print(f"[palantir] terminals: {tn} (Neut), {tm} (Mono)")

    pr = palantir.core.run_palantir(
        ms, start_cell, terminal_states=terminals, knn=30, num_waypoints=1200,
        seed=seed, use_early_cell_as_start=False,
    )
    mye.obs["pseudotime"] = pr.pseudotime.reindex(mye.obs_names).values.astype(np.float32)
    mye.obs["entropy"] = pr.entropy.reindex(mye.obs_names).values.astype(np.float32)
    bp = pr.branch_probs.reindex(mye.obs_names).fillna(0.0)
    mye.obsm["branch_probs"] = bp.values.astype(np.float32)
    mye.uns["branch_names"] = list(bp.columns)
    print(f"[palantir] done in {time.time()-t0:.1f}s")


def fit_scjdo_features(mye, seed):
    """Run fit_drift_branches per branch with vel_scale=0. Extract per-cell:
      - Re(lambda_max) at cell's pseudotime bin (interpolated)
      - Leading-J projection: cell's X_fa @ leading real eigenvector at its bin
      - Per-branch patterns (K, D, D) for consensus procedure
    Returns dict: {branch: {re_lambda_max, lead_proj, patterns, t_centers, J_tensor, bandwidth}}."""
    from scjdo.tl import fit_drift_branches
    torch.manual_seed(seed); np.random.seed(seed)
    b = mye.copy()
    # branch_masks from Palantir branch_probs: cell belongs to branch if prob >= 0.5
    bp = pd.DataFrame(b.obsm["branch_probs"], index=b.obs_names,
                      columns=b.uns["branch_names"])
    masks = (bp >= 0.5).astype(bool)
    # ensure each branch has cells
    for name in bp.columns:
        if masks[name].sum() < 100:
            masks[name] = bp[name] >= bp[name].quantile(0.5)
    b.obsm["branch_masks"] = masks

    b.obs["cell_fate"] = "Other"
    for name in bp.columns:
        b.obs.loc[bp[name].values >= 0.7, "cell_fate"] = name
    b.obs.loc[b.obs["day"] == 2, "cell_fate"] = "Progenitor"

    branches = list(bp.columns)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_drift_branches(
            b, rep="X_fa", branch_key="branch_masks", branch_names=branches,
            time_key="pseudotime", groupby="cell_fate",
            progenitor_cluster="Progenitor",
            terminal_clusters={name: name for name in branches},
            bias_strength=1.5, n_archetypes=K_ARCH, n_epochs=3000,
            vel_scale=0.0, hidden=256, depth=4, sigma=0.10,
            windowing="kernel", bandwidth="auto", grid_size=150,
            seed=seed, verbose=False,
        )
    out = {}
    tau_all = b.obs["pseudotime"].values.astype(np.float64)
    X_fa = b.obsm["X_fa"].astype(np.float64)
    for name in branches:
        key = f"scjdo_{name}"
        res = b.uns[key]
        Jt = np.asarray(res["J_tensor"])           # (T, D, D)
        tc = np.asarray(res["t_centers"])          # (T,)
        max_re = np.asarray(res["max_real_eig"])   # (T,)
        pats = np.asarray(res["patterns"])         # (K, D, D)
        bw = float(res.get("bandwidth", 0.05))

        # per-cell interpolated Re(lambda_max)
        idx = np.searchsorted(tc, tau_all).clip(0, len(tc) - 1)
        cell_re = max_re[idx]  # nearest bin

        # per-cell leading real eigenvector projection
        # compute leading eigvec per grid, then per-cell projection
        cell_proj = np.zeros(b.n_obs, dtype=np.float64)
        for bin_i in range(len(tc)):
            eigval, eigvec = np.linalg.eig(Jt[bin_i].astype(np.float64))
            k_top = int(np.argmax(np.real(eigval)))
            v = np.real(eigvec[:, k_top])
            # cells at this bin
            in_bin = (idx == bin_i)
            if in_bin.any():
                cell_proj[in_bin] = X_fa[in_bin] @ v
        out[name] = {
            "re_lambda_max": cell_re.astype(np.float32),
            "lead_proj": cell_proj.astype(np.float32),
            "patterns": pats.astype(np.float32),
            "t_centers": tc.astype(np.float32),
            "J_tensor": Jt.astype(np.float32),
            "bandwidth": bw,
            "activations": np.asarray(res["activations"]).astype(np.float32),  # (T, K)
        }
    return out


def consensus_archetype_activation(per_seed, branch, mye_n_obs, tau, T=200):
    """LARRY-substrate consensus procedure per Gate 1 amendment.
    per_seed: list of dicts (one per scJDO seed) with 'patterns' (K, D, D)
              and 'activations' (T, K) and 't_centers'.
    Returns per-cell activation of the passing cluster centroid, or None if
    no cluster passes."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    R = len(per_seed)
    K = per_seed[0]["patterns"].shape[0]
    # sign-invariant flatten
    all_pats = np.concatenate([ps["patterns"].reshape(K, -1).astype(np.float64)
                                 for ps in per_seed], axis=0)
    n = np.linalg.norm(all_pats, axis=1, keepdims=True) + 1e-12
    F = all_pats / n
    lead = F[:, 0]
    F = F * np.where(lead < 0, -1.0, 1.0).reshape(-1, 1)
    S = np.clip(np.abs(F @ F.T), 0.0, 1.0)
    D = 0.5 * (1.0 - S + (1.0 - S).T)
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="single")
    labels = fcluster(Z, t=K, criterion="maxclust")
    rep_ids = np.repeat(np.arange(R), K)
    passing = None
    for lbl in np.unique(labels):
        members = np.where(labels == lbl)[0]
        reps = set(int(rep_ids[m]) for m in members)
        if len(reps) >= int(np.ceil(0.8 * R)):
            passing = (lbl, members, len(reps))
            break
    if passing is None:
        return None
    lbl, members, R_cluster = passing
    # centroid per seed: mean of that seed's contributing patterns
    # per-cell activation: for each seed, project J_cell onto its centroid via
    # OLS; average across seeds. To keep it clean: use seed-0's contribution
    # to the centroid; regress J_cell (D, D) onto centroid (D, D) yielding a
    # scalar activation per cell per bin.
    # Simpler: use the AVERAGE seed's centroid and project J_cell interpolated at bin.
    centroid = F[members].mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
    D_sqrt = per_seed[0]["patterns"].shape[1]
    centroid_mat = centroid.reshape(D_sqrt, D_sqrt)
    # per-cell activation: use seed-0 J_tensor (features come from seed-0 fit)
    # activation_at_bin_i = <J_i, centroid_mat>_F / <centroid, centroid>_F
    per_seed_acts = []
    for ps in per_seed:
        Jt = ps["J_tensor"].astype(np.float64)
        tc = ps["t_centers"]
        num = np.tensordot(Jt, centroid_mat, axes=([1, 2], [0, 1]))  # (T,)
        denom = float((centroid_mat * centroid_mat).sum() + 1e-12)
        act_bin = (num / denom).astype(np.float32)  # (T,)
        # per-cell interpolation
        idx = np.searchsorted(tc, tau).clip(0, len(tc) - 1)
        per_seed_acts.append(act_bin[idx])
    avg_act = np.mean(per_seed_acts, axis=0).astype(np.float32)
    return {"activation_per_cell": avg_act, "R_cluster": R_cluster,
            "centroid": centroid_mat.astype(np.float32), "n_members": len(members)}


def local_covariance_features(mye, tau, grid, bandwidth, K_cov=5, seed=0):
    """Build (T, D, D) covariance tensor + semi-NMF, return per-cell activations
    at the cell's pseudotime bin plus local leading-eigenvalue feature."""
    from scjdo.archetypes.decompose import jacobian_modes
    X = mye.obsm["X_fa"].astype(np.float64)
    tau64 = tau.astype(np.float64)
    T = len(grid); D = X.shape[1]
    C = np.zeros((T, D, D), dtype=np.float32)
    for i, tc in enumerate(grid):
        w = np.exp(-((tau64 - tc) ** 2) / (2.0 * bandwidth ** 2))
        W = w.sum()
        if W < 1e-9: continue
        mu = (w[:, None] * X).sum(0) / W
        Xc = X - mu[None, :]
        C[i] = ((w[:, None, None] * Xc[:, :, None] * Xc[:, None, :]).sum(0) / W).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p, a, err = jacobian_modes(torch.tensor(C), rank=K_cov, n_restarts=5, seed=seed)
    activations = a.numpy()  # (T, K_cov)
    # leading local eigenvalue
    lead_ev = np.zeros(T, dtype=np.float32)
    for i in range(T):
        ev = np.linalg.eigvalsh(C[i].astype(np.float64))
        lead_ev[i] = float(ev[-1])
    idx = np.searchsorted(grid, tau).clip(0, T - 1)
    per_cell_acts = activations[idx]           # (N, K_cov)
    per_cell_lead = lead_ev[idx]               # (N,)
    return np.concatenate([per_cell_acts, per_cell_lead[:, None]], axis=1).astype(np.float32)


def cellrank_fate_probs(mye):
    """CellRank 2.x: CytoTRACE kernel? Use a Palantir-based approach:
    build a transition matrix from the pseudotime + connectivity, get fate probs.
    We use CellRank's PseudotimeKernel for simplicity — this uses NO clone labels."""
    import cellrank as cr
    from cellrank.kernels import PseudotimeKernel
    from cellrank.estimators import GPCCA
    k = PseudotimeKernel(mye, time_key="pseudotime")
    k.compute_transition_matrix()
    g = GPCCA(k)
    g.compute_schur(n_components=10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g.compute_macrostates(n_states=2, cluster_key="cell_type")
        # try to align macrostates with Neut / Mono
        macros = g.macrostates.unique().tolist()
        # pick terminal set = both (auto-pick top-2)
        g.set_terminal_states(macros[:2] if len(macros) >= 2 else macros)
        g.compute_fate_probabilities()
    fp = g.fate_probabilities.X  # (N, n_states)
    # map to (Neut, Mono) by cell-type enrichment in each macrostate
    # simplest: just take both cols, let LR sort out sign
    return np.asarray(fp).astype(np.float32)


def classification_cohort(mye):
    """From clone matrix, build day-2 fate labels + clone group ids.
    Reproduces the prereg's clone-purity rule."""
    C = mye.obsm["X_clone"]  # (N_myeloid, n_clones) may be smaller than original since we subset
    # NB: subsetting doesn't drop columns; empty clone cols are fine
    coo = C.tocoo()
    clone_days = {}; clone_fates = {}
    days = mye.obs["day"].values
    ct = mye.obs["cell_type"].values
    for r, c in zip(coo.row, coo.col):
        clone_days.setdefault(c, set()).add(int(days[r]))
        if days[r] in (4, 6) and ct[r] in ("Neutrophil", "Monocyte"):
            clone_fates.setdefault(c, Counter())[ct[r]] += 1
    neut_clones = set(); mono_clones = set()
    for c, dset in clone_days.items():
        if 2 not in dset or c not in clone_fates: continue
        fates = clone_fates[c]
        n_ct = fates.get("Neutrophil", 0); m_ct = fates.get("Monocyte", 0)
        if n_ct + m_ct < 2: continue
        frac_n = n_ct / (n_ct + m_ct)
        if frac_n >= 0.8: neut_clones.add(c)
        elif frac_n <= 0.2: mono_clones.add(c)
    # day-2 cells in these clones
    day2_idx = np.where(days == 2)[0]
    cell_label = np.full(mye.n_obs, -1, dtype=np.int8)
    cell_group = np.full(mye.n_obs, -1, dtype=np.int64)
    # for each day-2 cell, check clone membership
    Ccsr = C.tocsr()
    for cell_idx in day2_idx:
        cols = Ccsr.indices[Ccsr.indptr[cell_idx]:Ccsr.indptr[cell_idx + 1]]
        for cid in cols:
            if cid in neut_clones:
                cell_label[cell_idx] = 1; cell_group[cell_idx] = cid; break
            elif cid in mono_clones:
                cell_label[cell_idx] = 0; cell_group[cell_idx] = cid; break
    mask = cell_label >= 0
    print(f"[cohort] {mask.sum()} labelled cells "
          f"(neut={int((cell_label == 1).sum())}, mono={int((cell_label == 0).sum())}) "
          f"across {len(np.unique(cell_group[mask]))} clones")
    return mask, cell_label, cell_group


def main():
    print("=" * 80); print("Gate 1: LARRY feature build"); print("=" * 80)
    t0 = time.time()
    mye = build_myeloid_h5ad()
    print(f"myeloid substrate: {mye.shape}  time {time.time()-t0:.1f}s")

    palantir_pseudotime(mye, seed=0)
    tau = mye.obs["pseudotime"].values.astype(np.float32)

    # scJDO features × 3 seeds — per-seed subprocess call with cache + wall-clock cap
    import subprocess, pickle
    HELPER = Path(__file__).parent / "_scjdo_one_seed.py"
    per_seed_features = {}
    # Palantir pseudotime + branch probs saved to preproc h5ad so subprocess can load
    if "pseudotime" not in mye.obs or "branch_probs" not in mye.obsm:
        raise RuntimeError("Palantir outputs missing from mye")
    mye.write_h5ad(H5AD_PATH_PP)
    for seed in SCJDO_SEEDS:
        ckpt = OUT / f"gate1_scjdo_seed{seed}.pkl"
        if ckpt.exists():
            print(f"\n[scjdo] seed={seed} loading cache {ckpt.name}")
            per_seed_features[seed] = pickle.loads(ckpt.read_bytes())
            continue
        print(f"\n[scjdo] seed={seed} → subprocess {HELPER.name}")
        t_s = time.time()
        try:
            subprocess.run(
                [sys.executable, "-u", str(HELPER), str(seed)],
                cwd=str(REPO), check=True, timeout=2700,  # 45 min
            )
        except subprocess.TimeoutExpired:
            print(f"  seed {seed} TIMEOUT after 45 min — skipping"); continue
        except subprocess.CalledProcessError as e:
            print(f"  seed {seed} FAILED: {e}"); continue
        if ckpt.exists():
            per_seed_features[seed] = pickle.loads(ckpt.read_bytes())
            print(f"  seed {seed} done in {time.time()-t_s:.1f}s → cached")
    if not per_seed_features:
        raise RuntimeError("All scJDO seeds failed")

    branches = list(next(iter(per_seed_features.values())).keys())
    print(f"branches: {branches}")

    # consensus archetype per branch across the completed seeds
    available_seeds = sorted(per_seed_features.keys())
    print(f"[consensus] using {len(available_seeds)} seeds: {available_seeds}")
    consensus_result = {}
    for branch in branches:
        per_seed = [per_seed_features[s][branch] for s in available_seeds]
        res = consensus_archetype_activation(per_seed, branch, mye.n_obs, tau)
        consensus_result[branch] = res
        if res is None:
            print(f"[consensus] {branch}: no passing cluster → archetype feature DROPPED")
        else:
            print(f"[consensus] {branch}: passing cluster R={res['R_cluster']} n={res['n_members']}")

    # local covariance features (use seed-0 bandwidth)
    print("\n[local-cov]")
    bw = per_seed_features[SCJDO_SEEDS[0]][branches[0]]["bandwidth"]
    tc0 = per_seed_features[SCJDO_SEEDS[0]][branches[0]]["t_centers"]
    covariance_feats = local_covariance_features(mye, tau, tc0, bw, K_cov=5, seed=0)
    print(f"  cov features: {covariance_feats.shape}  bandwidth={bw:.4f}")

    # cellrank fate probs
    print("\n[cellrank] pseudotime kernel + GPCCA 2 macrostates...")
    try:
        cr_feats = cellrank_fate_probs(mye)
        print(f"  cellrank features: {cr_feats.shape}")
    except Exception as e:
        print(f"  cellrank failed: {e}  → falling back to Palantir branch probs")
        cr_feats = mye.obsm["branch_probs"].astype(np.float32)
        print(f"  fallback shape: {cr_feats.shape}")

    # classification cohort
    print("\n[cohort]")
    mask, cell_label, cell_group = classification_cohort(mye)

    # build feature blocks (per prereg): per-cell per-branch scJDO features
    # S = for each branch: re_lambda_max + lead_proj + consensus_activation (if passing)
    S_blocks = []
    S_names = []
    for branch in branches:
        for s in SCJDO_SEEDS:
            f_re = per_seed_features[s][branch]["re_lambda_max"]
            f_lp = per_seed_features[s][branch]["lead_proj"]
            S_blocks.append(np.stack([f_re, f_lp], axis=1))  # (N, 2)
            S_names.extend([f"S_seed{s}_{branch}_re_lmax", f"S_seed{s}_{branch}_lead_proj"])
        if consensus_result[branch] is not None:
            S_blocks.append(consensus_result[branch]["activation_per_cell"][:, None])
            S_names.append(f"S_{branch}_consensus_arch_act")

    # For evaluation, we present features per-seed (per completed scJDO seed).
    S_per_seed = {}
    for s in available_seeds:
        cols = []; names = []
        for branch in branches:
            f_re = per_seed_features[s][branch]["re_lambda_max"]
            f_lp = per_seed_features[s][branch]["lead_proj"]
            cols.append(f_re[:, None]); names.append(f"S_{branch}_re_lmax")
            cols.append(f_lp[:, None]); names.append(f"S_{branch}_lead_proj")
            if consensus_result[branch] is not None:
                cols.append(consensus_result[branch]["activation_per_cell"][:, None])
                names.append(f"S_{branch}_consensus_arch_act")
        S_per_seed[s] = np.concatenate(cols, axis=1).astype(np.float32)
        if s == available_seeds[0]:
            print(f"S feature bundle per seed: {S_per_seed[s].shape}  ({names})")

    # Save everything
    print("\n[save] writing gate1_features.npz ...")
    save_kw = {
        "X_pca": mye.obsm["X_pca"].astype(np.float32),
        "X_fa": mye.obsm["X_fa"].astype(np.float32),
        "cov_features": covariance_feats.astype(np.float32),
        "cellrank_features": cr_feats.astype(np.float32),
        "cohort_mask": mask,
        "cohort_label": cell_label,
        "cohort_group": cell_group,
        "pseudotime": tau,
        "day": mye.obs["day"].values.astype(np.int32),
        "consensus_passing_neut": np.array([consensus_result[branches[0]] is not None]),
        "consensus_passing_mono": np.array([consensus_result[branches[1]] is not None]) if len(branches) > 1 else np.array([False]),
        "branches": np.array(branches, dtype=object),
        "available_seeds": np.array(available_seeds, dtype=np.int32),
    }
    for s in available_seeds:
        save_kw[f"S_seed{s}"] = S_per_seed[s]
    np.savez(OUT / "gate1_features.npz", **{k: v for k, v in save_kw.items() if not isinstance(v, (list, tuple))})
    print(f"[done] wrote {OUT / 'gate1_features.npz'}  total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
