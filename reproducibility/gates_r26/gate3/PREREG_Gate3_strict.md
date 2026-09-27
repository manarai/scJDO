# PREREG — Gate 3 strict (soft-mode falsification, day-2 covariance features only)

**Status**: pre-registered before any data download, feature extraction, label computation, or model fitting for Gate 3. Frozen against `1166786` (Gate 1 v2 DECISION update on remote / local main after cherry-pick). No changes to this file after commit.

## Purpose

Test whether local-covariance softening at day 2 predicts whether a LARRY clone is still undecided in fate, BEYOND expression alone. Not a test of scJDO's fitted field. Theory + retrospective analysis (Gate 0b + Gate 1) showed the fitted-field softness feature is redundant with local covariance under isotropic noise; the fitted-field arm is excluded from Gate 3. This is the last observational test on the soft-mode route.

## Data (Step 1)

- **Release**: Weinreb et al. 2020 LARRY in vitro state-fate. Files (already on disk from Gate 1: `/tmp/larry_data/`):
  - `stateFate_inVitro_normed_counts.mtx.gz` (2 GB decompressed → 5.6 GB text; verified `gunzip -t` OK).
  - `stateFate_inVitro_gene_names.txt.gz` (25,289 genes).
  - `stateFate_inVitro_metadata.txt.gz` (130,887 × 8: `Library`, `Cell barcode`, `Time point`, `Starting population`, `Cell type annotation`, `Well`, `SPRING-x`, `SPRING-y`).
  - `stateFate_inVitro_clone_matrix.mtx.gz` (130,887 × 5,864 sparse binary).
- **Cell filter**: `Time point == 2.0` for ALL feature construction (E, C, Ic). No day-4 or day-6 cell may enter normalization, HVG selection, FA fitting, k-NN construction, or covariance estimation. Day-4/6 cells are used only to build the clone → fate outcome matrix (for labels).
- **Filtering steps to be reported in the report**:
  - `n_total`: total cells in metadata.
  - `n_day2`: day-2 cells.
  - `n_day2_with_clone`: day-2 cells carrying any clone barcode.
  - `n_late`: day-4 + day-6 cells (used only for labels).
  - `n_clones_eligible`: clones passing "≥ 1 barcoded day-2 cell AND ≥ 5 mature descendants at day 4/6" (see labels).

## Labels (Step 2)

### Mapping table — 10 mature categories → 5 major lineage groups (frozen)

| Category (metadata) | Major group |
|:---:|:---:|
| Neutrophil     | Myeloid_Neu    |
| Monocyte       | Myeloid_Mono   |
| Erythroid      | ErythroMeg     |
| Meg            | ErythroMeg     |
| Baso           | MastGranu      |
| Mast           | MastGranu      |
| Eos            | MastGranu      |
| Lymphoid       | LymphoDC       |
| Ccr7_DC        | LymphoDC       |
| pDC            | LymphoDC       |

`Undifferentiated` is EXCLUDED from the mature-descendant count.

### Rarefaction procedure

For each clone with ≥ 1 barcoded day-2 cell AND ≥ 5 recorded mature descendants at day 4 or 6:
1. Collect the multiset of the clone's mature descendants' major groups (list of length ≥ 5 from the 5 groups).
2. Rarefy: sample exactly 5 descendants WITHOUT replacement, count distinct major groups spanned, mark `mixed = 1` if that count ≥ 2 else 0.
3. Repeat rarefaction 200 times with `numpy.random.default_rng(seed = clone_index * 7919)` (per-clone independent). Report the **mean** as `p_mixed` (a value in [0, 1]).
4. **Primary binary label** `y_mixed = 1 if p_mixed ≥ 0.5 else 0`.
5. **Secondary continuous labels**:
   - `p_mixed` itself (probability regression target).
   - `H_entropy` = mean Shannon entropy over the 5 major groups across the 200 rarefactions (natural log, in nats). Uses the empirical distribution of 5 sampled descendants each iteration.

### Reported alongside cohort

- Number of eligible clones.
- Positive fraction (mean of `y_mixed`).
- Distribution of mature-descendant counts per eligible clone: mean, median, min, max, quantiles [0.25, 0.5, 0.75].
- Distribution of `p_mixed`: histogram bins.

## Features (Step 3)

All features computed on day-2 cells only, then aggregated per clone by mean over that clone's day-2 cells.

### E — expression baseline (FA30)

- Restrict counts to day-2 cells (`X_day2`, shape (n_day2, 25289)).
- `sc.pp.normalize_total(target_sum=1e4)` on day-2 counts (fit within day-2 only).
- `sc.pp.log1p`.
- `sc.pp.highly_variable_genes(n_top_genes=2000, flavor="seurat")` fit on day-2 log-normed data.
- Subset to those 2,000 HVGs.
- `sc.pp.scale(max_value=10, zero_center=True)` on day-2 HVG matrix.
- `sklearn.decomposition.FactorAnalysis(n_components=30, random_state=0, max_iter=200)` fit on the dense scaled HVG matrix.
- Per-cell feature: 30-D FA vector. Per-clone: mean over that clone's day-2 cells → 30-D per clone. Store as `E_per_clone` (n_clones × 30).

### C — local-covariance softness ratio

Computed per day-2 cell, then averaged per clone.

- k-NN graph in FA30 space (from E above): for each day-2 cell, take its `k = 50` nearest neighbours (Euclidean, excluding self). Ties broken by cell index. Uses `sklearn.neighbors.NearestNeighbors(n_neighbors=51)`.
- For each day-2 cell `i`, form `X_nbr` = the 50 × 2000 HVG-scaled matrix of its neighbours. Compute the ridge-regularised sample covariance:
  `Σ_local_i = (X_nbr^T X_nbr) / (50 − 1) + λ · I_2000`, `λ = 1e−3 · tr(Σ_local_i_raw) / 2000` (0.1 % of mean diagonal). This is a Ledoit-Wolf-style stabiliser to avoid rank deficiency at k = 50 < d = 2000.
- Top-3 eigenvalues of Σ_local_i via `scipy.sparse.linalg.eigsh(Σ_local_i, k=3, which='LM')` on the (2000, 2000) covariance matrix (dense but with only top-3 requested).
- Compute `Σ_global` once = ridge-regularised covariance of the FULL day-2 HVG-scaled matrix (n_day2, 2000). Its top-3 eigenvalues via same eigsh call.
- Per-cell **softness ratio** `c_i = λ_1(Σ_local_i) / λ_1(Σ_global)`.
- Per-cell full features: `c_i` plus the top-3 eigenvalues of Σ_local_i (4 values per cell).
- Per-clone: mean over the clone's day-2 cells of the 4 values. Store as `C_per_clone` (n_clones × 4).

### Ic — Mojtahedi critical-transition index

Computed per day-2 cell (over its k = 50 neighbours), from **counts** (not log-normed, not scaled).

- For each day-2 cell `i`, take its 50 nearest neighbours (same k-NN graph as C).
- On this 50 × 2000 HVG-counts matrix, compute:
  - `R_gene`: mean pairwise |Pearson| between the 2000 HVG columns (2000 × 2000 gene-gene correlation, take abs, exclude diagonal, average).
  - `R_cell`: mean pairwise Pearson between the 50 rows (50 × 50 cell-cell correlation, exclude diagonal, average, NO absolute value per Mojtahedi et al. 2016 original definition).
  - `Ic_i = R_gene / R_cell` (Mojtahedi et al. 2016 "critical transition index").
- Per-clone: mean over the clone's day-2 cells. Store as `Ic_per_clone` (n_clones × 1).

### N — descendant-count covariate

- Per clone: `n_i = log(1 + number of mature descendants at day 4/6)`.
- Included in EVERY arm as a covariate (an unregularised control regressor when possible, otherwise an L2-regularised feature identical across arms).

### Saved artefacts

`gate3_strict_features_per_cell.parquet` (day-2 cells × [E30, softness_top3+ratio, Ic, cell_id, clone_id]) and `gate3_strict_features_per_clone.parquet` (clones × [E30, C4, Ic, N, y_mixed, p_mixed, H_entropy]).

## Models and evaluation (Step 4)

### Arms

All arms include N as a covariate.
- **Arm 1 — E**: 30 FA features + N.
- **Arm 2 — E + C**: 30 FA + 4 C features + N.
- **Arm 3 — E + Ic**: 30 FA + 1 Ic + N.
- **Arm 4 — E + C + Ic**: 30 FA + 4 C + 1 Ic + N (reported, not decisive).

### Models

- **Binary label `y_mixed`**: `sklearn.linear_model.LogisticRegressionCV(penalty="l2", Cs=np.logspace(-3, 3, 13), solver="liblinear", scoring="neg_log_loss", cv=inner_kf, max_iter=1000)` with an INNER 5-fold group-k-fold (grouped by clone) for hyperparameter selection. Features standardised inside each fold.
- **Continuous labels `p_mixed` and `H_entropy`**: `sklearn.linear_model.RidgeCV(alphas=np.logspace(-3, 3, 13), cv=inner_kf, scoring="neg_mean_squared_error")` with inner 5-fold group-k-fold. Same standardisation.

### Folds (outer CV)

- 5-fold `GroupKFold` grouped by clone.
- Repeated 5 times with different fold-shuffle seeds ∈ {42, 43, 44, 45, 46} (shuffled clone-index permutation via `sklearn.utils.shuffle(random_state=seed)`).
- Total 25 outer folds per arm.

### Metrics

- **Binary `y_mixed`**: AUROC (`sklearn.metrics.roc_auc_score`), log loss (`sklearn.metrics.log_loss` on the model's probability output).
- **Continuous `p_mixed`, `H_entropy`**: Spearman ρ (`scipy.stats.spearmanr`) between predicted and true on the held-out fold.

### Uncertainty

- **Paired clone-level bootstrap** (2000 resamples): for each arm, concatenate predictions across all 25 outer folds (one prediction per clone per fold-seed), then average per clone over the 5 fold-seeds. Bootstrap-resample clones (with replacement, n = n_clones) and recompute the metric.
- **Paired arm difference** vs Arm 1: same bootstrap resample used for both arms, difference computed within each resample. 95 % CI = 2.5th–97.5th percentiles.

## Pass criterion (Step 5, pre-registered, exact)

**PASS Gate 3 strict** if **Arm 2 OR Arm 3** satisfies BOTH:
1. mean AUROC vs Arm 1: ΔAUROC ≥ 0.02 with the 95 % paired-bootstrap CI EXCLUDING zero on the positive side.
2. mean log loss vs Arm 1: mean(log loss) of the winning arm is STRICTLY LOWER than Arm 1.

Both conditions must hold for the same arm (Arm 2 or Arm 3). Both on the primary binary label `y_mixed`. Anything else is FAIL. **Arm 4 is reported but does not decide.**

## Sanity checks (Step 6, reported before looking at primary result)

Run and print BEFORE running the primary evaluation.

1. **ID alignment**: assert that `features_per_clone.clone_id` and the label vectors have identical clone-ID ordering. Print the count and any mismatch.
2. **Pseudotime positive control**: Palantir pseudotime on day-2 X_FA (30 components). Report Spearman ρ of C softness ratio vs day-2 pseudotime AND of Ic vs day-2 pseudotime. Non-zero ρ → features carry signal.
3. **Non-trivial variance**: Report min, max, mean, std, and 5–95 % range of C softness ratio and Ic across clones.
4. **Confound check**: Report Spearman ρ of C, Ic, and an E-derived cell-cycle score (S+G2M composite via `scanpy.tl.score_genes_cell_cycle` on standard Tirosh 2016 gene lists) with N. Reports whether the clone-size covariate accounts for the effect.

## Deliverables (Step 7)

- `PREREG_Gate3_strict.md` (this file; committed first).
- `run_gate3_strict.py` — single deterministic entry point.
- `REPORT_Gate3_strict.md` — cohort table, per-arm metrics with CIs, sanity checks, verdict IN THE FIRST LINE.
- `gate3_strict_summary.json` (machine-readable).
- Feature tables (`.parquet`), fold assignments (`.csv`).

## Decision (Step 8, pre-committed, apply, do not reopen)

- **FAIL** → the soft-mode route to controllability / early warning is closed for snapshot data. No further observational operator test on this route. All future operator work moves to supervised settings.
- **PASS** → report as a finding about local covariance and commitment in hematopoiesis, NOT evidence for scJDO's fitted field. Do NOT add to any scJDO capability claim.

## Downstream chain (documented, not part of this gate)

After Gate 3, in this order:
1. **Gate 2 redo** with a velocity-matching loss instead of the `vel_scale` prior slot; three seeds; rerun P1–P3 with an expression-only regression baseline. Prereg first.
2. **Cell-cycle reversal figure** for the calibration paper.
3. **Gate 0d**: sign-preserving one-to-one consensus; fold-wise refits; shuffled-time null; tolerance pre-registered; precision-matrix baseline.
