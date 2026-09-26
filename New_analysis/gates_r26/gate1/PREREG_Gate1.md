# PREREG — Gate 1: LARRY early fate prediction (neutrophil vs monocyte)

**Status**: pre-registered before compute. Frozen against `598a160` (Gate 0c report).

## Question

Do scJDO per-cell operator features let a linear classifier predict, from a day-2 progenitor's transcriptional state alone, whether its clone will resolve to Neutrophil versus Monocyte at day 4/6 — better than expression-only baselines (PCA/FA), better than local-covariance features (Gate 0b baseline), and better than CellRank fate probabilities that use no clone labels?

## Dataset (frozen)

- **Source**: Weinreb et al. 2020, Klein-lab in vitro state-fate LARRY release. Files downloaded from `https://kleintools.hms.harvard.edu/paper_websites/state_fate2020/`:
  - `stateFate_inVitro_normed_counts.mtx.gz` (130,887 × 25,289; already library-normalised per the release)
  - `stateFate_inVitro_gene_names.txt.gz`
  - `stateFate_inVitro_metadata.txt.gz` (Time point ∈ {2, 4, 6}, Cell type annotation)
  - `stateFate_inVitro_clone_matrix.mtx.gz` (130,887 × 5,864 sparse binary; 49,302 cell-clone links)
- **Assembled**: AnnData `larry_full.h5ad` at `New_analysis/gates_r26/gate1/data/`.
- **Trajectory subset used for scJDO fit and pseudotime**: cells labelled `Undifferentiated`, `Neutrophil`, or `Monocyte` (all time points) — the neutrophil/monocyte myeloid arm. Basophil/Mast/Meg/Erythroid/Lymphoid excluded from the substrate.
- **Compute-scale amendment (recorded before compute, verified no scJDO features written yet)**: the myeloid subset is 113,612 cells; Palantir pseudotime on the full FA-30 rep produced NaN correlations (disconnected diffusion map, unsalvageable in single-machine memory). To keep the trajectory tractable, the substrate is subsampled to `keep = (all day-2 cells carrying any clone barcode) ∪ (all day-4/6 Neutrophil ∪ Monocyte) ∪ (10,000 random day-4/6 Undifferentiated) ∪ (5,000 random day-2 Undifferentiated without clone)`. This is a compute-tractability decision, not an analytic one — the classification cohort (day-2 cells whose clone resolves to Neut vs Mono) is fully preserved because *every* day-2 cell with a clone is retained. Random seed for the two Undiff subsamples = 0. Neut/Mono coverage at the differentiated end is exhaustive; only the Undiff density is reduced. The subsampled substrate is recorded as `larry_myeloid_preproc.h5ad`.
- **Classification cohort (evaluation targets, frozen at prereg)**:
  - A day-2 cell is included if it carries at least one LARRY clone label AND that clone has ≥ 2 late-time (day 4 or 6) cells among Neutrophil ∪ Monocyte, AND the fraction of Neutrophil among those late cells is either ≥ 0.8 (label = Neut) or ≤ 0.2 (label = Mono). Clones in [0.2, 0.8) are excluded.
  - Under these rules the substrate holds **400 day-2 cells in Neut-biased clones + 359 day-2 cells in Mono-biased clones = 759 cells** across **248 Neut clones + 246 Mono clones = 494 clones** (verified against the raw files at prereg time; will re-verify at run time and abort the gate if the counts drift).

## Features (frozen)

### Preprocessing
- `sc.pp.normalize_total` + `log1p` on the raw counts (the release is already library-normalised, but re-run for reproducibility of downstream tools that expect log1p input).
- HVG selection top 2,000 by seurat_v3.
- `X_pca` = 50 PCs on HVG-scaled X.
- `X_fa` = 30 factors via `sklearn.decomposition.FactorAnalysis` on the 50 PCs (matches r24 pipeline).

### scJDO substrate + fit
- Palantir pseudotime on X_fa (start cell = most Undifferentiated day-2 cell by SPRING coords).
- Branch masks: {Neutrophil, Monocyte}; branch probabilities from Palantir.
- `fit_drift_branches` with the Gate 0c chosen configuration: **vel_scale = 0.0**, bias_strength = 1.5, hidden = 256, depth = 4, n_epochs = 5000, n_archetypes = 5, grid_size = 200, kernel windowing, adaptive bandwidth. Seeds = {0, 1, 2}.
- Per branch, extract per-cell:
  - **Re(λ_max)**: real part of leading eigenvalue of the interpolated Jacobian at the cell's pseudotime bin.
  - **Leading-Jacobian projection**: cell's X_fa vector projected onto the leading real eigenvector of its interpolated Jacobian.
  - **Consensus archetype activation**: single activation of a passing-cluster centroid identified by applying the Gate 0c consensus procedure (single-linkage on `1 − |cos|`, threshold R ≥ 80 % of scJDO seeds) to the LARRY-substrate patterns across the 3 scJDO seeds. **NOTE on scope of Gate 0c centroids**: the `consensus_centroids_V0.npz` file produced in Gate 0c was learned in the marrow Ery FA basis (30-D), which is a linear basis distinct from LARRY's own FA basis. Applying that specific centroid to LARRY operators is not meaningful in operator space (different coordinate systems). The correct downstream use of Gate 0c is the *procedure*, not the specific centroid matrix. Amendment recorded before any Gate 1 compute (commit follows this edit). Activations are computed by regressing (least-squares) the per-cell per-bin J_cell onto the per-substrate consensus centroid (branch-specific). If Gate 0c's within-substrate consensus procedure finds no passing cluster on LARRY (R ≥ 80 % of the 3 scJDO seeds contributing to a single cluster), this feature is dropped and S becomes a 2-feature-per-branch = 4-feature bundle; this outcome is recorded, not adjusted for.
- These give **3 scJDO features per cell per branch → 6 features per cell** (Neut branch + Mono branch), or 4 features if the LARRY consensus procedure yields no passing cluster. Reported as three (or two) per branch when arm-wise comparison is needed.

### Baselines (frozen)
1. **Expression: PCA30** — 30 PCs of the log-normed HVG matrix.
2. **Expression: FA30** — 30 FA factors (same as scJDO input rep).
3. **Local-covariance features (Gate 0b baseline)**: the (T, D, D) covariance tensor built at the same grid, same bandwidth, decomposed with K = 5 semi-NMF. Per-cell features = the 5 activations at the cell's pseudotime bin + local-covariance leading eigenvalue at that bin (6 features).
4. **CellRank (no clone labels)**: CytoTRACE + Palantir kernel, `GPCCA` estimator with 2 macrostates matched to {Neut, Mono}. Per-cell feature = the 2 fate probabilities (Neut, Mono). No clone info.
5. **Dynamo**: NOT applicable — the Klein normed-counts release does not include spliced/unspliced. Recorded as a limitation; not a baseline.
6. **CoSpar**: NOT applicable in this gate — cospar is supervised on clone labels, and the user's own protocol says "without clone labels." Cospar would leak the label. Not a baseline here.

## Classifier + evaluation (frozen)

- **Classifier**: `LogisticRegression(penalty='l2', C=1.0, class_weight='balanced', solver='liblinear', max_iter=1000)` for every feature set. No hyperparameter tuning per split.
- **Splits**: `GroupKFold(n_splits=5)` with `groups = clone_id` (a cell's clone is its group). No clone can appear on both sides of any fold. Repeated across **CV seeds ∈ {42, 43, 44}** (three repeats of the 5-fold with shuffled group order via `sklearn.utils.shuffle` on the clone index list before the split).
- **Feature set variants** evaluated against each other:
  - E: PCA30
  - F: FA30
  - C: local-covariance features (6-D)
  - K: CellRank fate probabilities (2-D)
  - S: scJDO features (6-D)
  - E+S, F+S, C+S: expression / local-cov plus scJDO features (does S add on top?)
- **Metric**: AUROC (Neut = 1, Mono = 0). One AUROC per fold → 15 values per feature set (5 folds × 3 CV seeds). Report mean and 95% bootstrap CI over the 15 fold values.
- **scJDO feature stability**: the S features change with scJDO fit seed (0, 1, 2). Report S metrics as mean ± spread across the 3 scJDO seeds, in addition to the fold-level CI. This means the S feature set has 3 × 15 = 45 AUROC values; other feature sets have 15.

## Success criteria (frozen)

**Primary**: mean(S) ≥ mean(best of {E, F, C, K}) + 0.03 AND the 95% CI of the AUROC difference (S − best baseline) excludes zero, computed via paired bootstrap over the 15 folds (using per-fold means over the 3 scJDO seeds for S).

**Secondary (both must hold or interpretation reads as "S beats some baselines but not all")**:
- mean(S) > mean(C) (Jacobian beats local covariance)
- mean(S) > mean(F) (scJDO beats its own input rep)

**Reported regardless**: mean(E+S) − mean(E), mean(F+S) − mean(F), mean(C+S) − mean(C). Are these positive with CI excluding 0? If so, scJDO features add on top of the baseline; if S alone is worse than E/F/C but E+S/F+S/C+S add value, that is a different (weaker) positive claim.

## Stop rules (frozen)

- If success criteria met AND E+S/F+S adds on top → **PASS Gate 1**; proceed to Gate 2.
- If mean(S) is within 0.03 of the best baseline OR CI(S − best baseline) includes 0 → **FAIL Gate 1**; scJDO features do not carry information beyond expression/covariance/kernel-fate-probability baselines on this task. Per decision table → Calibration paper track (with negatives).
- If mean(S) is > 0.03 BELOW the best baseline → **FAIL Gate 1 negatively**; scJDO features are actively worse. Same decision.

## Reported quantities

- Per-feature-set mean AUROC + 95 % CI (bootstrap over folds).
- Paired difference (S − each baseline) with 95 % CI and p-value from paired bootstrap.
- Confusion matrix at threshold 0.5, per feature set (mean over folds).
- Per-scJDO-seed AUROC for S (spread across seeds 0, 1, 2).
- Time/RAM per fit.

## Not part of the gate

- Deep-learning classifiers; only regularised logistic regression per prereg.
- Feature selection within scJDO (all 6 features enter S together).
- Fate prediction beyond neut vs mono (basophil/mast/erythroid are excluded from the substrate).
- Days > 2 as the classification target (day-2 is the frozen decision point; day-4/6 is only the label source).

## Expected wall clock

- Preprocessing + h5ad build: ~10 min.
- Palantir pseudotime on the myeloid substrate (~72k Undiff + 22k Neut + 19k Mono ≈ 113k cells): ~15 min.
- `fit_drift_branches` × 3 seeds on the full substrate: ~15 min per seed × 3 = 45 min (2-branch fit is slower than 1-branch).
- CellRank fitting: ~10 min.
- Local-covariance tensor + semi-NMF: <5 min.
- Classifier CV × 3 CV seeds × 8 feature sets: <5 min total.
- **Total: ~1.5 h**.
