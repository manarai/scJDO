# PREREG — Gate 1 v2: reduced-row structure per user request

**Status**: pre-registered before compute for the v2 rerun. Supersedes the row structure of `PREREG_Gate1.md` (which itself remains authoritative for substrate, cohort, classifier, and success criteria). Frozen against `5c5e8c7` (Gates r26 DECISION).

## Change from v1

User requested (2026-09-27) that Gate 1 report exactly these five rows:
- `E_PCA`  — 30 PCs on scaled log-normed HVG(2000).
- `E_FA`   — 30 FactorAnalysis factors on the same scaled HVG matrix (matches Fig 3 pipeline convention).
- `E_scVI` — 30-D scVI latent on raw counts.
- `S_FA`   — scJDO features computed from `fit_drift_branches` fit on `X_FA` rep.
- `E_FA + S_FA` — concat.

Row `F` (was PCA30 aliased to E), `C` (local covariance), `K` (Palantir fate probs), and the E+S / F+S / C+S combinations are dropped from the reported table. All other v1 constraints (substrate, cohort, classifier, group-k-fold by clone, 3 CV seeds, paired-bootstrap CI, primary + secondary success criteria) carry over unchanged.

## Substrate

Same 12,550-cell subsample from PREREG_Gate1 (bd6fb0d amendment). Same classification cohort (154 day-2 cells × 106 clones from clone-purity rule).

## Preprocessing changes

- `E_PCA`: `sc.tl.pca(n_comps=30, zero_center=True)` on scaled HVG(2000). Corresponds to the "X_pca=X_fa" substitute used in v1.
- `E_FA`: `sklearn.decomposition.FactorAnalysis(n_components=30, random_state=0)` fit on the DENSE scaled HVG(2000) matrix. Tractable at 12,550 × 2 000 dense = 100 MB. Stored under `obsm["X_FA"]`; `varm["fa_loadings"]` = `fa.components_.T` (n_hvg, 30) so scJDO reads gene-space loadings correctly.
- `E_scVI`: `scvi.model.SCVI(n_latent=30, n_layers=2, n_hidden=128)` on the raw-count subset of the 12,550 cells. Layer input = raw counts (NOT log-normed). Standard `.setup_anndata` on the raw layer, `.train(max_epochs=200, early_stopping=True, batch_size=1024)`. Latent stored under `obsm["X_scVI"]`. Seed = 0 for the network initialization.

## scJDO fit change

Palantir pseudotime is recomputed on `X_FA` (rather than the PCA substitute) so the trajectory is anchored to the same representation scJDO uses. `fit_drift_branches` runs with `rep="X_FA"`, `vel_scale = 0.0` (Gate 0c default), `bias_strength = 1.5`, `n_epochs = 3000`, `grid_size = 150`, `n_archetypes = 5`. Three seeds (0, 1, 2). Per-seed subprocess wall-clock guard (45 min) as in v1.

## Feature bundles

- `E_PCA`, `E_FA`, `E_scVI` are each 30-D per-cell.
- `S_FA` is per-cell: Re(λ_max) at cell's pseudotime bin + leading-Jacobian projection + within-substrate consensus archetype activation (Gate 0c procedure applied within the 3 scJDO seeds) — per branch × 2 branches = 6-D. If the LARRY-substrate consensus procedure yields no passing cluster in a branch, that archetype feature is dropped and S becomes 4-D as declared in v1 amendment `39c35d3`.
- `E_FA + S_FA` is the 36-D concat (or 34-D if archetype dropped in either branch).

## Classifier + evaluation (unchanged from v1)

- `LogisticRegression(penalty='l2', C=1.0, class_weight='balanced', solver='liblinear', max_iter=1000)`, standardised features.
- `GroupKFold(n_splits=5)` with clone as group, shuffled by CV seed ∈ {42, 43, 44}.
- 15 AUROC values per feature set (5 folds × 3 CV seeds). For S_FA the 3 scJDO seeds give 3 × 15 = 45 AUROCs; report the per-fold mean across scJDO seeds and the across-scJDO-seed spread.
- Paired bootstrap over folds for the 95 % CI on any difference.

## Success criteria (unchanged from v1 primary + secondary)

Applied to the new row structure:
- **Primary**: mean(S_FA) ≥ mean(best of {E_PCA, E_FA, E_scVI}) + 0.03 AND paired-bootstrap 95 % CI on (S_FA − best baseline) excludes zero on the positive side.
- **Secondary**: mean(S_FA) > mean(E_FA); mean(S_FA) > mean(E_scVI). Both should hold for a clean pass.
- Additional readout: mean(E_FA + S_FA) − mean(E_FA), and its 95 % CI, to test whether S_FA adds ON TOP of E_FA even if it does not beat it alone.

## Not part of this rerun

- Local covariance features (dropped from the reported table; still available in `gate1_features.npz` for reference).
- CellRank / Palantir fate probs (dropped from the reported table).
- v1's per-branch feature enumeration and per-branch AUROC report.

## Expected wall clock

- Preprocessing FA + scVI: ~10 min.
- Palantir on X_FA: ~1 min.
- scJDO fit × 3 seeds on X_FA: ~50 min (subprocess timeout 45 min each).
- Classifier evaluation: <1 min.
- **Total: ~70 min**.
