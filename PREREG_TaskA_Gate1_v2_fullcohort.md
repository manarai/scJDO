# PREREG — Task A: Gate 1 v2 full-cohort rerun (day-2-only pipeline)

**Status**: pre-registered before compute. Frozen against `c53683a` (Task 7 draft-1 commit). No changes after commit.

## Purpose

Rerun the Gate 1 v2 five-row table on the FULL LARRY day-2 cohort with the day-2-only preprocessing pipeline of Gate 3 strict. Supersedes ledger row **EL10** (marks it as "superseded"); the new row is **EL10b**.

## What changes from Gate 1 v2

- **Substrate**: ALL 28,249 day-2 cells from the Klein LARRY in vitro release. No compute-driven subsampling of the myeloid trajectory (v2 used a 12,550-cell subsample; only 154 clones passed the cohort filter).
- **Pipeline**: Gate 3 strict's day-2-only preprocessing:
  - normalize_total(target_sum=1e4) + log1p on day-2 raw counts;
  - highly_variable_genes(n_top_genes=2000, flavor="seurat") on day-2 log-normed data;
  - scale(max_value=10, zero_center=True) on the day-2 HVG matrix;
  - FactorAnalysis(n_components=30, random_state=0, max_iter=200) fit on the dense scaled matrix → X_FA.
- **Pseudotime**: Palantir on day-2 X_FA (as in Gate 3 strict sanity control, where softness_ratio and Ic tracked τ at Spearman ≈ 0.5 — pseudotime is meaningful on day-2 alone).
- **scJDO fit**: on the day-2 substrate with the day-2 Palantir pseudotime. Same hyperparameters as Gate 1 v2 (`vel_scale = 0`, `bias_strength = 1.5`, `hidden = 256`, `depth = 4`, `n_epochs = 3000`, `grid_size = 150`, `n_archetypes = 5`). 3 scJDO seeds (0, 1, 2). Single-branch fit (`branch_names = ["Ery"]` placeholder branch spanning all day-2 cells; no branch masks since we don't have a terminal state at day 2).
- **Classification cohort**: day-2 cells whose CLONE resolves to Neut vs Mono at day 4/6 using the original clone-purity rule from Gate 1 v2:
  - clone has ≥ 1 barcoded day-2 cell AND ≥ 2 late-time Neu ∪ Mono cells;
  - clone's fraction of Neut among late Neu ∪ Mono ≥ 0.8 → label Neut; ≤ 0.2 → label Mono; else excluded.
- **Feature bundles** (per cell, aggregated to per-cohort-cell arrays):
  - E_PCA (30-D) — 30 PCs on scaled day-2 HVG (via `sc.tl.pca(n_comps=30, svd_solver="arpack")`).
  - E_FA (30-D) — FactorAnalysis(30) on scaled day-2 HVG.
  - E_scVI (30-D) — scVI trained on RAW day-2 counts, `n_latent=30, n_layers=2, n_hidden=128, max_epochs=200, batch_size=1024, early_stopping=True`.
  - S_FA (per prereg formula) — Re(λ_max) per cell + leading-J projection per cell + within-cohort consensus archetype activation per branch. Single-branch here so 3 features per cell (not 6 as in Gate 1 v2's two-branch fit). If the within-substrate consensus procedure yields no passing cluster, the archetype feature is dropped and S_FA is 2-D.
  - E_FA + S_FA = 33-D concat (or 32-D if archetype dropped).

## Classifier + evaluation (frozen)

- `LogisticRegression(penalty="l2", C=1.0, class_weight="balanced", solver="liblinear", max_iter=1000)`, standardised features (StandardScaler fit on train, applied to test).
- **5-fold `GroupKFold` grouped by clone × 5 CV seeds** (per user's "5×5") ∈ {42, 43, 44, 45, 46}. 25 outer folds per feature set.
- For S_FA the 3 scJDO seeds give 3 × 25 = 75 AUROC values; report the per-fold mean across scJDO seeds and the across-scJDO-seed spread.
- Paired clone-level bootstrap 2000 resamples for CIs on paired arm differences.

## Success criteria (unchanged from Gate 1 v2 primary + secondary)

- **Primary**: mean(S_FA) ≥ mean(best of {E_PCA, E_FA, E_scVI}) + 0.03 AND 95 % CI on (S_FA − best baseline) excludes zero on the positive side.
- **Secondary**: mean(S_FA) > mean(E_FA); mean(S_FA) > mean(E_scVI).
- **Additional readout**: mean(E_FA + S_FA) − mean(E_FA) with 95 % CI.

## Expected cohort size (verified against raw files at prereg time)

- Total day-2 cells: 28,249.
- Day-2 cells with any clone barcode: 4,638.
- Clones with ≥ 1 barcoded day-2 cell AND ≥ 2 late Neu ∪ Mono cells at ≥ 80% purity: 494 (248 Neut-biased + 246 Mono-biased) → 759 day-2 cells (400 Neut + 359 Mono).

These are the 759 target cells / 494 clones referenced in the original Gate 1 prereg cohort table.

## Compute budget

- Preprocess + FA + PCA + scVI: ~15 min (scVI on 28K cells).
- Palantir on day-2 X_FA: ~1 min.
- scJDO × 3 seeds on 28,249 cells: ~30-40 min per seed × 3 = ~90-120 min. Per-seed subprocess with 60-min wall-clock cap.
- Classifier eval: <5 min.
- **Total: ~2.5 hours.**

## Compute-scale amendment (recorded before scJDO features are extracted)

First run: scJDO fit_drift on 28,249 day-2 cells OOM-killed 2 of 3 seeds at ~4.3 GB peak (36-GB machine, competing background processes). To keep the 3-seed prereg AND all clone-cohort cells, the scJDO FIT SUBSTRATE is subsampled to **~14,000 day-2 cells** while E baselines (E_PCA, E_FA, E_scVI) remain on all 28,249 cells:

- keep = (all 4,638 day-2 cells carrying any clone barcode — the 759 cohort cells are a strict subset) ∪ (random ~9,400 day-2 Undifferentiated cells without clone, seed 0), total ~14,000.
- FA and PCA and scVI are computed on all 28,249 day-2 cells (unchanged; E_FA/E_PCA/E_scVI features are extracted at cohort cells from these full-cohort reps).
- Palantir pseudotime is recomputed on the ~14,000-cell subsample so scJDO sees a self-consistent pseudotime.
- scJDO fit_drift on ~14,000 cells; per-cell J_tensor/eigvec features for cohort cells extracted via nearest-bin from the fit's pseudotime bins.

This is a scale-only amendment; all 759 target cells / 494 clones are in the scJDO fit substrate. It preserves the "3 seeds" and the "day-2 only" constraints. Recorded before scJDO features exist.


## Ledger integration (frozen)

- New ledger row **EL10b** — mean AUROCs and 95 % CIs from the full-cohort rerun.
- **EL10** relabelled as "superseded (subsampled substrate; see EL10b for canonical numbers)".
- `EVIDENCE_LEDGER.md` amended in the same commit as the report.
- Ledger amendment is NOT a manuscript edit; the manuscript is frozen at draft 1 until Tasks A–C complete.

## Deliverables

- `PREREG_TaskA_Gate1_v2_fullcohort.md` (this file).
- `reproducibility/gates_r26/gate1/run_gate1_v2_fullcohort.py` — single deterministic entry.
- `reproducibility/gates_r26/gate1/REPORT_Gate1_v2_fullcohort.md`.
- `reproducibility/gates_r26/gate1/gate1_v2_fullcohort_summary.json`.
- Per-seed scJDO caches + features_per_cell + features_per_clone parquets under `reproducibility/gates_r26/gate1/data/`.
- `EVIDENCE_LEDGER.md` updated with EL10b row + EL10 marked superseded (same commit).
