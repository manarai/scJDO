# REPORT — Gate 1 v2: LARRY early fate prediction (E_PCA / E_FA / E_scVI / S_FA / E_FA + S_FA)

**Prereg**: `PREREG_Gate1_v2.md` (frozen `726828b`; inherits substrate/cohort/classifier/success criteria from `PREREG_Gate1.md`).
**Compute**: ~65 min build (FA + scVI 5 min + palantir 22 s + 3 × scJDO on X_FA ~15 min each) + <1 min eval.

## Verdict — **FAIL Gate 1 v2**

Per the frozen primary criterion:
> mean(S_FA) ≥ mean(best baseline of {E_PCA, E_FA, E_scVI}) + 0.03 AND paired-bootstrap 95 % CI on (S_FA − best baseline) excludes zero on the positive side.

**S_FA is 0.322 BELOW E_FA (best baseline), CI [−0.362, −0.283] entirely negative.**

## Row-level AUROC (15 folds = 5 folds × 3 CV seeds; N=154 cells / 106 clones)

| Row | Mean AUROC | Across-fold SD | Notes |
|:---:|:---:|:---:|:---|
| E_PCA | 0.8254 | 0.0884 | 30 PCs, scaled log-normed HVG(2000) |
| **E_FA**  | **0.8299** | 0.0862 | **best baseline** — FactorAnalysis(30) on scaled log-normed HVG |
| E_scVI | 0.8281 | 0.1214 | scVI(30-D) trained 200 epochs on raw counts |
| **S_FA**  | **0.5076** | 0.096 (across scJDO seeds; per-seed means 0.457, 0.484, 0.582) | scJDO features from `fit_drift_branches` on X_FA rep |
| E_FA + S_FA | 0.8293 | | 36-D concat |

## Paired bootstrap 95 % CIs (2 000 samples over 15 folds)

| Contrast | Δ | 95 % CI | Excludes 0? |
|:---:|:---:|:---:|:---:|
| S_FA − E_PCA  | −0.3178 | [−0.3576, −0.2793] | yes (negative) |
| S_FA − E_FA   | −0.3224 | [−0.3622, −0.2827] | yes (negative) |
| S_FA − E_scVI | −0.3205 | [−0.3656, −0.2718] | yes (negative) |
| E_FA + S_FA − E_FA | −0.0006 | [−0.0058, +0.0045] | no |

## Interpretation

- **All three expression representations agree**: PCA30, FA30, and scVI(30) reach 0.825–0.830 AUROC on this task — the cohort is clearly separable by 30-D expression alone, and the choice of linear/non-linear method doesn't materially change accuracy. So the expression signal is real, robust, and rep-agnostic.
- **S_FA is at chance**: 0.508 mean across scJDO seeds. Per-seed means 0.46, 0.48, 0.58 — the fit is not only weak, it's seed-unstable on this substrate. Corrected pipeline (proper FA30, scJDO fit on `X_FA` rather than the v1 PCA substitute) yields a WORSE S value than v1 (0.62); the v1 apparent "signal" was rep-substitution noise, not a real S contribution.
- **S_FA is dominated (regularised away) inside E_FA+S_FA**: adding S_FA to E_FA moves the mean by −0.0006 (CI includes 0). L2 logistic regression at C = 1.0 discards scJDO features when informative expression features are present.
- **Compared to v1**: v1 reported S mean 0.62 vs E 0.83 (Δ −0.22). v2 tightens this to S_FA 0.51 vs E_FA 0.83 (Δ −0.32). The direction and interpretation are unchanged; the correction to the pipeline sharpens the negative.

## Notes on v2 methodology

- E_FA now uses `sklearn.decomposition.FactorAnalysis(30)` fit on the DENSE scaled HVG(2000) matrix (feasible at 12,550 × 2,000). In v1 this was substituted with PCA30 due to a 58,951-cell OOM constraint that no longer applies at 12,550.
- scJDO is refit from scratch with `rep = "X_FA"` (the true FA rep). Palantir pseudotime is recomputed on X_FA as well, so the trajectory is anchored to the same rep scJDO uses.
- E_scVI trained with default settings (`n_latent=30, n_layers=2, n_hidden=128, batch_size=1024, max_epochs=200, early_stopping=True`, CPU). Warmup converged well within 200 epochs. Latent shape (12,550 × 30) used at day-2 cohort indices only.
- Cohort, groups, splits, classifier, bootstrap unchanged from v1.

## Downstream decision — unchanged

Combined outcome across all gates:
- Gate 0 (0a/0b/0c): **PASS**.
- Gate 1 v2: **FAIL** (this report supersedes v1's row-level AUROC table; substrate + verdict are unchanged in direction and magnitude).
- Gate 2: **FAIL**.

Per the frozen decision table: {Pass, Fail, Fail} → **Calibration paper with negatives**. The v2 rerun does not shift the decision path.

## Deliverables

- Prereg: `PREREG_Gate1_v2.md` (`726828b`).
- Runners: `run_gate1_v2_build.py`, `_scjdo_one_seed_v2.py`, `run_gate1_v2_eval.py`.
- Machine-readable summary: `gate1_v2_eval_summary.json`.
- Per-seed scJDO caches: `gate1_v2_scjdo_seed{0,1,2}.pkl`.
- Combined features: `gate1_v2_features.npz`.
- Per-fold AUROC arrays: `gate1_v2_eval_aucs.npz`.
- Logs: `run_v2_build.log`, `run_v2_eval.log`.

**Superseded**: `REPORT_Gate1.md` (v1); its row-level table (E, F, C, K, S, E+S, F+S, C+S) is retained for the record but the v2 rows are canonical.
