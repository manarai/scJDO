# REPORT — Task A: Gate 1 v2 full-cohort rerun

**Base**: PREREG_TaskA_Gate1_v2_fullcohort.md (`7f7888e`) + amendments `b576308` (am1, ~14K) → `88b320e` (am2, ~9.6K, voided) → `e2731f6` (am3, ~14K + 5,000 epochs).

**Launched**: 2026-09-28T00:03 (start of the amendment-3 relaunch). **Fits completed**: 2026-09-28T01:30 (~87 min for 3 seeds sequentially on M2 laptop CPU, one Task A subprocess at a time). **Eval completed**: 2026-09-28T01:35.

**Runner**: `reproducibility/gates_r26/gate1/run_gate1_v2_fullcohort.py`; per-seed subprocess `_scjdo_one_seed_taskA.py`; evaluation `run_gate1_v2_fullcohort_eval.py`. Log: `taskA_run.log`.

## Cohort table

| Cohort | Cells fit on (scJDO substrate) | Cells classified (day-2 barcoded eligible) | Barcoded cells retained in fit | Eligible clones | Positive fraction |
|:---|:---:|:---:|:---:|:---:|:---:|
| 154-cell v2 (Gate 1 v2, `2d5f60e`) | 12,550 (myeloid compute-subsample) | 154 | 4,550 barcoded day-2 in fit substrate | 106 | Neut 0.61 / Mono 0.39 |
| **Full eligible cohort (Task A, this report)** | **14,000** (`4,638 barcoded day-2` + `9,362 unbarcoded day-2` fill, seed 0) | **970** | **4,638 (100 %)** | **632** | **Neut 0.539 / Mono 0.461** |

Notes:
- Fit substrate under amendment 3 retains all 4,638 barcoded day-2 cells; the subsample of ~9,362 is applied only to unbarcoded day-2 cells to hit the ~14,000 total.
- The eligible-clone count (632) is larger than the base prereg's estimate (494) because the clone-late-fate parquet has more clones passing the `≥ 2 late Neu ∪ Mono AND ≥ 80 % purity` rule than initially counted from the compute-subsampled substrate; the E baselines and S_FA projection here operate on the full 28,249 day-2 cohort, so all such clones are visible. The classification cohort grows accordingly from 759 to 970 cells.

## Five-row arm table

`LogisticRegression(penalty="l2", C=1.0, class_weight="balanced", solver="liblinear", max_iter=1000)`, StandardScaler; 5-fold `GroupKFold` × 5 CV seeds ∈ {42, 43, 44, 45, 46} = 25 outer folds per arm. `S_FA` averaged over 3 scJDO seeds within each outer fold (per-fold-mean).

| Arm | Feature dims | Mean AUROC | Fold SD | ΔAUROC vs E_FA | 95 % CI | Excl. zero? |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| E_PCA | 30 | 0.8672 | 0.033 | +0.0030 | (paired-bootstrap CI computed vs S_FA/best-baseline only in the runner; consistent with E_PCA within ~0.005 of E_FA) | — |
| E_FA | 30 | 0.8642 | 0.031 | reference | — | — |
| E_scVI | 30 | 0.8786 | 0.032 | +0.0144 | (best baseline; used as anchor for the primary contrast) | — |
| **S_FA** | 3 | **0.5540** | S_FA_seed0 0.5414 / seed1 0.5744 / seed2 0.5463 (per-scJDO-seed) | **−0.3102** | **[−0.3244, −0.2974]** | **yes** |
| E_FA + S_FA | 33 | 0.8631 | seed0 0.8634 / seed1 0.8634 / seed2 0.8627 | −0.0011 | [−0.0024, +0.0001] | no |

Additional paired-bootstrap contrasts (from `gate1_v2_fullcohort_summary.json`):

| Contrast | Δ | 95 % CI | Excl. zero? |
|:---|:---:|:---:|:---:|
| S_FA − E_PCA | −0.3132 | [−0.3289, −0.2991] | yes |
| S_FA − E_scVI (primary contrast vs best baseline) | −0.3246 | [−0.3398, −0.3092] | yes |

**Log loss**: not computed in this eval run; the runner emits AUROC and paired bootstrap CIs only. Given the −0.31 AUROC gap with a CI that excludes zero on the negative side by ~30 σ, the log-loss story adds nothing to the verdict; the summary JSON does not carry it.

## Pre-registered success criteria (from base prereg §"Success criteria")

- **Primary**: mean(S_FA) ≥ mean(best of {E_PCA, E_FA, E_scVI}) + 0.03 AND 95 % CI on (S_FA − best baseline) excludes zero on the positive side.
  - Result: mean(S_FA) − mean(E_scVI) = **−0.3246**, CI **[−0.3398, −0.3092]**. Excludes zero on the *negative* side. **PRIMARY: FAIL.**
- **Secondary**: mean(S_FA) > mean(E_FA); mean(S_FA) > mean(E_scVI).
  - Result: mean(S_FA) < mean(E_FA) and < mean(E_scVI) by ~0.31. **SECONDARY: FAIL both.**
- **Additional readout**: mean(E_FA + S_FA) − mean(E_FA) with 95 % CI.
  - Result: **−0.0011** [−0.0024, +0.0001] — CI straddles zero on the positive side; scJDO adds nothing to expression baseline.

## Verdict (one line)

**Gate 1 v2 full-eligible-cohort: FAIL.** scJDO per-cell operator features do NOT predict Neu-vs-Mono clone commitment on the LARRY day-2 full eligible cohort beyond expression baselines; the negative direction of Gate 1 v2 (154-cell cohort) is preserved on the full eligible cohort (970 cells / 632 clones) with a paired ΔAUROC of −0.31 vs the best baseline (E_scVI) and a CI that excludes zero on the negative side.

## EL10 → EL10b in EVIDENCE_LEDGER.md

**EL10** is marked superseded by **EL10b** (this row) as the canonical Gate 1 v2 fate-prediction row.

**EL10b** (new row):

- Claim: scJDO per-cell operator features do NOT predict Neu-vs-Mono clone commitment on LARRY day-2 full eligible cohort beyond expression baselines; the negative holds when the cohort grows from 154 cells to the full eligible set.
- Source: Task A (Gate 1 v2 full-cohort rerun).
- Key numbers: E_PCA 0.867 / E_FA 0.864 / E_scVI 0.879 / S_FA 0.554 / E_FA + S_FA 0.863; ΔAUROC(S_FA − E_scVI) = −0.325 95 % CI [−0.340, −0.309]; ΔAUROC(S_FA − E_FA) = −0.310 95 % CI [−0.324, −0.297]; ΔAUROC(E_FA + S_FA − E_FA) = −0.001 95 % CI [−0.002, +0.000]. N = 970 cells / 632 clones (5-fold GroupKFold × 5 CV seeds). S_FA 3 scJDO seeds spread: 0.541 / 0.574 / 0.546 (range 0.033).
- Substrate: field trained on ~14K-cell day-2 subsample (all 4,638 barcoded + 9,362 unbarcoded fill); E baselines on all 28,249 day-2 cells.
- Commit: Task A commit.
- Fig/Table: Fig 4A (two-cohort bar chart with 154 v2 and full eligible rows).
- Status: negative.

## Deliverables

- This report.
- `reproducibility/gates_r26/gate1/TASKA_PROTOCOL_CHECK.md` (updated with completion status).
- `reproducibility/gates_r26/gate1/gate1_v2_fullcohort_summary.json` — five-arm mean AUROCs, per-scJDO-seed spread, paired bootstrap CIs, verdict string.
- `reproducibility/gates_r26/gate1/gate1_v2_fullcohort_aucs.npz` — per-fold AUROC arrays for all arms + per-seed for S_FA and E_FA + S_FA.
- `reproducibility/gates_r26/gate1/gate1_v2_taskA_features.npz` — five-arm features aligned to full 28,249 day-2 cohort, with cohort mask, label and clone group.
- Per-seed scJDO caches (fit-only + projected): `gate1_v2_taskA_scjdo_seed{0,1,2}.pkl` and `.fitonly.pkl` under `reproducibility/gates_r26/gate1/`.
- `reproducibility/gates_r26/gate1/taskA_run.log` (single log spans two invocations: the first crashed after all three seeds completed on a post-fit `KeyError: 'pseudotime'` bug in `run_gate1_v2_fullcohort.main`; the fix reloads the 28K cache after the seed loop so it picks up the pseudotime that `_scjdo_one_seed_taskA.project_to_full_cohort` writes back. The second invocation cache-hit all three seeds and completed. Neither the fits nor the numbers were changed by the fix.)
