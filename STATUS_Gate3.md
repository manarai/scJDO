# STATUS — Gate 3 strict

## Existence check

- `New_analysis/gates_r26/gate3/PREREG_Gate3_strict.md` — **present** (frozen commit `5055fd5`, 2026-09-27).
- `New_analysis/gates_r26/gate3/REPORT_Gate3_strict.md` — **present** (committed `6edbb1e`, 2026-09-27).

Gate 3 strict already ran on 2026-09-27 as the fourth follow-up in the r26 gate chain, per its own frozen prereg. No re-run required for Task 0. Copying the arm table below verbatim from `gate3_strict_summary.json`.

## Cohort

- LARRY in vitro release, day-2 cells only for feature construction.
- 130,887 total cells → 28,249 day-2 → 4,638 day-2 with any clone barcode.
- 102,638 late (day 4/6) cells used only to build the clone → fate outcome matrix.
- **520 eligible clones** (≥ 1 barcoded day-2 cell AND ≥ 5 mature descendants).
- Positive fraction `mean(y_mixed = 1)` = **0.275**.
- Mature-descendant counts per eligible clone: mean 14.2, median 9.5, min 5, max 144, IQR [6, 16].

## Arm table (frozen from Gate 3 strict, 5-fold GroupKFold by clone × 5 fold seeds = 25 outer folds; inner 5-fold LogisticRegressionCV/RidgeCV for hyperparameter selection; all arms include N = log(1 + descendants))

| Arm | Features | AUROC (mean, n=15 fold-seed means from 25 folds) | 95% bootstrap CI (2000 clone-level resamples) | Log loss | Δ vs Arm 1 AUROC | Paired 95% CI on ΔAUROC | Δ vs Arm 1 log loss |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Arm 1 — E   | FA30 + N               | **0.6464** | [0.5993, 0.7027] | 0.5720 | — | — | — |
| Arm 2 — E + C   | FA30 + softness_ratio + top-3 Σ_local eigvals + N | 0.6480 | [0.5990, 0.7021] | 0.5721 | +0.0001 | [−0.0122, +0.0115] | −0.0000 |
| Arm 3 — E + Ic  | FA30 + Mojtahedi Ic + N | 0.6466 | [0.5999, 0.7032] | 0.5722 | +0.0004 | [−0.0019, +0.0028] | +0.0002 |
| Arm 4 — E + C + Ic | FA30 + C + Ic + N   | 0.6481 | [0.6006, 0.7031] | 0.5717 | +0.0012 | [−0.0109, +0.0124] | −0.0006 |

## Verdict (as pre-registered)

**FAIL Gate 3 strict.** Pass criterion required Arm 2 OR Arm 3 to satisfy BOTH (a) ΔAUROC ≥ 0.02 vs Arm 1 with 95 % CI EXCLUDING zero on the positive side, AND (b) strictly lower log loss than Arm 1.

- Arm 2: ΔAUROC = +0.0001 (< 0.02); 95 % CI includes zero; Δlogloss = 0.0000. Fails both.
- Arm 3: ΔAUROC = +0.0004 (< 0.02); 95 % CI includes zero; Δlogloss = +0.0002 (worse). Fails both.
- Arm 4 not decisive per prereg.

## Sanity checks (all passed BEFORE the primary evaluation, per prereg Step 6)

- ID alignment: 520 clones, all `clone_id` unique.
- Pseudotime positive control (day-2 Palantir on FA30):
  Spearman(softness_ratio, τ) = +0.473; Spearman(Ic, τ) = +0.491.
  Features carry the expected structural signal at the single-cell level.
- Non-trivial variance: `softness_ratio` mean 2.88, std 2.72, range [1.10, 23.14]; `Ic` mean 0.0043, std 0.0011, range [0.0011, 0.0110].
- Clone-size confound (Spearman with `log(1 + n_desc)`): softness_ratio = −0.09, Ic = +0.09, cell-cycle score = +0.03. Small; not driving the null result.

## Downstream (per pre-committed Step 8 of Gate 3 strict prereg)

Applied: **the soft-mode route to controllability / early warning is closed for snapshot data.** No further observational operator test proposed on this route.

## Pointers

- Prereg: `New_analysis/gates_r26/gate3/PREREG_Gate3_strict.md` (5055fd5).
- Report: `New_analysis/gates_r26/gate3/REPORT_Gate3_strict.md` (6edbb1e).
- Summary JSON: `New_analysis/gates_r26/gate3/gate3_strict_summary.json`.
- Runner: `New_analysis/gates_r26/gate3/run_gate3_strict.py`.
- Fold assignments: `New_analysis/gates_r26/gate3/gate3_strict_folds.csv`.
- Feature tables: `New_analysis/gates_r26/gate3/data/features_per_{cell,clone}.parquet`.
