# REPORT — Gate 3 strict

**Verdict: FAIL Gate 3 strict.** Neither Arm 2 (E + C) nor Arm 3 (E + Ic) meets the ΔAUROC ≥ 0.02 threshold with 95 % CI excluding zero.

**Prereg**: `PREREG_Gate3_strict.md` (frozen `5055fd5`, committed before any Gate-3 data touch).
**Compute**: 209 s total (Phase 3 feature build dominates: FA on (28249, 2000) plus per-cell Σ_local + Ic loops).

## Cohort table (Step 1 + Step 2 outputs)

| Quantity | Value |
|:---|:---:|
| Total cells (metadata) | 130,887 |
| Day-2 cells | 28,249 |
| Day-2 cells carrying any clone barcode | 4,638 |
| Day-4/6 cells (used only for labels) | 102,638 |
| Eligible clones (≥ 1 barcoded day-2 cell AND ≥ 5 mature descendants) | **520** |
| Positive fraction `mean(y_mixed = 1)` | 0.275 |
| Descendant counts per eligible clone | mean 14.2, median 9.5, min 5, max 144, IQR [6, 16] |

Mapping table used (major lineage groups):
- Myeloid_Neu ← Neutrophil
- Myeloid_Mono ← Monocyte
- ErythroMeg ← Erythroid, Meg
- MastGranu ← Baso, Mast, Eos
- LymphoDC ← Lymphoid, Ccr7_DC, pDC

## Sanity checks (Step 6, run BEFORE the primary evaluation)

1. **ID alignment**: 520 clones, all `clone_id` unique. OK.
2. **Pseudotime positive control** (day-2-only Palantir on FA30):
   - Spearman(softness_ratio, pseudotime) = **+0.473**
   - Spearman(Ic, pseudotime) = **+0.491**
   The features track pseudotime as expected — they carry real signal about the cell's day-2 state.
3. **Non-trivial variance across clones**:
   - softness_ratio: mean = 2.88, std = 2.72, range [1.10, 23.14].
   - Ic: mean = 0.0043, std = 0.0011, range [0.0011, 0.0110].
4. **Confound check with N = log(1 + descendants)**:
   - Spearman(softness_ratio, log_n_desc) = −0.094
   - Spearman(Ic, log_n_desc) = +0.094
   - Spearman(cc_score, log_n_desc) = +0.034
   No large clone-size confound. Cell-cycle score is essentially independent of clone size.

## Per-arm metrics (Step 4)

15 features per Arm 1, +4 for Arm 2, +1 for Arm 3, +5 for Arm 4. All arms include `log_n_desc`. Trained via 5-fold GroupKFold × 5 fold-shuffle seeds (25 outer folds); inner 5-fold LogisticRegressionCV / RidgeCV for hyperparameter selection.

| Arm | Features | AUROC (mean) | 95 % boot CI | Log loss | Spearman ρ p_mixed | Spearman ρ H_entropy |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| Arm 1 — E | FA30 + N | **0.6464** | [0.5993, 0.7027] | 0.5720 | +0.273 | +0.290 |
| Arm 2 — E + C | FA30 + 4C + N | 0.6480 | [0.5990, 0.7021] | 0.5721 | +0.272 | +0.288 |
| Arm 3 — E + Ic | FA30 + 1Ic + N | 0.6466 | [0.5999, 0.7032] | 0.5722 | +0.272 | +0.289 |
| Arm 4 — E + C + Ic | FA30 + 4C + 1Ic + N | 0.6481 | [0.6006, 0.7031] | 0.5717 | +0.271 | +0.286 |

## Paired clone-level bootstrap (2 000 resamples, Step 4 + Step 5)

| Contrast | ΔAUROC | 95 % paired CI | Δlogloss |
|:---|:---:|:---:|:---:|
| Arm 2 − Arm 1 | +0.0001 | [−0.0122, +0.0115] | −0.0000 |
| Arm 3 − Arm 1 | +0.0004 | [−0.0019, +0.0028] | +0.0002 |
| Arm 4 − Arm 1 | +0.0012 | [−0.0109, +0.0124] | −0.0006 |

## Pass criterion (Step 5)

**Required**: Arm 2 OR Arm 3 satisfies BOTH
  (a) ΔAUROC ≥ 0.02 vs Arm 1 with 95 % CI EXCLUDING zero on the positive side, AND
  (b) log loss strictly lower than Arm 1.

**Observed**:
  - Arm 2: ΔAUROC = +0.0001 (< 0.02); 95 % CI [−0.012, +0.012] INCLUDES zero; Δlogloss = 0.0000 (not lower). Fails both.
  - Arm 3: ΔAUROC = +0.0004 (< 0.02); 95 % CI [−0.002, +0.003] INCLUDES zero; Δlogloss = +0.0002 (higher, i.e. worse). Fails both.

Neither arm passes. **Verdict: FAIL Gate 3 strict.**

Arm 4 is reported per Step 5 but does not decide.

## Interpretation

- The features work as advertised at the SINGLE-CELL level: both `softness_ratio` and `Ic` correlate with day-2 pseudotime at ρ ≈ 0.5. So the features are not broken; they carry the expected structural signal.
- When AGGREGATED per clone and used to predict clone commitment status, C and Ic add nothing to a Factor-Analysis-of-expression baseline. Arm 2 and Arm 3 each move mean AUROC by less than 0.001 and their 95 % paired CIs include zero.
- The expression baseline itself achieves AUROC ≈ 0.65 — modest but non-trivial signal about clone commitment lies in day-2 mean expression. Local covariance softness at day 2 does not sharpen that signal.
- The clone-size covariate N was included in every arm and is only weakly correlated with C and Ic (ρ ≤ 0.10), so the failure is not a clone-size confound.

## Decision (Step 8, applied)

Per the pre-committed decision:

> If FAIL: the soft-mode route to controllability and early warning is closed for snapshot data. Do not propose another observational operator test. All further operator work moves to supervised settings.

**Applied.** No further observational operator test is proposed. Future operator work moves to supervised settings.

## Deliverables

- Prereg: `PREREG_Gate3_strict.md` (`5055fd5`).
- Runner: `run_gate3_strict.py` (single deterministic entry point).
- This report.
- `gate3_strict_summary.json` — machine-readable numbers.
- `data/features_per_cell.parquet` (28,249 × 38: FA30, C4, Ic, pseudotime, cc_score, cell_index).
- `data/features_per_clone.parquet` (520 × 44: aggregated features + labels).
- `gate3_strict_folds.csv` — full fold assignments (25 folds × 5 seeds).
- Run log: `run_gate3_strict.log`.

## Downstream chain (not part of this gate)

Per prereg Step 8 downstream ordering, next steps in this order:
1. Gate 2 redo with a velocity-matching loss (not the `vel_scale` prior slot); three seeds; rerun P1–P3 with an expression-only regression baseline. Prereg first.
2. Cell-cycle reversal figure for the calibration paper.
3. Gate 0d: sign-preserving one-to-one consensus, fold-wise refits, shuffled-time null, tolerance pre-registered, precision-matrix baseline.
