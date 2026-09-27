# REPORT — Gate 2 redo (velocity-matching loss + expression-only baseline)

**Verdict: FAIL Gate 2 redo.** P1 fails, P2 passes, P3 fails.

**Prereg**: `PREREG_Gate2_redo.md` (frozen `1e12082`, before compute).
**Compute**: ~16 min (dynamo preproc 12 s; G × 3 seeds ≈ 12 min; L × 3 seeds ≈ 4 min via vmapped Jacobian; baseline_E + metrics < 1 min).

## Numeric summary

| Prediction | Pre-declared threshold | Observed | Pass? |
|:---|:---|:---:|:---:|
| P1a G median cos(leading eigvec, R) | ≥ 0.50 | 0.340 | no |
| P1b Spearman(Re λ_max_G, Re λ_max_R) | ≥ 0.50 | 0.125 | no |
| P2 cos(vel_L, R) > cos(vel_G, R), CI positive | Δ_L−G > 0, CI excludes 0 positively | **+0.4392**, 95 % CI [+0.428, +0.451] | **yes** |
| P3 median var_L / var_G, CI < 1 | < 1 | 1.7437, 95 % CI [1.691, 1.805] | no |
| — informative: Δ_L − baseline_E | — | −0.7306, 95 % CI [−0.742, −0.720] | (L worse than linear regression) |
| — informative: cos(v_baseline_E, R) | — | **+0.9933** | (near-perfect linear map) |

## Per-arm details

- **R** (labelling-projected reference): identical to Gate 2 v1. `|v_ref|` median = 12.93.
- **G** (geometry-only, vel_scale = 0), 3 seeds: identical Jacobian tensors as v1. R Re(λ_max) range [0.72, 0.83].
- **L** (velocity-matching loss, λ = 1.0), 3 seeds:
  - Loss curve endpoints (per seed): dsm ∈ {2.82, 2.87, 2.83}; match ∈ {0.0032, 0.0033, 0.0032}. Match loss converged near zero — the training successfully matched drift direction to `V_ref` on the training batches.
  - Median cos(leading Jacobian eigvec, R's leading eigvec) = **0.974** (vs 0.030 in v1). Velocity-matching loss dramatically improves alignment of the operator's leading direction with R.
  - Spearman(Re λ_max_L, Re λ_max_R) = +0.316 (vs −0.578 in v1). No longer inverted; positive but weak.
  - Cross-seed variance of per-cell drift: 0.031 (vs 0.306 in v1). 10 × more stable, but still 1.74 × more variable than G's 0.016.
- **Baseline_E** (LinearRegression X_pca → R.v_ref via 5-fold KFold): mean cos(v_baseline, R) = **+0.9933**. A plain linear map from `X_pca` almost perfectly reconstructs R.

## What the numbers say

1. **The velocity-matching loss works, in the sense that it does what the loss says**: L's drift on training batches converges to a match term of 0.003 (cos ≈ 0.997 per batch). At the operator level, L's leading Jacobian eigvec agrees with R at cos = 0.97 (vs 0.03 in v1 additive-prior).

2. **But P2 passing is misleading in absolute terms**: L reaches cos(vel_L, R) = +0.26 across all 3,060 cells. A linear regression from `X_pca` alone reaches +0.99 on the same target. The scJDO operator plus a supervised loss recovers only about a quarter of the R signal that any linear map recovers for free.

3. **This reveals a weakness in R's construction**: R = `log1p(M_n_hvg) @ PCs_hvg` is a linear function of the new-transcript expression matrix. Because new transcripts are a subset of total transcripts, and the HVG + PCA basis is shared, R is nearly a linear function of `X_pca`. Any dataset where "the reference velocity is essentially another linear projection of the same expression matrix" makes the P2 test trivially winnable by a linear regressor but hard for a differential operator like a Jacobian, which is not free to be identity map.

4. **P3 still fails**: L's per-cell drift varies about 1.74 × more across seeds than G's, with a tight bootstrap CI [1.69, 1.81] that stays above 1. The velocity supervision adds seed-dependent slack to the DriftField that doesn't exist in the geometry-only fit. Much better than v1's 16.6× ratio, but still failing the pre-declared < 1 threshold.

## Interpretation

The velocity-matching loss is a real improvement over the additive V_ref prior slot from Gate 2 v1 — both in alignment (leading eigvec cos: 0.03 → 0.97) and in seed stability (var ratio: 16.6 → 1.74). The verdict FAIL is driven by:
- P1: G is unchanged from v1 (the loss doesn't touch G). G's data-determined output at cos = 0.34 was < 0.5 in v1 and remains so; this is a property of scJDO's geometry-only Jacobian on this substrate, not of the L arm.
- P3: L's added variance across seeds, though heavily reduced, still exceeds G's.

And the informative baseline_E readout says even PASSING all three predictions would not be a strong claim: on this reformulated R, a plain linear map from expression achieves cos = 0.99 with R. Scoring L at +0.26 puts scJDO's operator far below the linear-map ceiling.

## Compute-scale amendments — none

No amendments to this prereg. λ_match = 1.0 as declared; no post-hoc sweep. All frozen constants held.

## Decision (item 1 of the Gate 3 downstream chain)

Per prereg:
- **FAIL** → both v1 (additive V_ref) and this redo (matching loss) failed. The labelling-supervised route on scNT-seq is closed.

Moving to item 2 (cell-cycle reversal figure) and then item 3 (Gate 0d) per the chain declared in Gate 3 Step 8.

## Deliverables

- Prereg: `PREREG_Gate2_redo.md` (`1e12082`).
- Runner: `run_gate2_redo.py` (single deterministic entry point).
- This report.
- `gate2_redo_summary.json` — machine-readable numbers.
- `gate2_redo_arms.npz` — J_R, R.v_ref, X_pca, tau, t_centers, baseline_v, G/L per-seed J tensors.
- Log: `run_gate2_redo.log`.
