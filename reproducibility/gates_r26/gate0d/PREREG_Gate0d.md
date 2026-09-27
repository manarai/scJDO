# PREREG — Gate 0d: sign-preserving one-to-one consensus + fold refits + shuffled-time null + precision-matrix baseline

**Status**: pre-registered before compute. Frozen against `fc17265` (cell-cycle reversal report). No changes after commit.

## Purpose

Item 3 of the Gate 3 downstream chain. Refine Gate 0c's consensus procedure so that:
1. Archetype matching is 1:1 (Hungarian assignment) rather than agglomerative clustering.
2. Sign of the pattern is preserved (uses signed cosine, not `|cos|`; each pattern flipped once at ingestion to a canonical sign).
3. Consensus is evaluated across FOLDS of cells (not just seeds), to test spatial-cross-validation stability.
4. A **shuffled-time null** provides a baseline for what consensus rate arises when pseudotime is randomised.
5. A **precision-matrix baseline** (analogue to Gate 0b's covariance baseline) tests whether scJDO Jacobian archetypes match precision-matrix archetypes.

## Substrate (frozen)

Same as Gate 0c and Gate 0b: marrow Ery branch (r24 pinned pipeline, 1,151 cells, FA-30 rep). Palantir pseudotime already stored.

## Design (frozen)

### Fits

- **CV folds**: 3-fold random split of the 1,151 cells (`sklearn.model_selection.KFold(n_splits=3, shuffle=True, random_state=0)`). Each fold's TRAIN subset (~767 cells) is fit; the held-out cells do not enter that fit.
- **Seeds per fold**: 2 (`{0, 1}`).
- **scJDO configuration**: `vel_scale = 0` (Gate 0c default), `bias_strength = 1.5`, `hidden = 256`, `depth = 4`, `n_epochs = 3000`, `grid_size = 150`, `n_archetypes = K = 5`.
- **Conditions**:
  - **REAL**: fit with pseudotime = palantir pseudotime.
  - **NULL**: fit with pseudotime = a fixed random permutation of the palantir pseudotime (seed 0). Same permutation for all folds and seeds in the null condition.
- Total fits: 3 folds × 2 seeds × 2 conditions = **12 scJDO fits**.

### Consensus procedure (Hungarian, sign-preserving, tolerance-thresholded)

For each condition (REAL or NULL) separately:
- Collect all 6 replicates' 5 patterns (30, 30) → 30 patterns.
- **Sign normalisation**: for each pattern, flatten to 900-D, unit-L2 normalise, then flip so the leading-magnitude element is positive.
- **Reference set** = fold-0 seed-0's 5 patterns (this fold-seed's archetypes are the "atlas" against which the others are matched).
- For each other replicate (5 replicates in each condition), Hungarian-match its 5 patterns to the reference via `scipy.optimize.linear_sum_assignment` on cost = `1 - signed_cosine` matrix. Report the 5 matched signed cosines per replicate.
- For each of the 5 reference archetypes: compute `min_signed_cos_across_5_replicates` (worst-case match across all 5 non-reference replicates).
- **An archetype is "consensus-stable"** if `min_signed_cos ≥ tolerance = 0.7` (frozen).
- Report `n_consensus_stable / 5` per condition.

### Precision-matrix baseline (Gate 0b analogue)

- Build a kernel-aggregated (T, D, D) precision-matrix tensor `P_tensor` on the SAME grid + adaptive bandwidth as scJDO's J_tensor: for each grid point τ_i, compute local kernel-weighted covariance `Σ_i` (as in Gate 0b), then `P_i = pinv(Σ_i + λI)` with `λ = 1e-3 · trace(Σ_i) / D` (small ridge for stability). Same grid, same bandwidth as scJDO's fold-0 seed-0 real-condition fit.
- Semi-NMF decompose `P_tensor` into K=5 patterns using the same `jacobian_modes(K=5, n_restarts=5, seed=0)`.
- Hungarian-match Jacobian archetypes (reference fit fold-0 seed-0 REAL) to precision archetypes on signed-cos. Report the 5 matched cosines.

## Frozen tolerances + pass/fail criteria

- **Consensus tolerance**: `min_signed_cos ≥ 0.7` per reference archetype.
- **Real vs null**: `(n_stable_REAL − n_stable_NULL) ≥ 2`, i.e. at least 2 more archetypes pass consensus under REAL than under a shuffled-time null. If REAL and NULL yield the same n_stable, the fitted-field consensus carries no information beyond arbitrary temporal ordering.
- **Precision baseline**: if the reference Jacobian archetypes match precision archetypes at `median(matched cos) ≥ 0.7`, then the Jacobian's stable content is precision-matrix content; if `median < 0.7`, Jacobian carries structure beyond precision. (Analogue to Gate 0b's covariance conclusion.)

## Success criteria

- **PASS Gate 0d** requires ALL THREE:
  1. `n_stable_REAL ≥ 2` (at least 2 archetypes consensus-stable under REAL).
  2. `n_stable_REAL − n_stable_NULL ≥ 2` (real beats shuffled null).
  3. `median(Jacobian vs precision matched cos) < 0.7` (Jacobian ≠ precision).

- Anything else is **FAIL**.

## Reported quantities

- Per condition: 5 × 5 Hungarian cost matrices between each non-reference replicate and the reference.
- Per condition: 5 per-archetype `min_signed_cos` values.
- Per condition: `n_stable` and full 30-pattern signed-cos distribution.
- Precision baseline: 5 matched signed cosines + median.
- All reported regardless of outcome.

## Not part of this gate

- Multi-branch fits (single-branch Ery only).
- Alternative K values (K=5 fixed per Gate 0c).
- Alternative vel_scale (vel_scale=0 fixed per Gate 0c).
- Regeneration of pseudotime (Palantir output frozen from Gate 0b/c pipeline).

## Expected wall clock

- 12 scJDO fits × ~4 min = ~48 min.
- Precision-matrix build + semi-NMF: ~3 min.
- Consensus + matching + metrics: <1 min.
- **Total: ~55 min**.

## Downstream decision

- **PASS** → scJDO's fitted archetypes carry stable, spatial-CV-robust, null-outperforming, non-precision-matrix structure. Update calibration paper to include Gate 0d as positive evidence of one substrate where the fitted field admits ≥ 2 stable modes distinct from precision.
- **FAIL** → the fitted-field archetype route yields no more than 1 stable spatial-CV mode, or is indistinguishable from a shuffled-time null, or is precision-matrix content. Add to the calibration paper as a substantive negative alongside Gates 1, 2, 2-redo, 3.

No further gates queued after this. Chain (Gate 3 → item 1 → item 2 → item 3) closes with Gate 0d's outcome.
