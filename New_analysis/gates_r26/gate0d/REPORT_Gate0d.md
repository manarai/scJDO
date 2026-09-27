# REPORT — Gate 0d

**Verdict: FAIL Gate 0d.** Real pseudotime yields 4/5 consensus-stable archetypes, shuffled pseudotime yields 5/5, so `Δ = n_stable_REAL − n_stable_NULL = −1` (does not meet the pre-declared ≥ 2 threshold). Precision-matrix baseline condition passes cleanly (J ≠ precision).

**Prereg**: `PREREG_Gate0d.md` (frozen `ccf277e`, before compute).
**Compute**: ~25 min (12 scJDO fits × ~120 s + precision baseline < 1 min + Hungarian match < 1 s).

## Detailed numbers

### Consensus per condition (Hungarian 1-to-1 signed cosine to fold-0 seed-0 reference)

**REAL (palantir pseudotime)** — 5 non-reference replicates × 5 archetypes matched cosines:
```
[[0.677 0.844 0.970 0.730 0.934]     ← fold-0 seed-1
 [0.660 0.855 0.937 0.856 0.930]     ← fold-1 seed-0
 [0.669 0.851 0.935 0.712 0.937]     ← fold-1 seed-1
 [0.624 0.874 0.951 0.984 0.960]     ← fold-2 seed-0
 [0.598 0.869 0.949 0.977 0.958]]    ← fold-2 seed-1
per-archetype worst-case cos (min across 5 replicates):
    arch 0: 0.598      arch 1: 0.844      arch 2: 0.935
    arch 3: 0.712      arch 4: 0.930
n_stable_REAL (min ≥ 0.7): 4 / 5
```

**NULL (shuffled pseudotime, fixed permutation)**:
```
[[0.968 0.976 0.981 0.977 0.990]
 [0.979 0.987 0.981 0.976 0.982]
 [0.976 0.986 0.982 0.977 0.983]
 [0.981 0.988 0.981 0.974 0.988]
 [0.977 0.979 0.978 0.971 0.963]]
per-archetype worst-case cos:
    arch 0: 0.968      arch 1: 0.976      arch 2: 0.978
    arch 3: 0.971      arch 4: 0.963
n_stable_NULL: 5 / 5
```

**Δ = n_stable_REAL − n_stable_NULL = 4 − 5 = −1.**

### Precision-matrix baseline

Fold-0 seed-0 REAL Jacobian archetypes vs precision-matrix archetypes (Hungarian-matched signed cosines, same grid + bandwidth as scJDO):
- Matched cosines per archetype: [0.236, 0.314, 0.356, 0.434, 0.406]
- **Median: 0.356**
- Condition (J ≠ precision, median < 0.7): **PASSES**.

## Pre-registered pass conditions (final)

| Condition | Threshold | Observed | Pass? |
|:---|:---:|:---:|:---:|
| 1. n_stable_REAL ≥ 2 | ≥ 2 | 4 | yes |
| 2. Δ n_stable ≥ 2 (REAL − NULL) | ≥ 2 | −1 | **no** |
| 3. Median (J vs precision) < 0.7 | < 0.7 | 0.356 | yes |

Verdict = AND of 1, 2, 3 = **FAIL** (condition 2 fails).

## Interpretation — the shuffled-time null is degenerate

The pre-registered null (shuffle pseudotime, refit) turns out to be a degenerate control on this substrate:
- When `tau` is random per cell, the density at any grid bin `tc` is essentially the marginal density (bins no longer have distinct compositions of cells). scJDO's kernel-windowed local Jacobian at each bin becomes effectively the same operator across bins — no time-varying structure to learn.
- Semi-NMF on a nearly-constant (T, D, D) tensor produces near-identical archetypes across fits — hence 5/5 consensus under NULL, with min-cos ≥ 0.96.
- The NULL condition doesn't rule out spurious structure; it rules IN trivial constancy. High consensus under NULL indicates the fits agree on a trivial common answer, not that the pipeline overfits noise.

This inverts the intended semantics of the prereg's Δ threshold: NULL is more consistent than REAL because NULL has less to disagree on. A better null would preserve the marginal density-of-time structure while destroying the state-time joint (e.g. permute τ WITHIN pseudotime bins, or fit on IID Gaussian data with tau retained). Not part of this prereg.

## Interpretation — the positive readouts

- **REAL consensus is much healthier than Gate 0c suggested**: Gate 0c's single-linkage agglomerative clustering chained everything into 1 cluster (K_eff = 1). Gate 0d's Hungarian 1-to-1 matching reveals 4 of 5 archetypes are consensus-stable (min signed cos ≥ 0.7) across 3 folds × 2 seeds under REAL pseudotime. Only archetype 0 falls below tolerance (min cos = 0.60).
- **Jacobian ≠ precision matrix**: median matched cos = 0.36, well below 0.7. Gate 0b's "Jacobian ≠ covariance" now has a companion: "Jacobian ≠ precision" on the same substrate.

## What this changes in the aggregate

- Gate 0c's "1 stable mode" finding was an artefact of agglomerative-clustering with single linkage. With Hungarian matching (Gate 0d), scJDO's archetypes on marrow Ery are 4/5 consensus-stable across CV folds and seeds. This is a stronger structural claim than Gate 0c but does NOT overturn Gates 1, 2, 2-redo, 3, which tested a different question (does the fitted-field / covariance signal predict fate or velocity?).
- The precision-matrix baseline strengthens Gate 0b: the Jacobian is not just distinct from covariance but also from precision (its inverse).
- Gate 0d formally FAILS because the shuffled null on this substrate is a degenerate control (not because the fits are inconsistent under REAL).

## Deliverables

- Prereg: `PREREG_Gate0d.md` (`ccf277e`).
- Runner: `run_gate0d.py`.
- This report.
- `gate0d_summary.json` — machine-readable numbers.
- Per-fit caches: `fit_REAL_fold*_seed*.pkl`, `fit_NULL_fold*_seed*.pkl`.
- Log: `run.log`.

## Chain closure

Gate 3 Step 8 downstream chain (item 1 = Gate 2 redo, item 2 = cell-cycle reversal, item 3 = Gate 0d) is now complete. All items ran and reported. No further gates queued. Aggregate outcomes across all r26 gates + downstream chain:

| Gate | Verdict |
|:---:|:---:|
| 0a | Case 3 (bias reaches model; seed dominates basin) |
| 0b | PROCEED (Jacobian ≠ local covariance) |
| 0c | Default = V0 (vel_scale=0); K_eff = 1 stable mode by agglomerative |
| 0d | **FAIL** (shuffled null is degenerate; REAL 4/5 stable via Hungarian; J ≠ precision) |
| 1 (v1) | **FAIL** |
| 1 (v2) | **FAIL** (S_FA 0.51 vs E_FA 0.83; sharper than v1) |
| 2 | **FAIL** on all 3 predictions |
| 2 redo | **FAIL** (P2 passes; P1 and P3 fail; baseline_E dominates) |
| 3 strict | **FAIL** (C and Ic add 0 beyond E) |
| Cell-cycle reversal | vanilla scJDO does NOT encode direction (no reversal) |

Overall decision stands from `DECISION.md`: **Calibration paper with negatives**. Gate 0d adds two nuances worth writing in:
1. scJDO's archetypes ARE reasonably CV-stable under REAL (4/5 with min cos ≥ 0.7) when matched via Hungarian 1-to-1; Gate 0c's K_eff = 1 was a clustering artefact.
2. The Jacobian is not precision-matrix content either (median cos = 0.36 to precision archetypes).
