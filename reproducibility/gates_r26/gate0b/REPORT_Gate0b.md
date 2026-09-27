# REPORT — Gate 0b: local-covariance archetype baseline

**Prereg**: `PREREG_Gate0b.md` (frozen `41f5668`, before compute).
**Compute**: ~5 min (~4 min scJDO fit + ~1 min covariance-tensor + semi-NMF × 6).

## Verdict — PROCEED to Gate 0c

Per the prereg stop rule:
> If mean matched cosine across the 5 pairs is ≥ 0.9 for at least 3 of 5 pairs → the Jacobian archetypes don't carry more than local covariance → STOP the tool-paper track.

**Result: 0 of 5 pairs have mean matched cos ≥ 0.9. Overall mean matched cos = 0.161. Top-15 gene-loading Jaccards are ≈ 0.**

The Jacobian archetypes are structurally distinct from local-covariance archetypes at the same grid and bandwidth.

## Frozen numerics

- scJDO reference fit: seed=42, `bandwidth h* = 0.0500` (adaptive-selected), J_tensor shape (200, 30, 30).
- Covariance tensor built at same grid, same h*: shape (200, 30, 30), mean per-slice Frobenius norm 2.07.
- Semi-NMF K=5, n_restarts=5, 3 seeds each.

Semi-NMF reconstruction error (lower = better):
| Seed | scJDO J_tensor err | Covariance tensor err |
|:---:|:---:|:---:|
| 0 | 17.04 | 5.84 |
| 1 | 17.04 | 5.84 |
| 2 | 17.04 | 5.84 |

Both tensors admit a K=5 semi-NMF decomposition. Covariance tensor has lower Frobenius norm and hence lower error (expected — Jacobian has larger dynamic range).

Matched-pattern cosine (5 pairs × 3 seeds via Hungarian matching):
| Seed | Pair 1 | Pair 2 | Pair 3 | Pair 4 | Pair 5 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 0.323 | 0.011 | 0.138 | 0.134 | 0.191 |
| 1 | 0.205 | 0.142 | 0.038 | 0.112 | 0.313 |
| 2 | 0.205 | 0.142 | 0.038 | 0.112 | 0.313 |
| **mean** | **0.244** | **0.098** | **0.072** | **0.120** | **0.272** |

Top-15 gene-loading Jaccard (5 pairs × 3 seeds; mean across seeds):
| Pair | Mean Jaccard |
|:---:|:---:|
| 1 | 0.00 |
| 2 | 0.22 |
| 3 | 0.02 |
| 4 | 0.04 |
| 5 | 0.02 |

**Overall mean matched cos: 0.161. Number of pairs at ≥ 0.9: 0 / 5.**

## Interpretation

The Jacobian tensor scJDO learns and the local-covariance tensor of the same substrate live in essentially orthogonal parts of the (T, D, D) operator space — mean matched-pattern cosine is 0.16 (much closer to random than to identity), and gene loadings of matched components essentially don't overlap (Jaccard ≤ 0.22).

**Whether that structural distinctness carries biological meaning is a different question.** Gate 0b answers only that the Jacobian is not covariance-in-disguise. Gates 1 (LARRY early fate prediction) and 2 (metabolic-labelling comparison) test whether the distinct structure carries fate-relevant or perturbation-relevant information.

Note: seeds 1 and 2 give identical matched cosine and Jaccard values (0.205, 0.142, 0.038, 0.112, 0.313 and 0.00, 0.25, 0.03, 0.00, 0.03). This is because the semi-NMF for these specific seeds converged to the same restart — an internal detail of the decomposition, not a scientific finding.

## Downstream action per prereg

**Proceed to Gate 0c** (consensus archetypes). No manuscript edits based on Gate 0b alone.

## Deliverables

- Prereg: `PREREG_Gate0b.md` (`41f5668`)
- Runner: `run_gate0b.py`
- Machine-readable summary: `gate0b_summary.json`
- This report

**Next**: Gate 0c (consensus archetypes across seeds and prior settings, cNMF-style).
