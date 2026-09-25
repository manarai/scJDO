# REPORT — Gate 0a: bias-plumbing diagnostic

**Prereg**: `PREREG_Gate0a.md` (committed `3f7e040`, before compute).
**Compute**: 6 fits × ~4 min = ~29 min on CPU (marrow Ery, 1,151 cells, 5,000 epochs each).

## Verdict — Case 3

Per the prereg table:

> **Case 3**: v_hat DIFFERS, weights DIFFER. Bias reaches model and produces different weights, but seed dominates basin selection at the top-K display level. Not a bug. Manuscript "terminal-state guidance" claim needs softening to reflect what bias empirically accomplishes.

## Frozen numerics

Per-seed diff of A1 (`vel_scale=2.0, bias_strength=1.5`) vs B1 (`vel_scale=2.0, bias_strength=0.0`), same seed:

| seed | V_ref max \|Δ\| | V_ref bitwise identical | state_dict identical / total | max \|Δparam\| | top-15 overlap |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 1.099 | False | 3 / 53 | 1.099 | 13/15 |
| 1 | 1.099 | False | 3 / 53 | 1.099 | 14/15 |
| 2 | 1.099 | False | 3 / 53 | 1.099 | 14/15 |

Per-seed max_eig (peak) and top-5 (r24 reproduction verified — bit-identical to `r24_fig3_prior_audit_summary.json` A1/B1 entries):

| seed | A1 peak τ | A1 max_eig | B1 peak τ | B1 max_eig | A1 top-5 | B1 top-5 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 0.020 | −0.045 | 0.020 | −0.031 | RP11-1I2.1, CLU, IGHM, MEG3, CRHBP | RP11-1I2.1, CLU, MEG3, IGHM, CRHBP |
| 1 | 0.671 | −0.019 | 0.671 | −0.026 | HBB, AHSP, CA1, FAM178B, APOC1 | HBB, AHSP, CA1, FAM178B, APOC1 |
| 2 | 0.691 | −0.023 | 0.691 | −0.023 | HBB, AHSP, CA1, FAM178B, APOC1 | HBB, AHSP, CA1, FAM178B, APOC1 |

## Explanation for r24's "A1/B1 matched to three decimals"

The user's r24 observation was that `A1_default_fig3` and `B1_no_bias` gave identical WITHIN-CONFIG reproducibility numbers (both 0.437 eigvec-cos / 0.292 top-15 Jaccard). Gate 0a resolves the confusion:

**A1 and B1 are NOT bit-identical fits.** V_ref differs (max Δ 1.099), state_dict differs (50/53 params differ, max Δ 1.099), max_eig differs (seeds 0 and 1 give different max_eig; seed 2 happens to match to 3 decimals). Top-15 overlaps are 13-14/15 per seed, not 15/15.

**The identity 0.437/0.292 is real but it's a same-pattern-of-seed-instability identity, not a bit-identical-fits identity.** Both A1 and B1 have the same qualitative pattern (seed 0 in plasma basin, seeds 1-2 in erythroid basin). The within-config pairwise metric across the 3 seeds is dominated by this basin-pattern, which is set by seed regardless of bias.

**Bias empirically accomplishes ~1-2 top-15 gene swaps per seed on this branch, and never a basin change.** The seed determines which of the two basins the training lands in. Bias perturbs the loss landscape but the basin structure is dominated by initialization + Palantir pseudotime + FA embedding.

## Not a bug

- V_ref differs → the biased velocity IS being fed to `DriftField`.
- state_dict differs → training dynamics DO see different signals.
- Top-15 differs by 1-2 genes per seed → the bias has a measurable but small effect on the leading direction.

**The bias plumbing is working correctly.** The manuscript's phrasing that "terminal-state guidance supervises the branch fields" needs softening to reflect what bias empirically accomplishes on this branch (a 1-2 gene swap in the top-15, not a basin change), but the code is functioning as designed.

## Downstream action per prereg

- **Case 3 → proceed to Gate 0b immediately. No manuscript edits based on Gate 0a alone.**
- No code fix required.
- The r25 "bias is inert" language in the manuscript needs correction (it's empirically non-load-bearing for basin selection at the top-K display level, but it IS reaching the model). Save that correction for the post-Gate rewrite pass.

## Deliverables

- Prereg: `PREREG_Gate0a.md` (frozen before compute; commit `3f7e040`)
- Runner: `run_gate0a.py`
- Machine-readable summary: `gate0a_summary.json`
- This report

**Next**: Gate 0b (local-covariance archetype baseline).
