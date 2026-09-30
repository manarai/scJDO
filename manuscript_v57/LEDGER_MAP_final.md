# LEDGER_MAP_final.md — Draft 5

`[ EL## ]` tags have been stripped from `manuscript_v57/scJDO_v57_draft5.md` per the Draft-5 rule. They remain in the supplement (`supplement_v57.md`) and in the figure legends (`FIGURE_LEGENDS_v57.md`). This file records the ledger-ID → main-text-location mapping so a reviewer can walk any main-text sentence back to its ledger row.

## Mapping

| Ledger ID | Main-text location(s) (Draft 5) |
|:---:|:---|
| EL01 | §2.1 "Established" paragraph — Jacobian archetypes ≠ local covariance archetypes (mean matched cosine 0.16, top-15 Jaccard ≤ 0.22, zero of five pairs at cos ≥ 0.9). |
| EL02 | §2.1 "Established" paragraph — Jacobian ≠ local precision (median matched cosine 0.36). |
| EL03 | §2.1 "Established" paragraph — whitened symmetric part vs Lyapunov prediction (`|cos|` 0.33 ± 0.15; Frob 0.46 ± 0.01; precision baseline 0.28 ± 0.07, −0.38 ± 0.00); §3 "What snapshot-derived Jacobians can be trusted for"; §2.3 "The whitened symmetric-part comparison of §2.1 identifies the substrate…"; Box 1 rows 1, 2, 5 left column. |
| EL04 | §2.3 "Prior audit at default settings" (leading-eigvec cos 0.44 across seeds; top-15 Jaccard 0.29); §Methods (measured behaviour of the additive pseudotime-gradient term); Box 1 rows 1, 2 middle column, row 5 middle column. |
| EL05 | §2.5 "Synthetic circular substrate — direction is prior-selected" (`vel_scale = 2` forward `+0.149`, reverse `−0.204`, 3 seeds); §Methods "measured behaviour" note; §3 experimental-design implication; Box 1 row 4 middle column. |
| EL06 | §2.5 velocity-matching-loss paragraph (median `|cos(v_L, v_R)|` = 0.974; L median contrast 0.532; across-seed spread halved vs G); §3 "With velocity supervision (§2.5), the learned Jacobian tensor becomes time-varying…". |
| EL07 | §2.2 first paragraph — synthetic estimator-fidelity comparison, argmax within 0.01–0.09 of oracle argmax across 3 seeds (Fig 2C); Box 1 row 3 left column ("Estimator fidelity to the analytic Jacobian on synthetic ground truth"). Synthetic-only. |
| EL08 | §2.2 second paragraph — V2-fixed crossing scorecard (oracle-at-FP 6/6; oracle-at-cells 0/6; learned-at-cells 0/6); §3 "What snapshot-derived Jacobians cannot be trusted for"; Box 1 out-of-table note (bifurcation location on cell-evaluated Jacobians). |
| EL09 | §2.3 "Real-data reproducibility of the eigenvalue curve" paragraph (peak-τ 0.020 ± 0.000; pairwise Pearson 0.806 across 3 seeds); Box 1 row 4 left column ("real-data reproducibility of the per-cell-aggregation `Re λ_max` curve on marrow Ery"). |
| EL10 | §2.4 "Fate prediction at day 2" — 154-cell / 106-clone compute-subsampled cohort, S_FA 0.51 (per-scJDO-seed 0.46 / 0.48 / 0.58); cohort table row 1. |
| EL10b | §2.4 "Fate prediction at day 2" — full eligible cohort (970 cells / 632 clones); primary contrast ΔAUROC(E_FA + S_FA − E_FA) = −0.001 [−0.002, +0.000]; secondary ΔAUROC(S_FA − E_scVI) = −0.325 [−0.340, −0.309] and ΔAUROC(S_FA − E_FA) = −0.310 [−0.324, −0.297]; cohort table row 2; §3 Limitations. |
| EL11 | §2.4 "Soft-mode falsification" paragraph — Arm 1 AUROC 0.646; Arm 2 ΔAUROC +0.0001 [−0.012, +0.012]; Arm 3 ΔAUROC +0.0004 [−0.002, +0.003]; §3 "What snapshot-derived Jacobians cannot be trusted for" (indirectly, via the "adds nothing to expression" wording). |
| EL13 | §2.5 "But: `R` on this cohort is nearly a linear function…" (`RidgeCV(X_pca)` R² 0.92 vs J_L features 0.43); §3 "What velocity supervision changes and what it does not"; §3 experimental-design implication (near-linear-reference caveat); Box 1 row 3 right column (per-cell velocity direction recoverable only if the reference is not collinear with the input rep). |
| EL14 | §2.5 "Synthetic circular substrate — direction is prior-selected" (`vel_scale = 0` magnitudes ≤ 0.05, no reversal); §Methods "measured behaviour" note. |
| EL15 | §2.3 "Rank of the temporal operator" paragraph (K=1 gain 0.938 vs null 0.937; K ≥ 2 real never exceeds block-null 95th); Box 1 row 3 middle column (archetype activation timing prior-selected) and row 4 left column (rank of the temporal operator snapshot-determined); Box 1 row 5 middle column (archetype pattern basis at default settings). |
| EL16 | §3 Limitations — Klein LARRY release lacks spliced/unspliced or metabolic-labeling layers so Dynamo cannot be added as a comparator to the fate-prediction cohort. |
| EL20 | Supplement §S5 (retracted-claims table); §Methods "training basins" wording (implicit, not tagged in main text). |
| EL21 | Supplement §S5; §1 identifiability framing (implicit, not tagged in main text). |
| EL22 | Supplement §S5; §2.2 / §2.3 split of "r11 estimator fidelity" vs "Round A real-data reproducibility" (see EL07 / EL09 rows above). |
| EL23 | Supplement §S5; §2.3 `K_eff = 1` (Gate 0d v2 held-out gain, not the retracted Gate 0c clustering result). |
| EL24 | Terminology — "leading-direction loadings" throughout; §2.3, Box 1. |
| EL25 | Terminology — "additive pseudotime-gradient prior" throughout; §Methods, §2.5. |
| EL30 | §2.4 — 154-cell cohort description (compute-driven subsample). |
| EL31 | §2.4 — soft-mode-falsification 520-clone cohort (positive fraction 0.275). |
| EL32 | §2.5 — scNT-seq neuron labeling cohort (3,060 KCl-stimulated cortical neurons; Dynamo kinetic pipeline collapses). |
| EL33 | §2.3 — marrow Ery calibration cohort (1,151 cells; Palantir 1.4.4 + MAGIC pipeline pin). |

## Where tags remain

- `manuscript_v57/supplement_v57.md` — S5 retracted-claims table and other body text carry EL IDs.
- `manuscript_v57/FIGURE_LEGENDS_v57.md` — every panel legend carries an EL ID.
- `manuscript_v57/LEDGER_MAP.md` — the earlier draft-3(b) form of this file, retained for the change log; superseded by this file for Draft 5.
- `EVIDENCE_LEDGER.md` — the canonical ledger itself.
