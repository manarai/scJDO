# LEDGER_MAP.md — ledger ID → v57 draft-2 manuscript location

Every ledger ID from `EVIDENCE_LEDGER.md` used in `manuscript_v57/scJDO_v57_draft2.md`, with its location(s) in the draft. Updated for draft 2: EL03 rewording; EL07/EL08 added (Task C); EL10b added (Task A); Box 1 rebuilt from ledger rows.

| Ledger ID | Manuscript location(s) |
|:---:|:---|
| EL01 | §2.1 (Section body) — Jacobian archetypes ≠ local covariance archetypes (mean matched cos 0.16); Box 1 (row 2 left column) |
| EL02 | §2.1 — Jacobian ≠ local precision (median matched cos 0.36); Box 1 (row 2 left column) |
| EL03 | §2.1 — whitened symmetric part shows partial agreement with Lyapunov prediction (median `|cos|` 0.33 ± 0.15, Frob corr 0.46 ± 0.01) and beats the local-precision baseline (Task B six-fit); Fig 3C; §3 "What snapshot-derived Jacobians can be trusted for"; Box 1 (row 1 left column and row 5 left column) |
| EL04 | §2.3 — prior audit at default settings (leading-eigvec cos 0.44 across seeds; top-15 Jaccard 0.29); §Methods — bias-strength pathway reaches model, seed dominates basin; Box 1 (row 1 and row 2 middle column); §S2 |
| EL05 | §2.5 — synthetic circular substrate: rotation reversal at `vel_scale = 2` (A_12 forward +0.149, reverse −0.204); Fig 5C; Box 1 (row 5 middle column); §3 experimental-design implication; §Methods |
| EL06 | §2.5 — velocity-matching loss produces time-varying seed-stable operator (median `|cos(v_L, v_R)|` = 0.974; L median contrast 0.532; L across-seed SD 0.011); Fig 5A/B; §3 "What velocity supervision changes and what it does not" |
| EL07 | §2.2 — per-cell aggregation of learned Jacobian tracks analytic at-cells `Re λ_max` curve (argmax within 0.01–0.09 of oracle, 3 seeds); Round A on marrow Ery (peak-τ 0.020 ± 0.000, pairwise curve Pearson 0.806, 3 seeds); Fig 2C; Box 1 (row 3 and row 5 right column); §3 "What snapshot-derived Jacobians can be trusted for" |
| EL08 | §2.2 — V2-fixed crossing scorecard: oracle-at-FP 6/6, oracle-at-cells 0/6, learned-at-cells 0/6 (modality limit; no cell-evaluated Jacobian localises the bifurcation); Fig 2A/B; Box 1 (row 4 middle column — bifurcation location prior-selected); §3 "What snapshot-derived Jacobians cannot be trusted for" |
| EL10 | §2.4 — LARRY day-2 fate prediction on the 154-cell v2 cohort: S_FA 0.51 vs E_FA 0.83; cohort table first row; superseded by EL10b as canonical numbers |
| EL10b | §2.4 — full eligible LARRY cohort (970 cells / 632 eligible clones; scJDO fit substrate 4,638 barcoded + 9,362 unbarcoded fill = 14,000); E_PCA 0.867 / E_FA 0.864 / E_scVI 0.879 / S_FA 0.554 / E_FA+S_FA 0.863; ΔAUROC(S_FA − E_scVI) = −0.325 [−0.340, −0.309]; cohort table row 2; §3 Limitations; Fig 4A |
| EL11 | §2.4 — Gate 3 strict: C and Ic add 0 beyond E; Fig 4 arm table; §3 "What snapshot-derived Jacobians cannot be trusted for" |
| EL12 | §2.5 — additive V_ref prior slot (Gate 2 v1) implied by `vel_scale = 0` reference; superseded by EL06 (matching-loss L arm) |
| EL13 | §2.5 — held-out prediction: J_L features R² 0.43 vs RidgeCV(X_pca) R² 0.92; §3 "What velocity supervision changes and what it does not"; §3 experimental-design implication (near-linear-reference caveat); Box 1 (row 3 right column) |
| EL14 | §2.5 — `vel_scale = 0` does NOT encode direction on synthetic circular substrate; Fig 5C; §Methods (measured behaviour) |
| EL15 | §2.3 — real and null tensors both ≈ rank one (K=1 gain 0.938 vs null 0.937); Fig 3A/B; Box 1 (row 3 middle column — archetype activation timing prior-selected — and row 4 left column — rank of temporal operator snapshot-determined) |
| EL16 | §3 Limitations — Klein LARRY lacks spliced/unspliced or labelling; Dynamo arm impossible on this cohort |
| EL20 | Supplement S5 (retracted-claims table); §Methods (bias-strength wording — "training basins", not "attractors") |
| EL21 | Supplement S5; §1 (identifiability framing does not treat `vel_scale × bias` as attractor structure) |
| EL22 | Supplement S5; §2.2 (r11 as estimator fidelity, per-cell aggregation of learned Jacobian; Round A as real-data reproducibility) |
| EL23 | Supplement S5; §2.3 (K_eff = 1 confirmation via Gate 0d v2 held-out gain; Gate 0c clustering artefact retracted) |
| EL24 | Terminology throughout — "leading-direction loadings" (not "sensitivity genes") |
| EL25 | Terminology throughout — "additive pseudotime-gradient prior" (not "velocity prior"); §Methods; §2.5 |
| EL30 | §2.4 — 154-cell cohort description (compute-driven subsample) |
| EL31 | §2.4 — Gate 3 strict substrate (520 clones, positive fraction 0.275) |
| EL32 | §2.5 — scNT-seq neuron labelling cohort (3,060 KCl-stimulated cortical neurons; Dynamo kinetic pipeline collapses on this dataset — 4 genes have finite velocity) |
| EL33 | §2.3 — marrow Ery calibration cohort (1,151 cells; Palantir 1.4.4 + MAGIC pipeline pin) |

**Ledger IDs NOT used in the draft**: none. All 24 IDs (EL01–08, EL10, EL10b, EL11–16, EL20–25, EL30–33) have a location.

**Ledger IDs used only in supplement / retracted table**: EL20, EL21, EL22, EL23.

**Note on citation format**: in `scJDO_v57_draft2.md`, ledger IDs appear as bracketed footnotes `[ EL## ]`. These are for internal traceability and are to be stripped before submission per the block's frozen rule ("strip nothing — keep [ EL## ] tags in draft 2 for review").
