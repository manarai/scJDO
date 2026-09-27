# LEDGER_MAP.md — ledger ID → v57 draft-1 manuscript location

Every ledger ID from `EVIDENCE_LEDGER.md` used in `manuscript_v57/scJDO_v57_draft1.md`, with its location(s) in the draft.

| Ledger ID | Manuscript location(s) |
|:---:|:---|
| EL01 | §2.1 (Section body) — Jacobian archetypes ≠ local covariance archetypes (mean matched cos 0.16) |
| EL02 | §2.1 — Jacobian ≠ local precision (median matched cos 0.36) |
| EL03 | §2.1 — whitened symmetric part matches Lyapunov prediction `−½ Σ^{−1/2} D̂ Σ^{−1/2}` (median |cos| 0.454, Frob corr +0.468) and matches local precision poorly (|cos| 0.224, Frob corr −0.368); Fig 1C; Fig 3C; §3 "What snapshot-derived Jacobians can be trusted for"; Box 1 (row 1) |
| EL04 | §2.3 — seed / prior audit table reference; §3 "Reporting checklist"; Box 1 (row 2 left column); §1 (mentioned as basin phenomenon); referenced in retracted-claims table (EL20, EL21) |
| EL05 | §2.5 — rotation reversal at vel_scale = 2 (A_12 forward +0.149, reverse −0.204); Fig 5C; Box 1 (row 5); §3 experimental-design implication |
| EL06 | §2.5 — velocity-matching loss produces time-varying seed-stable operator (median |cos(v_L, v_R)| = 0.974; L median contrast 0.532; L across-seed SD 0.011); Fig 5A/B; Box 1 row 3 (estimator fidelity); §3 "What velocity supervision changes" |
| EL10 | §2.4 — LARRY day-2 fate prediction: S_FA 0.51 vs E_FA 0.83; Fig 4A; §3 discussion of what velocity supervision does not overcome |
| EL11 | §2.4 — Gate 3 strict: C and Ic add 0 beyond E; Fig 4B; Box 1 row 2 (activation timing analog); §3 discussion of soft-mode falsification |
| EL12 | §2.5 — additive V_ref prior slot (Gate 2 v1) result implied by "with `vel_scale = 0` and no external velocity" reference; superseded by EL06 (matching-loss L arm) |
| EL13 | §2.5 — held-out prediction: J_L features R² 0.43 vs RidgeCV(X_pca) R² 0.92; Fig 5D; §3 "What velocity supervision changes and what it does not"; §3 experimental-design implication (near-linear-reference caveat) |
| EL14 | §2.5 — vel_scale = 0 does NOT encode direction; Fig 5C; Box 1 row 5 |
| EL15 | §2.3 — K_eff = 1 by held-out gain against block-permutation null; Fig 3A/B; Box 1 row 2 |
| EL16 | §3 Limitations — Klein LARRY lacks spliced/unspliced or labelling; Dynamo arm impossible on this cohort |
| EL20 | Supplement S5 (retracted-claims table); §Box 1 (row 2 wording "training basins", not "attractors") |
| EL21 | Supplement S5; §1 (identifiability framing does not treat vel_scale × bias as attractor structure) |
| EL22 | Supplement S5; §2.2 (Round A framed as real-data eigenvalue-curve reproducibility, r11 as estimator fidelity) |
| EL23 | Supplement S5; §2.3 (K_eff correction: Gate 0c's K_eff = 1 was clustering artefact; K_eff = 1 STANDS via Gate 0d v2 held-out gain — different reasoning, same number) |
| EL24 | Terminology throughout — "leading-direction loadings" (not "sensitivity genes"); §1, §3 reporting checklist |
| EL25 | Terminology throughout — "additive pseudotime-gradient prior" (not "velocity prior"); §Methods; §2.5 |
| EL30 | §2.4 substrate description ("N = 154 cells across 106 clones after compute-driven subsampling") |
| EL31 | §2.4 Gate 3 strict substrate description (520 clones, positive fraction 0.275) |
| EL32 | §2.5 substrate description (scNT-seq neuron labelling, 3,060 Neu; Dynamo kinetic pipeline collapses on this dataset) |
| EL33 | §2.3 substrate description (marrow Ery 1,151 cells; Palantir 1.4.4 + MAGIC pipeline pin) |

**Ledger IDs NOT used in the draft**: none. All 22 IDs have a location.

**Ledger IDs used only in supplement / retracted table**: EL16, EL20, EL21, EL22, EL23.

**Note on citation format**: in `scJDO_v57_draft1.md`, ledger IDs appear as bracketed footnotes `[EL##]`. These are for internal traceability and are to be stripped before submission per Task 7's frozen rule.
