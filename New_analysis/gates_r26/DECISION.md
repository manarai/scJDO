# DECISION — Gates protocol r26

**Status**: all four gates reported. Decision from the frozen decision table.

## Summary of gate outcomes

| Gate | Substrate | Verdict | Commit |
|:---|:---|:---:|:---:|
| 0a — bias-plumbing empirical | marrow Ery, A1 vs B1 | Case 3 (bias reaches model; seed dominates basin) | `2a41fa1` |
| 0b — local-covariance baseline | marrow Ery Jacobian vs cov tensor | PROCEED (Jacobian ≠ local covariance; mean matched cos = 0.16) | `9d4e5ab` |
| 0c — consensus archetypes | marrow Ery, V0 vs V2 arms | Default = V0 (vel_scale=0); K_eff = 1 stable mode, 4 archetypes are seed-noise | `598a160` |
| 1 — LARRY early fate prediction (v1) | LARRY day-2 neut vs mono | **FAIL** (S mean 0.62 vs E mean 0.83; S − E = −0.22, CI [−0.27, −0.16]) | `44e05f7` |
| 1 — LARRY early fate prediction (v2, canonical rows) | same substrate | **FAIL, sharper** — E_PCA 0.8254 / **E_FA 0.8299 (best)** / E_scVI 0.8281 / **S_FA 0.5076** / E_FA + S_FA 0.8293; S_FA − E_FA = −0.32, CI [−0.36, −0.28] | `3be140b` |
| 2 — scNT-seq metabolic labelling | scNT-seq Neu, R vs G vs L | **FAIL on all 3 predictions** (P1: cos 0.34 < 0.5; P2: Δ_L−G = −0.057 CI negative; P3: var_L/var_G = 16.6) | `315b8d5` |

## Decision-table read-out

Per the frozen decision table (`{Stop / Pass-Pass / Pass-Fail-Pass / Pass-Fail-Fail} → {Calibration paper / Tool paper / Assay-design paper / Calibration paper with negatives}`):

Combined outcome: {Gate 0 PASS, Gate 1 FAIL, Gate 2 FAIL} = **Pass-Fail-Fail**.

→ **Calibration paper with negatives.**

## What the manuscript can claim

Positive (all gate 0 outcomes):
- The scJDO Jacobian tensor is a real object with a well-defined bandwidth-selected estimator, computable at scale, distinct from local covariance in operator space (Gate 0b: 0/5 matched pairs at cos ≥ 0.9; mean matched cos = 0.16).
- The bias-plumbing pathway is functional (Gate 0a: V_ref max |Δ| = 1.099 between A1 and B1, state_dict differs on 50/53 params) — an earlier interpretive claim of "bias is inert" was refuted empirically.
- The archetype-decomposition step yields ONE consensus-stable mode across seeds; the other K − 1 archetypes reported per fit are seed-specific decomposition slack (Gate 0c).
- The `vel_scale = 0` (no velocity prior) configuration is at least as consensus-stable as `vel_scale = 2` on the reference substrate (Gate 0c tiebreak: intra-cluster |cos| 0.856 vs 0.844).

Negative (all downstream tests):
- **Fate prediction (LARRY)**: scJDO's per-cell Jacobian features (Re λ_max + leading-J projection + consensus archetype activation) do NOT beat expression baselines for neut-vs-mono clone-fate prediction at day 2. v2 canonical rows: E_PCA 0.83, E_FA 0.83, E_scVI 0.83 (all three linear/scVI reps agree within 0.005 AUROC). **S_FA 0.51 — essentially chance; per-seed 0.46/0.48/0.58 shows scJDO fit-instability, not signal.** E_FA + S_FA equals E_FA (Δ = −0.0006, CI includes 0) — S_FA is dominated by regularisation when combined with E_FA.
- **Labelling-anchored operator agreement (scNT-seq)**: scJDO's geometry-only Jacobian does not agree with a labelling-derived reference on data-determined outputs (median leading-eigvec cos = 0.34, Spearman on Re λ_max = 0.13). Injecting the labelling reference as V_ref makes disagreement worse (cos → 0.03) and destabilises fits across seeds (var_L / var_G = 16.6).

## What the manuscript cannot claim

- That scJDO features carry information useful for early cell-fate prediction on any tested substrate (LARRY refuted this).
- That scJDO's operator recovers labelling-based velocity structure (scNT-seq refuted this).
- That the archetype panel of K = 5 modes reflects K = 5 stable biological objects (Gate 0c refuted this).
- Any figure comparing G / L / R that positions scJDO as advancing beyond standard operators.

## Deferred manuscript corrections (from the user's own r24-reading corrections)

Executable now that all gates are reported. From the user's Aug/Sep session:

1. **"Two attractors" → "optimization basins of training"**. The observed vel_scale × seed × configuration behaviour that motivated the original phrasing is a property of the SDE training landscape's local minima, not a dynamical-systems attractor. Fix wherever the manuscript reads "two attractors" and rewrite the surrounding paragraph. Gate 0a Case 3 supports this framing (basin dominated by seed under fixed data + prior).

2. **Round A characterisation**. Not "estimator fidelity" — that phrase belongs to the r11 synthetic learned-vs-oracle match. Round A is real-data eigenvalue-curve reproducibility. Both belong in the calibration tier but as separate items.

3. **Bias plumbing**. r24 read "bias is inert" as a bug hypothesis. Gate 0a empirically refuted the bug hypothesis (Case 3 confirmed). Update any manuscript language that presumed a bug; the observation is that the bias signal reaches the model, but the training basin under the current setup is dominated by seed initialization at these fit parameters.

## Deliverables per gate

Every gate directory contains: `PREREG_Gate<N>.md`, `run_gate<N>*.py`, `gate<N>_summary.json`, and `REPORT_Gate<N>.md`. All prereg + report commits happened before their downstream commits.

- `New_analysis/gates_r26/gate0a/` — 3f7e040 (prereg), 2a41fa1 (report)
- `New_analysis/gates_r26/gate0b/` — 41f5668 (prereg), 9d4e5ab (report)
- `New_analysis/gates_r26/gate0c/` — dbd67e4 (prereg), 598a160 (report)
- `New_analysis/gates_r26/gate1/` — 7829ecc (prereg) + 39c35d3, a403743, bd6fb0d (amendments before compute) + 44e05f7 (report)
- `New_analysis/gates_r26/gate2/` — 9aef316 (prereg) + 6f1d1cb (amendment before compute) + 315b8d5 (report)

## Next action (per protocol)

- No new tags until manuscript is rewritten around the decision.
- Manuscript rewrite proceeds against the calibration-with-negatives frame, incorporating the three deferred corrections above.
- No further gate work until the rewrite is drafted.
