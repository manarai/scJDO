# Supplement — scJDO v57 draft 1

## S1 — Constructed obstruction tables

**Table S1.** Parametrisation of the `ψ ∝ ρ` subfamily as the `d = 2` instance of the Lyapunov gauge.

Let `p(x) ∝ exp(−V(x))` and `D(x)` be the diffusion tensor. In the Lyapunov gauge:

- `f(x) = −(D(x) − 2 A(x)) · ∇V(x)`
- `J(x) = (−D(x)/2 + A(x)) · Σ(x)^{−1}`, with `Σ(x)` the local covariance in the linearised neighbourhood.

For `d = 2` the antisymmetric matrix `A(x)` has one degree of freedom (a scalar rotation rate `ω(x)`). The identifiability price schedule (Fig 1B) enumerated in analytic form:

| Output | Requires |
|:---|:---|
| `A(x)` at any single point | `d − 1 = 1` static perturbation of one input coordinate |
| Sign of `ω(x)` (rotation direction) | 1 time-resolved perturbation observed at two times |
| Per-cell velocity vector | metabolic labelling with a reference that is NOT in the linear span of the input representation |
| Per-cell lineage displacement | clonal barcode + late-time re-sampling |

**Table S2.** Instance of the price schedule for the neuron-labelling cohort. The reference velocity `v_ref = log1p(M_n_hvg) · PCs_hvg` is a linear function of `X_pca` up to the log1p transform on new counts; a `LinearRegression(X_pca → v_ref)` reaches cell-wise cosine 0.993 and held-out R² 0.920. This makes the labelling reference on this cohort a near-degenerate satisfier of price schedule item 3.

## S2 — Prior / seed audit tables

**Table S3.** Bias-strength plumbing empirical trace (marrow Ery, `vel_scale = 2`, `bias_strength ∈ {0, 1.5}`, three seeds; from `reproducibility/gates_r26/gate0a/gate0a_summary.json`).

| Seed | V_ref max abs Δ | state_dict tensors identical / total | top-15 gene overlap | Aggregate outcome |
|:---:|:---:|:---:|:---:|:---:|
| 0 | 1.099 | 3 / 53 | 13 / 15 | Case 3 |
| 1 | 1.099 | 3 / 53 | 14 / 15 | Case 3 |
| 2 | 1.099 | 3 / 53 | 13 / 15 | Case 3 |

Interpretation: the V_ref buffer and the neural-network weights differ between the two bias values at the same seed, i.e. the bias-strength pathway is empirically active. The training basin at these parameters is dominated by the seed, which means aggregate-level agreement of gene lists at the same seed under different bias values is high but not because bias is inert — it is because both fits fell into similar basins under this seed's initialisation. See `reproducibility/gates_r26/gate0a/REPORT_Gate0a.md`.

**Table S4.** vel_scale × bias factorial (marrow Ery, r22 audit; from `reproducibility/operator_claims_benchmark/`).

*Placeholder for the r22 factorial table — see the referenced source directory for full numbers. In v57's main text, the audit outcome is summarised by the "leading-direction loadings are seed-dependent under default settings" statement of §3.*

## S3 — Gate preregs and full results

Preregs and reports for every gate cited in the main text, with commit hashes. Every prereg was committed before its downstream compute.

| Gate / task | Prereg path | Report path | Commit |
|:---|:---|:---|:---:|
| Gate 0a — bias plumbing | `reproducibility/gates_r26/gate0a/PREREG_Gate0a.md` | `reproducibility/gates_r26/gate0a/REPORT_Gate0a.md` | `130642f` |
| Gate 0b — local-covariance baseline | `reproducibility/gates_r26/gate0b/PREREG_Gate0b.md` | `reproducibility/gates_r26/gate0b/REPORT_Gate0b.md` | `2ca0f09` |
| Gate 0c (superseded) — consensus archetypes | `reproducibility/gates_r26/gate0c/PREREG_Gate0c.md` | `reproducibility/gates_r26/gate0c/REPORT_Gate0c.md` | `e89f2dd` |
| Gate 0d v1 (superseded) — Hungarian 1-to-1 + degenerate null | `reproducibility/gates_r26/gate0d/PREREG_Gate0d.md` | `reproducibility/gates_r26/gate0d/REPORT_Gate0d.md` | `93d15f7` |
| **Gate 0d v2** — proper nulls + held-out gain + whitened precision | `PREREG_Task3_Gate0d_v2.md` | `reproducibility/gates_r26/gate0d_v2/REPORT_Gate0d_v2.md` | `41a509d` |
| Gate 1 v1 (superseded row structure) | `reproducibility/gates_r26/gate1/PREREG_Gate1.md` | `reproducibility/gates_r26/gate1/REPORT_Gate1.md` | `d000078` |
| **Gate 1 v2** — canonical 5-row | `reproducibility/gates_r26/gate1/PREREG_Gate1_v2.md` | `reproducibility/gates_r26/gate1/REPORT_Gate1_v2.md` | `2d5f60e` |
| Gate 1 Dynamo arm (SKIPPED) | (skip rule in Task 4) | `reproducibility/gates_r26/gate1/REPORT_Gate1_dynamo_arm.md` | `b72bb1a` |
| Gate 2 v1 (superseded by redo) | `reproducibility/gates_r26/gate2/PREREG_Gate2.md` | `reproducibility/gates_r26/gate2/REPORT_Gate2.md` | `504cb7b` |
| **Gate 2 redo** — velocity-matching loss | `reproducibility/gates_r26/gate2_redo/PREREG_Gate2_redo.md` | `reproducibility/gates_r26/gate2_redo/REPORT_Gate2_redo.md` | `376796e` |
| **Gate 2 temporal contrast** (Task 1) | `PREREG_Task1_contrast.md` | `reproducibility/gates_r26/gate2_redo/REPORT_Gate2_contrast.md` | `2c1f63f` |
| **Gate 3 strict** | `reproducibility/gates_r26/gate3/PREREG_Gate3_strict.md` | `reproducibility/gates_r26/gate3/REPORT_Gate3_strict.md` | `6edbb1e` |
| Cell-cycle v1 (superseded) | `reproducibility/gates_r26/cellcycle_reversal/PREREG_cellcycle_reversal.md` | `reproducibility/gates_r26/cellcycle_reversal/REPORT_cellcycle_reversal.md` | `fc17265` |
| **Cell-cycle Task 2** — 4-panel + A_12 metric | `PREREG_Task2_cellcycle.md` | `reproducibility/gates_r26/cellcycle_reversal/REPORT_cellcycle.md` | `227df70` |

Bold items are the canonical evidence sources referenced in the main text. Superseded items are retained for the audit trail.

## S4 — Reprogramming and K562 as illustrations only, with their limitations

The r15 K562 Perturb-seq analysis (Replogle 2022) and r22 reprogramming benchmark are retained under `reproducibility/regulator_benchmark/` and `reproducibility/blind_saddle_benchmark/` as historical illustrations. Neither is used to support a main-text claim in the v57 draft because:

- **r15 K562** — CRISPRi Perturb-seq is not a snapshot of an intact trajectory; the ground truth is a perturbation-response matrix, not fate probabilities. The comparison to scJDO features was inconclusive and does not meet the reporting checklist of §3.
- **r22 reprogramming** — the reprogramming dataset is a valuable stress test for the operator machinery but the r22 audit produced a mixed outcome (one PASS, one PARTIAL, one REFUTED across the three operator claims tested). No single verdict of the form used in Gates 1–3 emerges. See `reproducibility/operator_claims_benchmark/` for full details.

Both are illustrations, not evidence. Any reader who wishes to include them as motivation should treat them as such and not as substitutes for the ground-truth benchmarks in §2.4 and §2.5.

## S5 — Retracted-claims table

Every v56-era claim retracted by evidence gathered in r22–r26 + Tasks 0–7, with the evidence pointer and ledger ID.

| Retracted v56 claim | Evidence that retracts it | Superseding ledger ID | Terminology change required |
|:---|:---:|:---:|:---:|
| "Two attractors" describing vel_scale × bias outcomes | Gate 0a Case 3 — basin dominated by seed; empirical V_ref delta 1.099 shows plumbing works | EL04 | Replace with "training basins" |
| "Bias is inert" as a bug hypothesis | Gate 0a Case 3 (bug refuted); r22 factorial audit; EL04 | EL04 | — |
| "Round A = estimator fidelity" | r11 (synthetic learned-vs-oracle match) is estimator fidelity; Round A is real-data eigenvalue-curve reproducibility (a distinct calibration item) | (no single ID; split into §2.2 and §2.3 in v57) | — |
| Gate 0c archetype-consensus statement of "K_eff = 1" as an intrinsic property | Gate 0c used absolute cosine + single linkage — methodologically flawed. Gate 0d v2 held-out gain confirms K_eff = 1 via a different, valid mechanism | EL15 | — |
| "Sensitivity genes" language | Terminology change; instruments produce leading-direction loadings, not "sensitivity genes" | — | Replace with "leading-direction loadings" |
| "Velocity prior" language for the additive V_ref slot | Terminology change; the slot is an additive pseudotime-gradient prior with an external `V_ref` accepted through the same channel | — | Replace with "additive pseudotime-gradient prior" |
