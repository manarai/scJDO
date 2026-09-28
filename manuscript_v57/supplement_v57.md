# Supplement — scJDO v57 draft 1

## S1 — Constructed obstruction tables

**Table S1a. Algebra of the `ψ ∝ ρ` subfamily (`d = 2`).**

Let `p(x) ∝ exp(−V(x))` be the stationary density of an Itô diffusion `dX_t = f(X_t) dt + σ(X_t) dW_t` with **noise covariance** `D(x) = σ(x) σ(x)^T`. In the Lyapunov gauge:

- `f(x) = −( D(x)/2 − A(x) ) · ∇V(x)`,
- `J(x) = ( −D(x)/2 + A(x) ) · Σ(x)^{−1}`,

with `A(x)` antisymmetric and `Σ(x) = ⟨(x−μ_x)(x−μ_x)^T⟩` the local covariance in the linearised neighbourhood. On the `ψ ∝ ρ` subfamily, the antisymmetric part carries a single scalar `Q(x)` = the (1,2) entry of `A(x)`. Under the family of transformations that vary `A(x)` while holding `p` and `D` fixed, three pointwise algebraic invariants control what the snapshot can and cannot resolve:

| Invariant | Statement | Consequence for the snapshot |
|:---|:---|:---|
| Trace preservation | `tr J(x)` is invariant under changes of `c(x)` at every `x` | The pointwise trace of the drift Jacobian is a snapshot-determined scalar. Any snapshot-derived estimator that agrees with `p` and `D` reproduces `tr J`. |
| Determinant identity | `det ∇f′(x) = (D² + c²) det H(x)`, where `H = −∇²V` is the Hessian of `−V` and `c(x)` is the scalar free parameter of `Q(x)` | The sign of `det ∇f′` is preserved regardless of `c`; adding rotational content does not change classification of `x` as saddle / node / spiral. |
| `Q` tracks `c` | `Q(x) = c(x)` (up to a fixed diffusion factor) | The antisymmetric scalar is exactly the free parameter of the family; recovering it requires evidence outside the snapshot. |

For `d = 2` the antisymmetric matrix `A(x)` has one degree of freedom (`c(x)`, equivalently a scalar rotation rate `ω(x)`). The identifiability price schedule (Fig 1B) enumerates the routes to recover `Q`:

| Output | Requires |
|:---|:---|
| `A(x)` at any single point | `d − 1 = 1` static perturbation of one input coordinate at that point |
| Sign of `ω(x)` (rotation direction) | 1 time-resolved perturbation observed at two times |
| Per-cell velocity vector | metabolic labelling with a reference that is NOT in the linear span of the input representation |
| Per-cell lineage displacement | clonal barcode + late-time re-sampling |

**Table S1b. Instance of the price schedule for the scNT-seq KCl-stimulated cortical-neuron cohort.** The reference velocity `v_ref = log1p(M_n_hvg) @ PCs_hvg` is by construction a linear function of `X_pca` up to the `log1p` transform on new counts. A `LinearRegression(X_pca → v_ref)` reaches cell-wise cosine 0.993 and held-out R² 0.920 (§2.5). This makes the labelling reference on this cohort a near-degenerate satisfier of price schedule item 3: it lies in the linear span of the input representation and therefore cannot fix `Q(x)` against a rotational alternative that is also linear in `X_pca`.

**Table S1c. Empirical footprint of the invariants on marrow Ery.** On six real Gate 0d v2 fits (3 folds × 2 seeds), the whitened symmetric part of scJDO's Jacobian has leading-eigenvector `|cos|` of 0.33 ± 0.15 vs the Lyapunov-gauge prediction `−½ Σ^{−1/2} D̂ Σ^{−1/2}` and 0.28 ± 0.07 vs the local-precision baseline; the Frobenius correlation is 0.46 ± 0.01 vs the prediction and −0.38 ± 0.00 vs precision (Task B, `reproducibility/gates_r26/gate0d_v2/task_B_precision_allfits.json`). The matrix-level agreement is better than the naive alternative but the leading direction is not reliably recovered [ EL03 ].

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

Every v56-era claim retracted by evidence gathered in r22–r26 + Tasks 0–7 + draft-2 rewrite, with the evidence pointer and ledger ID.

| Retracted v56 / draft-1 claim | Evidence that retracts it | Superseding ledger ID | Terminology / wording change |
|:---|:---:|:---:|:---:|
| "Two attractors" describing `vel_scale × bias` outcomes | Gate 0a Case 3 — basin dominated by seed; empirical V_ref delta 1.099 and 50/53 state-dict tensors differ; plumbing works | EL04 | Replace with "training basins" |
| "Bias is inert" as a bug hypothesis | Gate 0a Case 3 (bug refuted); r22 factorial audit; EL04 | EL04 | — |
| "Round A = estimator fidelity" | r11 (per-cell aggregation of learned Jacobian on lifted-truth synthetic) is estimator fidelity; Round A is real-data eigenvalue-curve reproducibility (peak-τ across seeds); a separate calibration item | EL07 (both merged) | Draft 2 §2.2 now cites EL07 for the estimator-fidelity claim explicitly |
| Gate 0c archetype-consensus statement of "K_eff = 1" as an intrinsic property | Gate 0c used absolute cosine + single linkage — methodologically flawed. Gate 0d v2 held-out gain confirms K_eff = 1 via a different, valid mechanism (real and null tensors both ≈ rank one) | EL15 | Add: "real and null tensors both approximately rank one" |
| "Sensitivity genes" language | Terminology change; instruments produce leading-direction loadings, not "sensitivity genes" | — | Replace with "leading-direction loadings" |
| "Velocity prior" language for the additive V_ref slot | Terminology change; the slot is an additive pseudotime-gradient prior with an external `V_ref` accepted through the same channel | — | Replace with "additive pseudotime-gradient prior" |
| Draft-1 §2.2 claim "scJDO's per-cell Jacobian estimate matches the oracle to within fit noise once cells at the symmetric fixed point are provided with an oracle-rotation reference" | V2-fixed benchmark: oracle-at-cells 0/6 and learned-at-cells 0/6; only oracle-at-FP crosses (6/6). No cell-evaluated Jacobian localises the bifurcation on this substrate | EL08 | Replace with "per-cell aggregation reproduces the analytic at-cells `Re λ_max` profile; the crossing is a property of the fixed point, not of the estimator" |
| Draft-1 abstract "the antisymmetric part … has not been reported honestly" | Editorial. The identifiability limit is a well-established property of snapshot inference; the abstract now cites Weinreb 2018 for the limit and drops the editorial line | (no ledger row; wording change) | Removed |
| Draft-1 abstract "a scored estimator recovers the analytic Jacobian at the cell level once an oracle rotation is supplied" | V2-fixed benchmark contradicts the "recovers the analytic Jacobian at the cell level" reading; only per-cell aggregation of `Re λ_max` reproduces the analytic curve (EL07), while cell-evaluated crossings do not exist (EL08) | EL07, EL08 | Replaced by two-sentence EL07/EL08 statement in abstract |
| Draft-1 §2.5 unqualified "labelling ground truth" framing | The scNT-seq KCl-neuron cohort's `R` is nascent-transcript projected onto `X_pca`; a linear regression on `X_pca` already reaches R² 0.92 to `R`. This bounds what the labelling comparison here can test | EL13 | Section title now labels the substrate; §2.5 states the limit; Discussion narrows the labelling paragraph to this dataset and reference type |
| Draft-1 Box 1 "kernel-selected bandwidth" in the snapshot-determined column | Bandwidth is a fitting hyperparameter chosen by the reproducibility × contrast × localisation criterion; it does not fall out of the snapshot | — | Removed from Box 1 |
| Draft-1 Box 1 "sign flip of `A_12`" in the snapshot-determined column | The sign flip requires the additive pseudotime-gradient prior to be active (`vel_scale = 2`); at `vel_scale = 0` there is no sign flip. The output is prior-selected | EL05, EL14 | Moved to the prior-selected column |
