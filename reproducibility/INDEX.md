# reproducibility/INDEX.md

Every gate and round has its prereg, report, runner, and commit listed below. Paths are relative to the repo root. Commits refer to `origin/main` after the cherry-pick to include only r26 gate work + Task 0–7 wrap-up.

## r26 gate chain (calibration paper evidence)

### Gate 0 — prerequisites

| Item | Prereg | Report | Runner(s) | Report commit |
|:---|:---|:---|:---|:---:|
| **Gate 0a** — bias plumbing empirical trace | `reproducibility/gates_r26/gate0a/PREREG_Gate0a.md` | `reproducibility/gates_r26/gate0a/REPORT_Gate0a.md` | `reproducibility/gates_r26/gate0a/run_gate0a.py` | `130642f` |
| **Gate 0b** — local-covariance baseline | `reproducibility/gates_r26/gate0b/PREREG_Gate0b.md` | `reproducibility/gates_r26/gate0b/REPORT_Gate0b.md` | `reproducibility/gates_r26/gate0b/run_gate0b.py` | `2ca0f09` |
| **Gate 0c** — consensus archetypes (cNMF-style) — SUPERSEDED by Gate 0d | `reproducibility/gates_r26/gate0c/PREREG_Gate0c.md` | `reproducibility/gates_r26/gate0c/REPORT_Gate0c.md` | `reproducibility/gates_r26/gate0c/run_gate0c.py` | `e89f2dd` |
| **Gate 0d v1** — Hungarian 1-to-1 + shuffled null (superseded by v2 due to degenerate null) | `reproducibility/gates_r26/gate0d/PREREG_Gate0d.md` | `reproducibility/gates_r26/gate0d/REPORT_Gate0d.md` | `reproducibility/gates_r26/gate0d/run_gate0d.py` | `93d15f7` |
| **Gate 0d v2** — proper nulls (block + circular) + held-out gain K_eff + whitened precision | `PREREG_Task3_Gate0d_v2.md` | `reproducibility/gates_r26/gate0d_v2/REPORT_Gate0d_v2.md` | `reproducibility/gates_r26/gate0d_v2/run_gate0d_v2.py` | `41a509d` |

### Gate 1 — LARRY fate prediction

| Item | Prereg | Report | Runner | Report commit |
|:---|:---|:---|:---|:---:|
| **Gate 1 v1** — first pass with amendments (superseded by v2 row structure) | `reproducibility/gates_r26/gate1/PREREG_Gate1.md` | `reproducibility/gates_r26/gate1/REPORT_Gate1.md` | `reproducibility/gates_r26/gate1/run_gate1_build.py`, `run_gate1_eval.py`, `_scjdo_one_seed.py`, `_build_h5ad.py` | `d000078` |
| **Gate 1 v2** — canonical 5-row (E_PCA / E_FA / E_scVI / S_FA / E_FA + S_FA) | `reproducibility/gates_r26/gate1/PREREG_Gate1_v2.md` | `reproducibility/gates_r26/gate1/REPORT_Gate1_v2.md` | `reproducibility/gates_r26/gate1/run_gate1_v2_build.py`, `run_gate1_v2_eval.py`, `_scjdo_one_seed_v2.py` | `2d5f60e` |
| **Gate 1 Dynamo arm (SKIPPED)** — Klein release lacks spliced/unspliced + labelling | (skip report) | `reproducibility/gates_r26/gate1/REPORT_Gate1_dynamo_arm.md` | — | `b72bb1a` |

### Gate 2 — scNT-seq metabolic labelling

| Item | Prereg | Report | Runner | Report commit |
|:---|:---|:---|:---|:---:|
| **Gate 2 v1** — R (labelling-projected) vs G (geom-only) vs L (additive V_ref prior) | `reproducibility/gates_r26/gate2/PREREG_Gate2.md` | `reproducibility/gates_r26/gate2/REPORT_Gate2.md` | `reproducibility/gates_r26/gate2/run_gate2.py` | `504cb7b` |
| **Gate 2 redo** — velocity-matching loss L arm + expression-only Baseline_E | `reproducibility/gates_r26/gate2_redo/PREREG_Gate2_redo.md` | `reproducibility/gates_r26/gate2_redo/REPORT_Gate2_redo.md` | `reproducibility/gates_r26/gate2_redo/run_gate2_redo.py` | `376796e` |
| **Gate 2 temporal contrast** (Task 1 reinterpretation) | `PREREG_Task1_contrast.md` | `reproducibility/gates_r26/gate2_redo/REPORT_Gate2_contrast.md` | `reproducibility/gates_r26/gate2_redo/task1_contrast.py` | `2c1f63f` |

### Gate 3 — soft-mode falsification

| Item | Prereg | Report | Runner | Report commit |
|:---|:---|:---|:---|:---:|
| **Gate 3 strict** — LARRY day-2 covariance-only (C + Ic) vs E | `reproducibility/gates_r26/gate3/PREREG_Gate3_strict.md` | `reproducibility/gates_r26/gate3/REPORT_Gate3_strict.md` | `reproducibility/gates_r26/gate3/run_gate3_strict.py` | `6edbb1e` |

### Cell-cycle reversal (calibration figure)

| Item | Prereg | Report | Runner | Report commit |
|:---|:---|:---|:---|:---:|
| Cell-cycle v1 (vel_scale=0 only) — SUPERSEDED | `reproducibility/gates_r26/cellcycle_reversal/PREREG_cellcycle_reversal.md` | `reproducibility/gates_r26/cellcycle_reversal/REPORT_cellcycle_reversal.md` | `reproducibility/gates_r26/cellcycle_reversal/run_cellcycle_reversal.py` | `fc17265` |
| **Cell-cycle Task 2** — adds vel_scale=2 + A_12 metric + 4-panel figure | `PREREG_Task2_cellcycle.md` | `reproducibility/gates_r26/cellcycle_reversal/REPORT_cellcycle.md` | `reproducibility/gates_r26/cellcycle_reversal/task2_vel2.py` | `227df70` |

### Decision + status documents

| Item | Path | Commit |
|:---|:---|:---:|
| DECISION (all gates + downstream chain) | `reproducibility/gates_r26/DECISION.md` | `41a509d` (last update) |
| Gate 3 STATUS (Task 0) | `STATUS_Gate3.md` | `57a2e08` |
| EVIDENCE_LEDGER | `EVIDENCE_LEDGER.md` | `30da82b` |

## Legacy work under `reproducibility/` (pre-r26)

Left in place for historical reproducibility; NOT part of the calibration paper's evidence base. Prereg / report status is described inside each directory's own docs where available.

| Directory | Content |
|:---|:---|
| `reproducibility/operator_claims_benchmark/` | r22 Operator/archetype/Constructed-Obstruction benchmark (contains `run_operator_claims.py` used by gates 0a/0b/0c/0d) |
| `reproducibility/blind_saddle_benchmark/` | r13 / r14 blinded saddle-identification (V2-fixed benchmark restored to main for EL08 evidence — see restored-file provenance below) |
| `reproducibility/regulator_benchmark/` | r14-r21 regulator-benchmark suite (retrospective; see EVIDENCE_LEDGER for status) |
| `reproducibility/r10_saddle_audit/` | r10 saddle audit outputs |
| `reproducibility/code/` | Ad-hoc simulation and follow-up scripts (G-series, FOLLOWUP-series; FOLLOWUP6_aggregation is the r11 per-cell aggregation source for EL07) |
| `reproducibility/data/` | Cached intermediate results referenced by the above (FOLLOWUP6_aggregation.json ships the r11 per-cell / matrix argmax table for EL07) |
| `reproducibility/figures/` | Historical figure PDFs from earlier rounds |
| `reproducibility/tests/` | Legacy tests |

## Restored files (Task C)

Files brought back onto `main` from the pre-cherry-pick backup so that EL07 and EL08 have first-class evidence paths. Each row lists the original commit on the backup ref, the backup ref name, and the new commit on `main`.

| File on `main` | Original commit (on backup ref) | Backup ref | New commit on `main` |
|:---|:---:|:---:|:---:|
| `reproducibility/blind_saddle_benchmark/REPORT_V2FIXED.md` | `363f870` | `pre-cherrypick-r26-backup` | Task C |
| `reproducibility/blind_saddle_benchmark/outputs/V2FIXED_SUMMARY.json` | `363f870` | `pre-cherrypick-r26-backup` | Task C |
| `reproducibility/operator_claims_benchmark/REPORT_OPERATOR_CLAIMS.md` | `3775e4e` | `pre-cherrypick-r26-backup` | Task C |
| `reproducibility/operator_claims_benchmark/outputs/roundA_temporal_summary.json` | `3775e4e` | `pre-cherrypick-r26-backup` | Task C |
| `reproducibility/operator_claims_benchmark/outputs/roundA_temporal_curves.npz` | `3775e4e` | `pre-cherrypick-r26-backup` | Task C |

Note: the paths on the backup ref lived under `New_analysis/blind_saddle_benchmark/` and `New_analysis/operator_claims_benchmark/`; Task 6 (`ed1499c`) relocated the parent tree under `reproducibility/`. Only the report/output files listed above were carried back; the pre-r26 saddle-branch scripts remain outside the calibration-paper evidence base.

## How to reproduce

Every gate under `reproducibility/gates_r26/` has a self-contained runner. Run in order:
1. Restore Palantir marrow input: `reproducibility/operator_claims_benchmark/scripts/run_operator_claims.py` (auto-invoked by the gate runners via `prepare_marrow_ery`).
2. Gates 0a, 0b, 0c: `python reproducibility/gates_r26/gate0a/run_gate0a.py`, etc.
3. Gates 1, 2, 3: their own runners (require LARRY at `/tmp/larry_data/`, scNT-seq via `dynamo.sample_data.scNT_seq_neuron_labeling()` respectively).
4. Cell-cycle reversal: purely synthetic; runs standalone.
5. Gates 0d v1, 0d v2, Gate 2 redo, Gate 2 contrast (Task 1), cell-cycle Task 2, Task 3: run last (depend on outputs of the primary gates).

Compute footprint: full chain is ~8-10 h wall-clock on 36 GB / 14-core M2 laptop; most time in scJDO fits.
