# PREREG amendment 3 — Task A: Gate 1 v2 full-cohort rerun

**Status**: pre-registered before compute. Frozen against `705383b` (Task C commit).
Supersedes the compute-scale amendments (`b576308` amendment 1, `88b320e`
amendment 2) with the block's decision-rule spec.

## What changes from amendments 1–2

Amendments 1 and 2 shrank the scJDO fit substrate and reduced epochs to fit
laptop RAM budget. Both trades are reversed here. The committed launch that
ran under amendment 2 (see `taskA_build.log` @ `88b320e`) is void: the
seed-0 subprocess launched at `n_epochs=1500` on 9,638 cells and never
produced per-seed `.pkl` outputs. No results from that attempt are used.

## Frozen substrate + hyperparameters (this amendment)

- **scJDO fit substrate**: all 4,638 day-2 clone-barcoded cells retained
  (this includes 100% of the 759 cells in the 494 eligible clones) +
  random day-2 unbarcoded cells subsampled to fill the substrate to
  ~14,000 total (so ~9,362 unbarcoded, seed 0 for the fill sample).
  Barcoded eligible cells cannot be dropped by the subsample; the
  subsample is applied only to unbarcoded cells.
- **E baselines**: E_PCA / E_FA / E_scVI on all 28,249 day-2 cells (unchanged
  from amendment 1).
- **scJDO `fit_drift`** on the ~14K substrate: `vel_scale=0`,
  `bias_strength=1.5`, `hidden=256`, `depth=4`, `sigma=0.10`, windowing
  `kernel`, bandwidth `auto`, **`grid_size=100`**, `n_archetypes=5`,
  **`n_epochs=5000`**. 3 scJDO seeds ∈ {0, 1, 2}.
- **Palantir pseudotime** recomputed on the ~14K substrate's X_FA
  (unchanged).
- **Per-cell feature projection** to the full 28,249-cohort via
  nearest-bin lookup on a lightweight full-cohort Palantir pseudotime
  (unchanged).

## Classifier + evaluation (unchanged)

- `LogisticRegression(penalty="l2", C=1.0, class_weight="balanced",
  solver="liblinear", max_iter=1000)`, StandardScaler.
- Five arms: E_PCA, E_FA, E_scVI, S_FA, E_FA + S_FA.
- 5-fold `GroupKFold` grouped by clone × 5 CV seeds ∈ {42, 43, 44, 45, 46}.
- Paired clone-level bootstrap 2000 resamples for CIs on paired arm
  differences vs E_FA.
- **Clone rule identical to Gate 1 v2**: clone has ≥ 2 late-time Neu ∪ Mono
  cells with ≥ 80% purity (Neut label if ≥ 80% Neut of late Neu ∪ Mono;
  Mono label if ≤ 20%; else excluded).

## Decision rule for the launch (from the block, reproduced verbatim)

> If the run dropped barcoded eligible cells, or used 2,000 epochs without
> an amendment committed beforehand, kill and relaunch with: all barcoded
> eligible day-2 cells + unbarcoded subsample to ~14K, 5,000 epochs, grid
> 100, 3 seeds. Do not report the compromised run's numbers anywhere.

That relaunch is what this amendment prescribes.

## Compute budget (revised)

- Preprocess: ~15 min (scVI is the bulk).
- Palantir on ~14K substrate: ~2 min.
- scJDO fit at 5,000 epochs on ~14K cells: ~2.5–4.0 h per seed on the
  M2 (36 GB, 14-core) laptop; three seeds sequential → ~8–12 h.
- Classifier eval: <5 min.

Amendments 1–2 were driven by two OOM events at 14K. If OOM recurs at 14K
+ 5,000 epochs, the launch is killed and the failure recorded in
`TASKA_PROTOCOL_CHECK.md`; no further compute-scale amendment is
permissible under the block's decision rule.

## Ledger integration

- **EL10b** row added on Task A commit, matches the amendment-3 substrate
  spec ("full eligible cohort; field trained on ~14K-cell day-2
  subsample").
- **EL10** marked superseded → EL10b.

## Deliverables (unchanged from base prereg)

- `PREREG_TaskA_Gate1_v2_fullcohort.md` (base) + `_amendment3.md` (this file).
- `reproducibility/gates_r26/gate1/run_gate1_v2_fullcohort.py` — updated to
  ~14K substrate.
- `reproducibility/gates_r26/gate1/_scjdo_one_seed_taskA.py` — updated to
  `n_epochs=5000`.
- `reproducibility/gates_r26/gate1/run_gate1_v2_fullcohort_eval.py` —
  five-arm evaluation.
- `reproducibility/gates_r26/gate1/TASKA_PROTOCOL_CHECK.md` — post-launch
  protocol-check answers (five questions from the block).
- `reproducibility/gates_r26/gate1/REPORT_TaskA_Gate1_fullcohort.md`.
- `reproducibility/gates_r26/gate1/taskA_run.log` (launch log).
- `EVIDENCE_LEDGER.md` updated with EL10b + EL10 marked superseded (same
  commit as report).
