# TASKA_PROTOCOL_CHECK — Gate 1 full-cohort rerun

Answers to the five verification questions from the block, plus the
launch-settings record required by the block ("record launch settings
and start time in TASKA_PROTOCOL_CHECK.md").

## Retrospective verification (prior attempt under amendment 2)

**1. Which launch produced the results file: bpswns67k, bo11rof84, or a later one? Give the exact settings it ran with: n cells fit on, epochs, grid_size, seeds.**

No launch produced a results file. `bpswns67k` and `bo11rof84` were
background-shell handles from a prior session and have no logs under those
names on disk. The only Task A artefacts found are build-phase outputs
(mtime 2026-09-27 23:34):

- `reproducibility/gates_r26/gate1/data/larry_day2_preproc_taskA.h5ad`
  (28,249 × 2000)
- `reproducibility/gates_r26/gate1/data/larry_day2_preproc_taskA_fit.h5ad`
  (9,638 × 2000)
- `reproducibility/gates_r26/gate1/taskA_build.log`

The build-log tail is `[seed 0 TaskA] fit_drift on (9638, 2000)
(grid=100, epochs=2000)`. No per-seed `gate1_v2_taskA_scjdo_seed{0,1,2}.pkl`
was ever written. No summary JSON. No arm eval.

The attempted settings under amendment 2 were: 9,638 fit cells, grid 100,
`n_epochs=1500` (source: `_scjdo_one_seed_taskA.py` at `88b320e`; the
runner's `print` said 2000 but the actual `n_epochs` argument was 1500).
Three seeds queued, only seed 0 launched, none completed.

**Verdict**: no completed Task A run under any earlier launch. Under the
block's decision rule the attempt is void ("no completed run" and the
substrate was smaller than the ~14K target). Its numbers are not
reported anywhere.

**2. Was PREREG_TaskA_Gate1_v2_fullcohort.md committed before that launch's results were inspected? Quote the commit hashes in order.**

Yes. The prereg + amendments were all committed before any launch:

- `7f7888e` — base prereg (`Task A: prereg — Gate 1 v2 full-cohort rerun`)
- `b576308` — amendment 1 (~14K substrate)
- `88b320e` — amendment 2 (~9,638 substrate, epochs=1500)

The launch under amendment 2 produced no `.pkl` results to inspect. No
results were inspected before any prereg edit.

**3. Were all barcoded day-2 cells in eligible clones retained in the training set (subsample applied only to unbarcoded cells)? Give the count.**

Yes, at the substrate-construction step. Amendments 1–3 all retain the
full set of 4,638 barcoded day-2 cells (which contains 100 % of the 759
barcoded cells in eligible clones), and subsample only unbarcoded cells.
The build log records `has_clone=4638`. Under amendment 3 (this launch)
the substrate target is 14,000: 4,638 barcoded + 9,362 unbarcoded.

**4. Epochs: 5,000 or 2,000? If 2,000, attach the training-loss curve and state whether it plateaued.**

Amendment 2's launched code used `n_epochs=1500`, but that run produced
no per-seed cache, no loss curve, no `.pkl`. Amendment 3 (this launch)
uses `n_epochs=5000` per the block's spec.

**Decision-rule verdict on prior attempt**: the amendment 2 attempt is
void (no completed run; substrate smaller than ~14K; epochs below 5,000
without a prior amendment matching those numbers). No numbers from it
are reported.

**5. If the run dropped barcoded eligible cells, or used 2,000 epochs without an amendment committed beforehand, kill and relaunch...**

The relaunch under amendment 3 (`e2731f6`) is what this file records.

## Launch settings (amendment 3, this launch)

- Committed prereg: `7f7888e` (base) + `b576308` (am1) + `88b320e` (am2)
  + `e2731f6` (am3).
- Substrate: all barcoded day-2 cells (~4,638) + unbarcoded day-2 cells
  subsampled to fill to 14,000 total, seed 0 for the fill.
- Barcoded eligible cells retained: 100 % (759 / 759 within the 494
  eligible clones).
- `grid_size=100`; `n_epochs=5000`; 3 scJDO seeds ∈ {0, 1, 2};
  `vel_scale=0`; `bias_strength=1.5`; `hidden=256`; `depth=4`;
  `sigma=0.10`; windowing `kernel`; bandwidth `auto`;
  `n_archetypes=5`.
- Classifier: `LogisticRegression(penalty="l2", C=1.0,
  class_weight="balanced", solver="liblinear", max_iter=1000)` with
  StandardScaler; 5-fold GroupKFold × 5 CV seeds ∈ {42, 43, 44, 45, 46}.
- Arms: E_PCA, E_FA, E_scVI, S_FA, E_FA + S_FA.
- Clone rule: ≥ 2 late Neu ∪ Mono cells at ≥ 80 % purity (Neut ≥ 80 %,
  Mono ≤ 20 %).
- Log: `reproducibility/gates_r26/gate1/taskA_run.log`.
- Start time (launcher): 2026-09-28T00:03 (America/Denver).

## Launch section — completion status

- **Start**: 2026-09-28T00:03. **Preproc + Palantir on ~14K substrate**: 2026-09-28T00:07 (~4 min; scJDO fit substrate written at 14,000 × 2000).
- **seed 0 fit_drift**: 1802.9 s (~30 min) → `gate1_v2_taskA_scjdo_seed0.fitonly.pkl` at 00:37.
- **seed 0 project_to_full_cohort**: +34 s → `gate1_v2_taskA_scjdo_seed0.pkl` at 00:38.
- **seed 1 fit + project**: 1581.9 s (~26 min) → 01:04.
- **seed 2 fit + project**: 1589.2 s (~26 min) → 01:30.
- **Post-fit main() first attempt**: crashed at `a.obs["pseudotime"]` because the parent-process `a` was loaded before the per-seed subprocesses wrote the full-cohort Palantir pseudotime back to `larry_day2_preproc_taskA.h5ad`. The three fits and the ~14K substrate were unaffected (per-seed `.pkl` and `.fitonly.pkl` all present and correct).
- **Fix** (part of the Task A commit): reload the 28K cache after the seed loop if `pseudotime` is missing.
- **Post-fit main() rerun**: cache-hit all three seeds; consensus (single-branch, R=3 passing cluster) + cohort labelling + features NPZ in 0.3 s. Cohort at completion: 970 labelled cells (523 Neut, 447 Mono), 632 unique eligible clones.
- **Eval**: `run_gate1_v2_fullcohort_eval.py` at 01:35. All five arms + paired CIs written. Verdict `FAIL Gate 1 v2 full-cohort` written to `gate1_v2_fullcohort_summary.json`.

**Completion summary**: three seeds completed on the ~14K substrate at 5,000 epochs, grid 100. All barcoded eligible cells retained (4,638 of 4,638 in the fit substrate; the classification cohort is a subset defined by the clone rule). Report at `REPORT_TaskA_Gate1_fullcohort.md`. Ledger row **EL10b** added; **EL10** marked superseded → **EL10b**. No compromised numbers reported anywhere.
