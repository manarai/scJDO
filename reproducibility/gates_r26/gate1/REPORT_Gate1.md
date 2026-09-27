# REPORT — Gate 1: LARRY early fate prediction (neutrophil vs monocyte)

**Prereg**: `PREREG_Gate1.md` (frozen through amendment `bd6fb0d`).
**Compute**: ~55 min (preproc + palantir 25 s + 3 × scJDO fits ~52 min via subprocess timeout guard + baselines).

## Verdict — **FAIL Gate 1**

Per the frozen primary criterion:
> mean(S) ≥ mean(best baseline) + 0.03 AND paired-bootstrap 95 % CI of (S − best baseline) excludes zero from the POSITIVE side.

**S is 0.215 BELOW the best baseline, with CI well-separated in the negative direction.**

| Feature set | AUROC (mean over 15 folds) | Notes |
|:---:|:---:|:---|
| E — PCA30 | **0.8345** | best baseline |
| F — FA30-substitute (=E per compute amendment) | 0.8345 | identical to E; PCA30 is the linear rep |
| K — Palantir fate probabilities (no clone labels) | 0.7840 | second-best baseline |
| C — local-covariance features (Gate 0b baseline) | 0.6220 | |
| **S — scJDO features (per-fold avg over 3 seeds)** | **0.6193** | S-features together underperform |
| E + S | 0.8239 | adding S HURTS E |
| F + S | 0.8239 | identical to E + S |
| C + S | 0.6382 | small non-significant lift over C |

Per-scJDO-seed spread of S alone: seed 0 = 0.6055, seed 1 = 0.5781, seed 2 = 0.6742 (across-seed spread 0.096 — larger than any real signal).

## Paired bootstrap 95 % CIs

| Contrast | Δ (S − baseline) | 95 % CI | Excludes 0? |
|:---:|:---:|:---:|:---:|
| S − E | -0.2152 | [-0.2692, -0.1563] | yes (negative) |
| S − F | -0.2152 | [-0.2692, -0.1563] | yes (negative) |
| S − K | -0.1647 | [-0.2143, -0.1191] | yes (negative) |
| S − C | -0.0027 | [-0.0432, +0.0360] | no |
| E+S − E | -0.0106 | [-0.0202, -0.0008] | yes (negative) |
| F+S − F | -0.0106 | [-0.0202, -0.0008] | yes (negative) |
| C+S − C | +0.0162 | [-0.0190, +0.0496] | no |

## Interpretation

- **Expression alone (PCA30 on 2 000 HVG at scaled log-normed counts) achieves AUROC 0.83** on day-2 fate prediction with clone-grouped 5-fold CV. That is the honest baseline for this task on this substrate.
- **scJDO features alone reach 0.62**, in the same range as local-covariance features (0.62). Both are far below expression. Every scJDO feature aggregates the raw expression signal through an operator + pseudotime-bin interpolation, which is exactly the kind of pooling that discards fine-grained cell-state distinctions among a cohort of early progenitors that all sit near the same pseudotime.
- **Adding scJDO features to expression significantly HURTS**: E+S is 0.010 lower than E, CI excludes zero. In a regularised LR at C = 1.0 and n = 154, extra noise features draw regulariser capacity away from the informative expression PCs.
- **CellRank fate probability (K) at 0.78** carries most of the trajectory-anchored signal without requiring scJDO's operator machinery. If the goal is fate prediction, the pseudotime + kNN structure alone (that CellRank uses) is closer to expression than scJDO's operator features.
- **The Jacobian carries no signal beyond local covariance on this task** (S − C = -0.003, CI includes 0). Gate 0b established that Jacobian archetypes are structurally distinct from covariance archetypes; Gate 1 establishes that this structural distinctness does not translate into fate-predictive signal on the LARRY substrate.

## Caveats from compute-scale amendments (all recorded before compute)

Each amendment was declared in the prereg before any downstream compute:
- **Substrate size**: 12,550 cells (from 113,612 myeloid). Every day-2 cell with a clone barcode is retained (4,550); day-4/6 Neut/Mono/Undiff terminals are subsampled to 2 000 each; day-2 Undiff without clone to 2 000. This shrinks the *resolvable* fate cohort from ~759 to **154** cells across **106** clones — because the "≥ 2 late Neut ∪ Mono cells" purity threshold requires late-time density that the subsample thins out.
- **Linear rep**: `X_fa` = 30 PCs (PCA30) rather than FactorAnalysis(30), because FA densified and OOM-killed on 36 GB. Consequently E and F are the same feature set here.
- **CellRank K feature**: full CellRank GPCCA on the pseudotime kernel failed at `set_terminal_states` (petsc4py absent → dense Brandts; a complex-eigenvalue split forced n_states=3 which broke the downstream index). Fell back to Palantir branch probabilities (still 2-D, still uses no clone labels — the K semantics are preserved: a fate-probability baseline).
- **scJDO fits**: n_epochs=3 000 (from 5 000), grid_size=150 (from 200), n_archetypes=5, vel_scale=0 per Gate 0c.

**Sensitivity of the verdict to these caveats**: the gap S ≪ E is ~0.22 absolute AUROC. The compute amendments cannot plausibly close that gap. If the full 759-cell cohort were available, sample-level power would rise but the AUROC gap is determined by the *feature-quality*, not the sample count. The negative CI on E+S − E rules out even a weak lift from adding scJDO on top of expression.

## Downstream decision per protocol

Per the user's decision table:
- Gate 0: PASS (0a Case 3, 0b PROCEED, 0c default = vel_scale=0 with one consensus mode).
- Gate 1: **FAIL**.
- Gate 2: still needed to decide {Pass-Fail-Pass → Assay-design paper} vs {Pass-Fail-Fail → Calibration paper with negatives}.

**Proceed to Gate 2** (scNT-seq metabolic labelling) to decide between the two remaining paths.

## Deliverables

- Prereg: `PREREG_Gate1.md` (frozen through amendment `bd6fb0d`).
- Runners: `run_gate1_build.py`, `_scjdo_one_seed.py`, `_build_h5ad.py`, `run_gate1_eval.py`.
- Machine-readable summary: `gate1_eval_summary.json`.
- Per-seed scJDO caches: `gate1_scjdo_seed{0,1,2}.pkl`.
- Combined features: `gate1_features.npz`.
- Per-fold AUROC arrays: `gate1_eval_aucs.npz`.
- Logs: `run_build.log`, `run_eval.log`.

**Next**: Gate 2 prereg + build.
