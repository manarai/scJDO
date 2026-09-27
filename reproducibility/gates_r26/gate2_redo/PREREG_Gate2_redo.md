# PREREG — Gate 2 redo (velocity-matching loss + expression-only baseline)

**Status**: pre-registered before compute. Frozen against `6edbb1e` (Gate 3 strict report). No changes after commit.

## Purpose

Rerun Gate 2's arm comparison using a **velocity-matching loss inside scJDO training** (not the `vel_scale` prior slot that destabilised the v1 L arm), plus a new **expression-only linear-regression baseline** to bound the ceiling of what any linear map from expression can achieve on this reference. Three seeds for the L arm as before; predictions P1, P2, P3 unchanged.

## Data (unchanged from Gate 2)

- `dynamo.sample_data.scNT_seq_neuron_labeling()` — 3,060 cells, 24,078 genes, all `Neu`.
- `time` converted from minutes to hours.
- Reference `v_ref = log1p(M_n_hvg) @ PCs_hvg` (Gate 2 amendment `6f1d1cb`).

## Arms (frozen)

- **R — reference velocity**: identical to Gate 2 v1 (labelling-projected, from `log1p(M_n_hvg) @ PCs_hvg`).
- **G — geometry-only scJDO**: identical to Gate 2 v1. `fit_drift` with `vel_scale=0`, `n_epochs=3000`, `grid_size=150`, `n_archetypes=5`. 3 seeds {0, 1, 2}.
- **L — velocity-matching scJDO (NEW)**: `fit_drift`-shaped training loop with an added loss term:
  - `L_match = lambda_match * mean_i (1 - cos(model(x_i, t_i), V_ref_i))`
  - `lambda_match = 1.0` (frozen).
  - Direction-only cosine loss (NOT MSE) to be invariant to `V_ref` magnitude (which is ~13 for this dataset).
  - Everything else identical to G: same `vel_scale=0` (V_ref does NOT enter through the additive prior slot), same rep, same hyperparameters. Only the training objective differs.
  - 3 seeds {0, 1, 2}.
- **Baseline_E (NEW) — expression-only linear regression**: fit `sklearn.linear_model.LinearRegression` from `X_pca` (3060 × 30) to `R.v_ref` (3060 × 30) with 5-fold `KFold` cross-validation (grouped by nothing; cells are exchangeable). Predicted per-cell velocity vectors for each held-out fold, concatenated to a full-cohort prediction `v_baseline`. This is the ceiling of what any linear map from expression achieves.

## Predictions (unchanged from Gate 2)

Same as `PREREG_Gate2.md`. Also add a new pass check on baseline_E:

- **P1 — data-determined**: G median cos(leading eigvec of J_G, J_R) ≥ 0.5 AND Spearman(Re λ_max_G, Re λ_max_R) ≥ 0.5.
- **P2 — prior-determined**: cell-wise cos(vel_L, R) > cell-wise cos(vel_G, R) with paired-bootstrap 95% CI excluding 0.
- **P3 — seed stability**: median var_L(per-cell drift) / var_G < 1 with 95% CI < 1.

Additional readouts (reported regardless):
- P2': cell-wise cos(vel_L, R) − cos(vel_baseline_E, R) with 95% CI. Tests whether L beats a linear expression baseline at reconstructing R.
- P4 (informative only): cos(vel_baseline_E, R) mean and 95% CI. Answers "how well can any linear map from X_pca predict R?"

## Success criteria (frozen)

- **Gate 2 redo PASS** requires ALL THREE of P1, P2, P3 to hold as pre-declared.
- P2' and P4 are reported for interpretation but do not decide the gate.
- Anything else is **FAIL** (as Gate 2 v1).

## Implementation notes (frozen)

- Velocity-matching loss added by locally duplicating `fit_drift`'s training loop in a helper `fit_drift_vmatch` inside this gate's runner (5 lines of extra code in the training loop; no changes to `scjdo` package). The duplicated loop reads V_ref for the SAME batch indices `idx` used to sample `xb, tb` and computes `1 - cos(model(xb, tb), V_ref[idx])` averaged over the batch.
- Seed handling: same `torch.manual_seed(seed)` + `np.random.seed(seed)` per arm × seed. Deterministic within the seed.
- No archetype consensus procedure (single-cell-type substrate, matches Gate 2 v1 design).
- Cell-level P2/P3 metrics computed identically to Gate 2 v1 (`compute_per_cell_velocity_from_J` from `run_gate2.py`).

## Stop rules (frozen)

- If R (labelling-projected reference) still cannot be produced — should NOT happen since the reformulation in Gate 2 amendment; stop and report.
- If any of G or L fits diverges (NaN, exception) across all seeds — stop and report, downstream falls back to Gate 2 v1 decision.
- No re-tuning `lambda_match`. If L underperforms with `lambda_match = 1.0`, the gate FAILS on P2 and that is the answer — do not sweep weights post-hoc.

## Reported quantities

- Per pseudotime bin: cosine of leading eigvec (G vs R, L vs R), Re(λ_max) curves.
- Per cell: cos(vel_G, R), cos(vel_L, R), cos(vel_baseline_E, R).
- Per seed: bin-wise + per-cell metrics.
- Cross-seed variance of per-cell drift for G and L, median ratio + 95% CI.
- Training losses per seed for G and L (curve endpoints only, for the record).

## Not part of the gate

- Different `lambda_match` values.
- Weight-sharing L2 penalty on `V_ref` layers (Gate 2 v1 used monkey-patched `V_ref` additive; that path is entirely superseded).
- Alternative reference (e.g. dynamo's full kinetic pipeline; broken on this dataset per Gate 2 v1).

## Expected wall clock

- Dynamo preproc + reference build: ~2 min.
- G × 3 seeds: 3 × ~4 min = 12 min.
- L × 3 seeds (same training with extra loss term): 3 × ~4 min = 12 min.
- Baseline_E: <1 min.
- Metrics + bootstrap: <2 min.
- **Total: ~30 min**.

## Decision (Step 8 from Gate 3 chain — this is item 1 of that chain)

- **PASS** → the labelling-supervised route via velocity-matching loss produces a scJDO field that agrees with R better than geometry and beats the expression baseline. Report as a genuinely supervised improvement; still not a claim about unsupervised scJDO.
- **FAIL** → both v1 (additive V_ref) and this redo (matching loss) failed. The labelling-supervised route on scNT-seq is closed. Move to item 2 (cell-cycle reversal figure) and item 3 (Gate 0d).
