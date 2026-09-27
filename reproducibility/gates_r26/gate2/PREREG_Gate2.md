# PREREG — Gate 2: scNT-seq metabolic labelling (Qiu et al. 2022, Cell)

**Status**: pre-registered before compute. Frozen against `44e05f7` (Gate 1 report).

## Dataset (frozen)

- **Source**: Qiu et al. 2022 Cell — scNT-seq (single-cell New Transcript sequencing) with 4sU metabolic labelling. Access via `dynamo.sample_data.scNT_seq_neuron_labeling()` (auto-downloads `neuron_labeling.h5ad`).
- **Shape**: 3,060 cells × 24,078 genes; all annotated `cell_type = 'Neu'`; obs has `4sU`, `treatment`, `time`.
- **Layers**: `new` (T→C converted, newly transcribed during the 4sU window), `total` (all mRNA).

## Question

Do scJDO's Jacobian outputs agree with a labelling-anchored velocity reference (Dynamo run with metabolic-labelling supervision) on data-determined quantities, disagree on prior-determined quantities, and does adding labelling supervision to scJDO's V_ref buffer (arm L) narrow the gap to R and reduce seed-to-seed spread compared to geometry-only scJDO (arm G)?

## Arms (frozen)

- **R — labelling-based velocity reference**. Original design was `dyn.pp.recipe_monocle` → `dyn.tl.dynamics(model='deterministic')` → `dyn.tl.cell_velocities(basis='pca')` → `dyn.vf.VectorField` → `dyn.vf.jacobian`. **Compute-scale amendment (recorded before any Gate 2 metric was computed)**: on this dataset dynamo's kinetic estimation collapses to "only 4 genes have finite velocity values" (a documented failure mode for scNT-seq datasets where `time` labels the biological state, not the labelling duration). The VectorField step then fails at `KeyError: 'velocity_pca'`. R is reformulated to a labelling-ratio-based reference that does not depend on dynamo's kinetic pipeline: `v_ref_gene = log1p(M_n_hvg)` (moment-smoothed new-transcript layer, log1p-normalised the same way total-count `X_pca` is derived), then `v_ref = v_ref_gene @ PCs_hvg` (projected into PCA-30 space using dynamo-supplied HVG PCA loadings). This preserves the intent — R uses the labelling signal (new vs total mRNA), independent of geometric kNN — while side-stepping the broken kinetics call. Both G and L still see the same X_pca substrate; only the definition of `v_ref` changed.
- **G — geometry-only scJDO**. `fit_drift_branches` with `vel_scale = 0.0`, `bias_strength = 0` (or the minimum documented in Fig 3), `hidden = 256`, `depth = 4`, `n_epochs = 3000`, `n_archetypes = 5`, `grid_size = 150`, single-branch (no split — all cells are Neu). Rep: PCA-30 on log-normed HVG. 3 seeds.
- **L — labelling-supervised scJDO**. Same as G, plus `vel_scale = 2.0` and `V_ref` fed with the per-cell Dynamo velocity vectors projected into PCA-30 space (i.e. Dynamo's `velocity_N` interpolated to the 30 PCs). 3 seeds. This uses scJDO's existing `V_ref` buffer mechanism, so no new code path is added.

## Predictions (frozen)

### P1 — data-determined outputs
The data-determined output is the *shape* of the operator, primarily the leading Jacobian eigenvector direction (a unit vector per pseudotime bin) and the Re(λ_max) profile along pseudotime. These are set by the *positions* of cells in the rep + the kernel bandwidth, not by a velocity prior. Prediction: **cos(leading eigvec, R) between G and R ≥ 0.5 median across pseudotime bins**, AND **Spearman(Re(λ_max)_G, Re(λ_max)_R) ≥ 0.5** on the 150 bins.

*Reference for R*: since Dynamo does not natively produce a "Jacobian tensor along pseudotime", we derive the reference operator by kernel-aggregating Dynamo's Jacobian `dyn.vf.jacobian` outputs along pseudotime — same grid + bandwidth as scJDO. This aligns the comparison.

### P2 — prior-determined outputs
The prior-determined output is the *direction of drift*: per-cell velocity vectors themselves. G has no velocity prior → its per-cell drift is whatever emerges from geometry. L is anchored to R's velocity through V_ref. Prediction: **cell-wise cos(vel_L, vel_R) > cell-wise cos(vel_G, vel_R) with paired-bootstrap 95 % CI on Δ excluding 0**.

### P3 — seed stability
Prediction: **cross-seed variance of L (per-cell drift, averaged over cells) is LOWER than cross-seed variance of G**, since L has an external anchor. Reported as `var_L / var_G` with 95 % CI (bootstrap over cells).

## Success criteria (frozen)

- **Gate 2 PASS** requires ALL THREE predictions to hold as stated (with CI on P2 and P3 excluding zero in the predicted direction).
- **Gate 2 FAIL** if any prediction fails in the predicted direction (either the point estimate is on the wrong side, OR the CI includes zero).
- Reported regardless: any correlation on data-determined outputs (P1), any Δ_cos on prior-determined outputs (P2), and any variance-ratio (P3) — with 95 % CIs, even if the sign is not the predicted one.

## Reported quantities

- Per pseudotime bin: cosine of leading eigvec (G vs R, L vs R), Re(λ_max) (G, L, R).
- Per cell: cos(vel_G, vel_R), cos(vel_L, vel_R).
- Per seed: bin-wise leading eigvec across G seeds and L seeds; per-cell drift across seeds.
- Cross-seed variance of per-cell drift, computed as ⟨‖v_seed_i − mean(v)‖²⟩ per cell → averaged over cells.

## Stop rules (frozen)

- If R (Dynamo reference) cannot be produced (velocity estimation fails, vector field unstable, dyn.vf.jacobian returns NaN) — report the failure and stop Gate 2; downstream decision defers to Gate 1 alone.
- If G and L do not converge (nan losses, exception) across all seeds — report and stop. Same downstream fallback.
- No archetype consensus procedure needed here (single-cell-type substrate, no branch masks; Gate 0c consensus was defined for multi-branch marrow).

## Not part of the gate

- Fate prediction (no fate labels; single cell type).
- CellRank/CoSpar baselines (irrelevant on single cell type).
- Retraining Dynamo's velocity model per seed (Dynamo is deterministic given inputs; only scJDO fits vary across seeds).

## Expected wall clock

- Dynamo pipeline: ~15 min (labelling dynamics + reduceDimension + VectorField + jacobian on 3K cells).
- scJDO G × 3 seeds: ~15 min (single-branch fit at 3K cells is fast).
- scJDO L × 3 seeds: ~15 min (same fit but with V_ref buffer populated).
- Comparison metrics: <5 min.
- **Total: ~50 min**.
