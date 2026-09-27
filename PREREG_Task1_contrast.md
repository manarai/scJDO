# PREREG — Task 1: Gate 2 redo temporal-contrast check

**Status**: pre-registered before compute. Frozen against `57a2e08` (Task 0 commit).

## Purpose

Reinterpret the Gate 2 redo results using two additional structural readouts on the already-computed R, G, and L Jacobian tensors:
1. **Temporal contrast** per fit (how much J varies across τ).
2. **Per-window L vs R** leading-eigvec cosine and Frobenius distance.
3. **Global linear baseline** J_lin = constant Jacobian from a `LinearRegression` of Dynamo velocities onto `X_pca`. Compare J_lin to R, J_L, J_G.

Uses the EXISTING artefacts saved in `New_analysis/gates_r26/gate2_redo/gate2_redo_arms.npz` — no new scJDO fits.

## Data (frozen)

- R: `R_v_ref` and `J_R` from `gate2_redo_arms.npz`.
- G: `G_seed{0,1,2}_J` from same archive.
- L: `L_seed{0,1,2}_J` from same archive.
- X_pca, tau, t_centers: from same archive.
- Reference from Gate 2 redo (bandwidth `h = 0.10`, `grid_size = 150`).

## Metrics (frozen)

### M1 — Temporal contrast per fit

For each Jacobian tensor `J` of shape (T, D, D):
- `J̄` = mean over the τ axis, shape (D, D).
- `contrast_i` = `‖J(τ_i) − J̄‖_F / ‖J̄‖_F` per grid point.
- Report the full profile (T = 150 values) and the median across τ.

Applied to: R (1 fit), G (3 seeds), L (3 seeds). For G and L, report per-seed values and mean ± spread.

### M2 — Per-window L vs R

At each grid point `τ_i`:
- Leading eigenvector: compute `eig(J(τ_i))` for R, and for each L seed. Take the leading (largest real part) eigenvector, sign-normalise. Compute `|cos(v_L_i, v_R_i)|` per L seed, average over seeds.
- Frobenius distance: `‖J_L(τ_i) − J_R(τ_i)‖_F` per L seed, average over seeds.

Report the full 150-point profiles + median.

### M3 — Global linear baseline

- `LinearRegression().fit(X_pca, R_v_ref)` on ALL 3060 cells (no CV — this is the population linear map).
- `J_lin = coef_.T`, a constant (D, D) = (30, 30) matrix. This is the constant global Jacobian implied by the linear map.
- Leading eigvec of J_lin sign-normalised.
- Per grid point τ_i and per L seed:
  - `|cos(leading eigvec of J_lin, leading eigvec of J_L(τ_i))|` — averaged over seeds and over τ.
  - Same for J_G and J_R.
- Normalised Frobenius correlation `⟨J_lin, J_L(τ_i)⟩ / (‖J_lin‖_F · ‖J_L(τ_i)‖_F)` — per grid point per L seed, averaged.
- Same for J_G, J_R.

## Pre-registered reading rule (frozen)

- **Reading A — "velocity supervision recovers the global linear response"** IF `median contrast(J_L) ≤ 1.2 × median contrast(J_R)` AND `cos(J_lin, J_L) > 0.95` (mean over τ and seeds).
- **Reading B — "velocity supervision yields a time-varying operator that removes seed dependence"** IF Reading A does NOT hold.

## Downstream test (conditional on Reading B)

If Reading B applies:
- **Held-out prediction test**: predict held-out cells' labelled velocity from `J_L` features vs `RidgeCV` on `X_pca`. Features per cell = `J_L(τ_i) @ (x_i − μ_bin)` (the same construction Gate 2 redo used for its per-cell drift; 30-D per cell per seed). 5-fold KFold on cells, 3 scJDO seeds. Report R² per fold, mean ± 95% bootstrap CI over the 15 fold values.
- If Reading A applies: this test is NOT run.

## P2 status (frozen adjustment)

- P2 from Gate 2 (cell-wise `cos(vel_L, R) > cos(vel_G, R)`) is **REMOVED from Gate 2's scored predictions** in this task's report. Reason (to be stated verbatim in the report): "R = `log1p(M_n_hvg) @ PCs_hvg` is nearly a linear function of `X_pca` on this dataset; the P2 comparison is dominated by a trivial linear-map baseline (Gate 2 redo Baseline_E reached cos = 0.99) rather than by scJDO structure. Retaining P2 in scored predictions would mis-attribute a linear-map identity to scJDO's operator."

## Reported artefacts

- `report_task1_contrast.md` — human-readable report.
- `gate2_task1_contrast.json` — machine-readable JSON.
- `fig_gate2_contrast.pdf` and `fig_gate2_contrast.png` — τ profiles of contrast, per-window cos(L, R), and cos(J_lin, J_arm).

## Compute budget

No scJDO fits — reuses existing arm caches. Estimated < 2 min for all metrics + figure.
