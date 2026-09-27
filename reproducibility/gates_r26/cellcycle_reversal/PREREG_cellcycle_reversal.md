# PREREG — Cell-cycle reversal calibration figure

**Status**: pre-registered before compute. Frozen against `376796e` (Gate 2 redo report).

## Purpose

Calibration demonstration for the calibration paper: fit scJDO on a cycling substrate, show that the inferred drift field's rotation reverses direction when pseudotime is reversed (`τ → 1 − τ`). Single figure.

Not a gate — no pass/fail threshold. The claim under test: "scJDO's drift field respects the direction of pseudotime such that reversing the pseudotime coordinate reverses the inferred rotation."

## Design (frozen)

- **Substrate**: synthetic 30-D circular trajectory. Ground truth:
  - `n_cells = 1500`.
  - Cell phase `φ_i ~ Uniform([0, 2π))`.
  - Position in latent 2-D circle: `(cos φ_i, sin φ_i)` + Gaussian noise `σ = 0.05`.
  - Embed in 30-D via a fixed random orthogonal projection (seed = 42) + Gaussian noise `σ_amb = 0.10`.
  - Pseudotime `τ_i = φ_i / (2π)` (normalised to [0, 1)).
- **Arms**:
  - `forward`: scJDO fit with `τ_i` as-given.
  - `reverse`: scJDO fit with `1 − τ_i` as pseudotime (same latent positions).
- **Per arm fits**: 3 seeds (0, 1, 2) each for `fit_drift`. `vel_scale = 0` (Gate 0c default), `n_epochs = 3000`, `grid_size = 100` (single-branch), `n_archetypes = 5`.

## Metric (frozen)

At each cell, compute per-cell drift `v_i = model(x_i, τ_i)`. Project onto the tangent-to-circle direction at the cell's true phase φ: `t_tangent_i = (−sin φ_i, cos φ_i)` in latent coordinates, then embedded via the same 30-D projection.

- `angular_drift_i = <v_i, t_tangent_i>` (scalar, signed).
- `mean_angular_drift_forward` = mean over cells of angular_drift under `forward` arm.
- `mean_angular_drift_reverse` = mean over cells under `reverse` arm.
- Report both, averaged over 3 seeds with per-seed spread.

**Predicted**: `mean_angular_drift_forward > 0` AND `mean_angular_drift_reverse < 0` AND `|mean_angular_drift_forward + mean_angular_drift_reverse| ≪ |mean_angular_drift_forward|` (nearly antisymmetric).

## Figure (frozen)

Single PDF `cellcycle_reversal.pdf` with 2 panels:
- Panel A: forward-tau fit. 2-D circle scatter (projected back to true (cos φ, sin φ) coordinates) coloured by `angular_drift_i` under forward arm. Overlay quiver arrows sampled at 60 evenly-spaced phases showing per-cell drift projected onto the tangent direction.
- Panel B: reverse-tau fit. Same but under reverse arm.

Titles include the mean angular drift value and 3-seed spread.

## Not part of this demo

- Statistical hypothesis testing. This is a visualisation + descriptive numbers. If the qualitative claim holds (forward > 0 rotation, reverse < 0 rotation, magnitudes similar), the figure ships. If not, the figure is generated anyway and the result is reported as-is.
- Real cycling data. Synthetic is intentional to isolate the reversal calibration from confounds.

## Expected wall clock

- Data synthesis: ~1 s.
- 3 seeds × 2 arms × fit_drift ~ 3 min each = ~18 min.
- Figure: <10 s.
- **Total: ~20 min**.
