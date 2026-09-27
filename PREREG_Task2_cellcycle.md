# PREREG — Task 2: cell-cycle reversal, complete the figure

**Status**: pre-registered before compute. Frozen against `2c1f63f` (Task 1 commit).

## Purpose

Complete the cell-cycle reversal calibration by adding `vel_scale = 2.0` × {forward, reverse} × 3 seeds. Same cells and settings otherwise as the existing `vel_scale = 0` arms.

## Data (frozen, unchanged from `PREREG_cellcycle_reversal.md`)

- Synthetic 30-D circular trajectory: 1500 cells, latent 2-D circle (`(cos φ, sin φ) + N(0, σ²)` with σ = 0.05), embedded via fixed random orthogonal projection (seed = 42) + N(0, σ²=0.10²) ambient noise.
- τ_forward = φ / (2π); τ_reverse = 1 − τ_forward.

## Arms (frozen)

- `vel0_fwd` — vel_scale = 0, forward τ, 3 seeds {0, 1, 2}. **Already computed** (from earlier cell-cycle reversal run; reused verbatim).
- `vel0_rev` — vel_scale = 0, reversed τ, 3 seeds. **Already computed.**
- `vel2_fwd` — vel_scale = 2.0, forward τ, 3 seeds. **NEW compute.**
- `vel2_rev` — vel_scale = 2.0, reversed τ, 3 seeds. **NEW compute.**

Every other hyperparameter identical to the original prereg: `n_epochs = 3000`, `n_archetypes = 5`, `grid_size = 100`, `hidden = 256`, `depth = 4`, `sigma = 0.10`.

## Metrics per arm (frozen)

1. **Signed mean tangential drift** around the cycle:
   - For each cell `i` with true phase `φ_i`, the latent tangent direction is `t_i = (−sin φ_i, cos φ_i)`, embedded via the fixed projection W into 30-D.
   - Per-cell drift `v_i` from `fit_drift`'s J_tensor + local mean (same as Task 2 v1: `v_i = J(τ_i) @ (x_i − μ_bin_i)`).
   - `angular_drift_i = ⟨v_i, t_i⟩`, signed.
   - Per arm: mean over cells, per seed; then mean and spread over 3 seeds.

2. **Antisymmetric Jacobian projected on cycle plane**:
   - Let `W ∈ R^{2 × 30}` be the fixed embedding whose rows are orthonormal.
   - For each grid point τ_k and each seed, project the (30, 30) Jacobian into the 2-D cycle plane: `J_2d(τ_k) = W · J(τ_k) · W.T ∈ R^{2×2}`.
   - Antisymmetric part: `A(τ_k) = (J_2d(τ_k) − J_2d(τ_k).T) / 2 ∈ R^{2×2}`.
   - The signed off-diagonal `A_12(τ_k) = A[0, 1]` is the ROTATION strength around the axis normal to the cycle plane. Sign = sense of rotation (positive = counter-clockwise in (cos φ, sin φ) latent).
   - Per arm: report mean and spread over 3 seeds of `median_over_τ(A_12)` and `mean_over_τ(A_12)`.

## Pre-registered prediction (frozen)

- **At vel_scale = 2**: sign(angular_drift) flips between forward and reverse; sign(A_12) also flips between forward and reverse; magnitudes are similar (nearly antisymmetric).
- **At vel_scale = 0**: both `angular_drift` and `A_12` are ≈ 0 in both arms (confirmed by Task 2 v1: forward mean drift = −0.00017, reverse mean drift = −0.00017, non-antisymmetric).

## Figure (frozen)

4-panel PDF/PNG `fig_cellcycle_reversal.pdf` (replaces the 2-panel from Task 2 v1). Panels in order:
- (top-left) vel0-forward: 2-D scatter of (cos φ, sin φ) coloured by per-cell tangent-projected drift, quiver overlay at 60 evenly-spaced phases. Title = mean drift ± spread.
- (top-right) vel0-reverse: same layout.
- (bottom-left) vel2-forward: same layout.
- (bottom-right) vel2-reverse: same layout.

Colour scale is shared across all four panels (max abs of any panel). Signed diverging colourmap `RdBu_r`.

## Reported artefacts

- `REPORT_cellcycle.md` — human-readable with all 4 arms' numbers, prediction test, and figure caption.
- `cellcycle_summary_task2.json` — machine-readable numbers.
- `fig_cellcycle_reversal.pdf` + `.png` — the 4-panel figure.

## Compute budget

- 2 new arms × 3 seeds = 6 new scJDO fits at 1500 cells. Estimated ~3 min per seed → ~18 min for new arms.
- Metrics + figure < 30 s.
- **Total: ~20 min.**
