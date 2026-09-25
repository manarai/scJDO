# PREREG — Gate 0b: local-covariance archetype baseline

**Status**: pre-registered before compute. Frozen against `2a41fa1` (Gate 0a report).

## Question

Do scJDO's Jacobian archetypes carry more than local covariance? If a kernel-aggregated tensor built from local covariance matrices along τ (same grid, same bandwidth) semi-NMF-decomposes into archetypes that MATCH scJDO's Jacobian archetypes at cosine ≥ 0.9 for most components, then the operator does not add anything beyond local covariance and the tool paper is not defensible.

## Design (frozen)

- **Substrate**: marrow Ery branch cells (r24 pinned pipeline, 1,151 cells, X_fa 30-D FA rep, palantir_pseudotime).
- **scJDO tensor (reference)**: `fit_drift_branches` at Fig 3 defaults (seed=42, `vel_scale=2.0`, `bias_strength=1.5`, `hidden=256`, `depth=4`, `n_epochs=5000`, `n_archetypes=5`, `grid_size=200`, kernel windowing with adaptive bandwidth). Extract `J_tensor` shape (200, 30, 30).
- **Covariance tensor (baseline)**: build a tensor of shape (200, 30, 30) whose slice at grid point τ_i is a kernel-weighted local sample covariance of the X_fa embedding:
  `C_i = Σ_j w_ij (x_j − μ_i)(x_j − μ_i)^T / Σ_j w_ij`
  where `w_ij = exp(−(τ_j − τ_i)² / (2 h²))`, `h` = the adaptive bandwidth used by scJDO (read back from `adata.uns['scjdo_Ery']['bandwidth']`), and `μ_i = Σ_j w_ij x_j / Σ_j w_ij`. Same grid, same bandwidth, same cells.
- **Semi-NMF decomposition** on both tensors: use `scjdo.archetypes.decompose.jacobian_modes` (the same decomposition scJDO applies internally) with K=5, `n_restarts=5`, seed=42. Extract patterns (5, 30, 30) and activations (200, 5).
- **Matching**: Hungarian match on `|pattern cosine|` between the 5 scJDO archetypes and 5 covariance archetypes. Report the matched cosine per pair.
- **Gene-loading overlap**: project each pattern's top eigenvector to gene space via `fa_loadings`; take top-15 |gene loadings|; Jaccard per matched pair.
- **Seeds**: 3 seeds each for the semi-NMF decomposition (K, R combined). Report matched cosine and Jaccard as mean ± std over 3 seeds. scJDO's `fit_drift_branches` itself is not re-run per seed here (that is fixed at seed=42 for the reference tensor).

## Stop rule (frozen)

- If mean matched cosine across the 5 pairs (over 3 seeds of semi-NMF) is ≥ **0.9 for at least 3 of 5 pairs** → the Jacobian archetypes don't carry more than local covariance → **STOP the tool-paper track**. Write the calibration paper only.
- If matched cosine is < 0.9 for majority of pairs → the Jacobian adds distinctive structure; proceed to Gate 0c.
- Explicit intermediate: 3 pairs at cos ≥ 0.9 AND < 3 pairs at cos ≥ 0.9 both handled by the ≥3 rule. No ambiguity.

## Reported quantities

- Per-pair matched cosine (5 pairs × 3 seeds → 15 values; mean per pair + overall mean).
- Per-pair top-15 gene-loading Jaccard.
- Diagnostic: cosine matrix (5 × 5) before Hungarian match, to expose whether some archetypes match tightly and others don't.
- Bandwidth h used for both tensors.
- scJDO reference max_eig curve and covariance tensor's leading-eigenvalue curve, side-by-side.

## Not part of the gate

- Whether either tensor is biologically meaningful. This is a purely structural comparison.
- Comparison against dynamo or other peer methods. Handled in Gates 1/2.

## Expected wall clock

~10 min: scJDO fit is cached from Gate 0a (seed 0 A1 already computed and can be re-used, though for cleanness this gate refits at seed 42). Covariance tensor + semi-NMF: <2 min.
