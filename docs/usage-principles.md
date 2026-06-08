# Usage principles

scJDO is designed for operator-level interpretation of cell-state dynamics. It should be used after careful single-cell quality control and with explicit attention to how pseudotime, source-target definitions, and latent representations affect the learned dynamical system.

## Recommended workflow

The recommended workflow begins with an `AnnData` object containing raw or normalized expression values and a biologically meaningful grouping variable. Users should prepare pseudotime with `sjd.pp.prepare_trajectory` or supply a precomputed pseudotime in `[0, 1]`. They should then fit either drift dynamics or a Schrödinger Bridge, extract instability genes, and finally run regulator inference if a transcription factor target network is available.

| Principle | Practical rule |
|---|---|
| Preserve reproducibility | Save the fitted `AnnData` object and all parameter settings. |
| Separate exploration from publication runs | Use short training runs for debugging and longer GPU runs for final analysis. |
| Interpret operators locally | Treat Jacobians and archetypes as local sensitivity summaries, not direct causal proof. |
| Validate regulator hypotheses | Use inferred TFs as candidate hypotheses for follow-up validation. |
| Compare directions explicitly | In bridge analyses, inspect forward and backward instability rather than averaging them prematurely. |

## Data expectations

scJDO expects cells to be embedded in a continuous latent space, usually `adata.obsm['X_pca']`. Pseudotime should be monotonic along the biological process being analyzed, and source-target labels should represent meaningful endpoint distributions for bridge analysis.

## Result storage

Both drift and bridge workflows write structured results into `AnnData`, enabling plotting without recomputation.

| Location | Content |
|---|---|
| `adata.uns['scjdo']` | Drift Jacobians, archetypes, eigenvalues, instability scores, and gene rankings |
| `adata.uns['scjdo_bridge']` | Forward and backward bridge Jacobians, archetypes, trajectories, and gene tables |
| `adata.uns['scjdo_regulators']` | Ranked transcription-factor table and network metadata |
| `adata.obsm['X_drift']` | Per-cell drift vectors |
| `adata.obsm['X_velocity_pseudo']` | Pseudotime-gradient velocity prior |
