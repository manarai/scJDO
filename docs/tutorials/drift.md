# Drift field analysis

This tutorial runs the core scJDO workflow on the Paul15 hematopoiesis dataset distributed with Scanpy. The analysis estimates a pseudotime-conditioned drift field, computes local Jacobian operators, decomposes those operators into archetypes, and extracts genes and regulators associated with instability.

## Complete workflow

```python
import scanpy as sc
import scjdo as sjd

adata = sc.datasets.paul15()

sjd.pp.prepare_trajectory(
    adata,
    groupby='paul15_clusters',
    root='7MEP',
    n_hvg=2000,
    n_pcs=50,
)

sjd.tl.fit_drift(
    adata,
    n_archetypes=5,
    n_epochs=5000,
)

sjd.pl.summary_figure(adata, save='results/figure3.pdf')

table = sjd.tl.get_instability_genes(adata, n_genes=20)
table.to_csv('results/instability_genes.csv', index=False)

df_reg = sjd.tl.infer_regulators(adata, organism='mouse')
df_reg.to_csv('results/regulators.csv', index=False)

sjd.pl.regulator_summary(adata, save='results/regulator_summary.pdf')
sjd.pl.regulator_network(adata, save='results/regulator_network.pdf')
```

## What happens at each stage

The preprocessing call normalizes counts, selects highly variable genes, computes PCA and a neighborhood graph, then estimates DPT pseudotime from a root population. The model-fitting call stores all results in `adata.uns['scjdo']`, while vector-valued drift quantities are written to `adata.obsm`.

| Stage | Function | Stored result |
|---|---|---|
| Trajectory preparation | `sjd.pp.prepare_trajectory` | `adata.obs['pseudotime']`, PCA, neighbors, optional UMAP |
| Drift training | `sjd.tl.fit_drift` | `adata.uns['scjdo']`, `adata.obsm['X_drift']`, `adata.obsm['X_velocity_pseudo']` |
| Instability genes | `sjd.tl.get_instability_genes` | Ranked `pandas.DataFrame` |
| Regulator inference | `sjd.tl.infer_regulators` | `adata.uns['scjdo_regulators']` |
| Visualization | `sjd.pl.summary_figure` and regulator plots | PDF or Matplotlib figure outputs |

## Parameter guidance

For quick exploration, `n_epochs=5000` is usually sufficient to test the workflow. For publication-quality experiments, run longer training on a GPU and record random seeds, software versions, and exact preprocessing choices.

| Parameter | Default | Practical interpretation |
|---|---|---|
| `n_archetypes` | `5` | Number of recurrent operator programs to extract |
| `windowing` | `'kernel'` | Temporal aggregation scheme: `'kernel'` (adaptive Gaussian) or `'fixed'` (legacy) |
| `bandwidth` | `'auto'` | Kernel bandwidth: `'auto'` picks $h^*$ by S=R·C·L; or pass a float |
| `grid_size` | `200` | Number of evaluation points across pseudotime (kernel mode) |
| `adaptive` | `False` | Use a kNN-adaptive bandwidth $h(\tau)$ for non-uniform pseudotime |
| `vel_scale` | `2.0` | Strength of pseudotime-gradient velocity prior |
| `vel_time_mode` | `'flat'` | Temporal gate shape for the velocity prior |
| `n_windows` | `100` | Window count when `windowing='fixed'` (legacy) |
| `smooth_sigma` | `1.5` | Post-binning Gaussian smoothing (legacy fixed-mode only) |
