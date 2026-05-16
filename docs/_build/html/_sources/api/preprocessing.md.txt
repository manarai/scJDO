# Preprocessing: `pp`

The preprocessing namespace contains the high-level trajectory preparation entry point.

## `scqdiff.pp.prepare_trajectory`

```{eval-rst}
.. autofunction:: scqdiff.pp.prepare_trajectory
```

## Summary

`prepare_trajectory` performs normalization, log transformation when appropriate, highly variable gene selection, PCA, nearest-neighbor graph construction, optional UMAP computation, and DPT pseudotime estimation when a root group is supplied.

| Parameter | Default | Description |
|---|---|---|
| `groupby` | `None` | `adata.obs` column containing cluster labels |
| `root` | `None` | Root cluster label for DPT pseudotime |
| `n_hvg` | `2000` | Number of highly variable genes |
| `n_pcs` | `50` | Number of PCA components |
| `n_neighbors` | `15` | Neighborhood graph size |
| `time_key` | `'pseudotime'` | Destination key in `adata.obs` |
