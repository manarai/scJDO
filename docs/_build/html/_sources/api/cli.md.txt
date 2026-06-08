# Command line interface

The command line interface exposes complete drift and bridge pipelines for `.h5ad` files.

```bash
scjdo drift INPUT \
  --groupby paul15_clusters \
  --root 7MEP \
  --n-hvg 2000 \
  --n-archetypes 5 \
  --n-epochs 5000 \
  --out results/paul15_drift/
```

```bash
scjdo bridge INPUT \
  --groupby paul15_clusters \
  --root 7MEP \
  --src-quantile 0.20 \
  --tgt-quantile 0.80 \
  --n-archetypes 4 \
  --out results/paul15_bridge/
```

| Command | Purpose | Principal outputs |
|---|---|---|
| `scjdo drift` | Drift field, archetypes, instability genes, regulators | Annotated `.h5ad`, CSV tables, drift and regulator figures |
| `scjdo bridge` | Schrödinger Bridge, forward/backward instability, regulators | Annotated `.h5ad`, direction-specific gene tables, bridge figures |

## Common options

| Option | Description |
|---|---|
| `--groupby` | `adata.obs` column containing labels |
| `--root` | Root label for DPT pseudotime |
| `--n-hvg` | Number of highly variable genes |
| `--n-pcs` | Number of PCA components |
| `--n-archetypes` | Number of operator archetypes |
| `--seed` | Random seed |
| `--out` | Output directory |
