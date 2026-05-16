# Use custom data

scQDiff can be used with any `AnnData` object that contains a suitable expression matrix and either a grouping variable for root-based DPT estimation or a precomputed pseudotime variable.

## Prepare a custom trajectory

```python
import anndata as ad
import scqdiff as sqd

adata = ad.read_h5ad('my_data.h5ad')
sqd.pp.prepare_trajectory(adata, groupby='cell_type', root='HSC')
sqd.tl.fit_drift(adata, time_key='pseudotime', n_archetypes=5)
```

## Use precomputed pseudotime

If `adata.obs['pseudotime']` already exists and is scaled to `[0, 1]`, you can skip DPT estimation and fit the drift model directly.

```python
sqd.tl.fit_drift(adata, time_key='pseudotime', n_archetypes=5)
```

## Use a custom TF-target network

A custom network should be a `pandas.DataFrame` with the columns `source`, `target`, and `weight`.

```python
import pandas as pd

my_net = pd.read_csv('my_network.csv')
df_reg = sqd.tl.infer_regulators(
    adata,
    network=my_net,
    organism='human',
)
```

| Column | Meaning |
|---|---|
| `source` | Transcription-factor gene symbol |
| `target` | Target gene symbol |
| `weight` | Edge confidence or signed regulatory weight |
