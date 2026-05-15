# scQDiff Quick-Start

Copy-paste any of these blocks to run a complete analysis.

---

## Drift field analysis (Paul15 hematopoiesis)

```python
import scanpy as sc
import scqdiff as sqd

# 1. Load data
adata = sc.datasets.paul15()

# 2. Preprocess — normalize, HVG, PCA, DPT pseudotime
sqd.pp.prepare_trajectory(
    adata,
    groupby = 'paul15_clusters',
    root    = '7MEP',          # progenitor root cluster
    n_hvg   = 2000,
    n_pcs   = 50,
)

# 3. Fit drift field + archetype decomposition
sqd.tl.fit_drift(
    adata,
    n_archetypes = 5,
    n_epochs     = 5000,       # use 50000 + GPU for publication quality
)

# 4. Generate all figures
sqd.pl.summary_figure(adata, save='results/figure3.pdf')

# 5. Extract instability genes + regulators
table = sqd.tl.get_instability_genes(adata, n_genes=20)
table.to_csv('results/instability_genes.csv', index=False)

df_reg = sqd.tl.infer_regulators(adata, organism='mouse')
df_reg.to_csv('results/regulators.csv', index=False)

sqd.pl.regulator_summary(adata, save='results/regulator_summary.pdf')
sqd.pl.regulator_network(adata, save='results/regulator_network.pdf')
```

---

## Schrödinger Bridge analysis (Paul15)

```python
import scanpy as sc
import scqdiff as sqd

adata = sc.datasets.paul15()
sqd.pp.prepare_trajectory(adata, groupby='paul15_clusters', root='7MEP')

# Train bridge: bottom 20% pseudotime → top 20% pseudotime
sqd.tl.fit_bridge(
    adata,
    src_quantile = 0.20,       # progenitors
    tgt_quantile = 0.80,       # committed cells
    n_archetypes = 4,
)

# Summary figure (7 panels)
sqd.pl.bridge_summary(adata, save='results/bridge_summary.pdf')

# Forward and backward instability genes
df_fwd, df_bwd = sqd.tl.get_bridge_instability_genes(adata)
df_fwd.to_csv('results/genes_forward.csv',  index=False)
df_bwd.to_csv('results/genes_backward.csv', index=False)

# Regulators — scored separately per direction
sqd.tl.infer_regulators(adata, key='scqdiff_bridge', direction='forward',
                          key_added='scqdiff_regulators_fwd')
sqd.tl.infer_regulators(adata, key='scqdiff_bridge', direction='backward',
                          key_added='scqdiff_regulators_bwd')
sqd.pl.regulator_network(adata, key='scqdiff_regulators_fwd',
                           scqdiff_key='scqdiff_bridge',
                           save='results/regulator_network_forward.pdf')
```

---

## Individual figure panels

Every panel in the summary figures is also callable standalone:

```python
# Drift field panels
sqd.pl.drift_field(adata,    save='drift_field.pdf')
sqd.pl.sensitivity(adata,    save='sensitivity.pdf')
sqd.pl.archetypes(adata,     save='archetypes.pdf')
sqd.pl.coordination(adata,   save='coordination.pdf')
sqd.pl.instability_genes(adata, n_genes=15, save='instability_genes.pdf')

# Bridge panels
sqd.pl.bridge_trajectories(adata, direction='forward',  save='traj_fwd.pdf')
sqd.pl.bridge_trajectories(adata, direction='backward', save='traj_bwd.pdf')
sqd.pl.bridge_instability(adata, save='instability.pdf')
sqd.pl.bridge_archetypes(adata,  save='archetypes.pdf')
sqd.pl.bridge_genes(adata,       save='genes.pdf')

# Regulator panels
sqd.pl.regulator_barplot(adata,  save='reg_bar.pdf')
sqd.pl.regulator_heatmap(adata,  save='reg_heatmap.pdf')
sqd.pl.regulator_scatter(adata,  save='reg_scatter.pdf')
sqd.pl.regulator_profiles(adata, save='reg_profiles.pdf')
sqd.pl.regulator_network(adata,  save='reg_network.pdf')
```

---

## CLI — complete pipeline in one command

```bash
# Drift field analysis
scqdiff drift paul15.h5ad \
  --groupby paul15_clusters \
  --root    7MEP \
  --n-hvg   2000 \
  --n-archetypes 5 \
  --n-epochs     5000 \
  --out results/paul15_drift/

# Schrödinger Bridge analysis
scqdiff bridge paul15.h5ad \
  --groupby    paul15_clusters \
  --root       7MEP \
  --src-quantile 0.20 \
  --tgt-quantile 0.80 \
  --n-archetypes 4 \
  --out results/paul15_bridge/
```

Both commands produce:

```
results/
  adata_drift.h5ad               ← annotated AnnData
  instability_genes.csv          ← ranked instability gene table
  figures/
    drift_summary.pdf            ← 4-panel figure
    drift_field.pdf
    sensitivity.pdf
    archetypes.pdf
    coordination.pdf
    instability_genes.pdf
    regulator_summary.pdf
    regulator_network.pdf
```

---

## Using your own data

```python
import anndata as ad
import scqdiff as sqd

adata = ad.read_h5ad('my_data.h5ad')

# Option A: define root by cluster name
sqd.pp.prepare_trajectory(adata, groupby='cell_type', root='HSC')

# Option B: supply pre-computed pseudotime and skip DPT
#   adata.obs['pseudotime'] already in [0, 1]  →  just run fit_drift
sqd.tl.fit_drift(adata, time_key='pseudotime', n_archetypes=5)

# Option C: bridge with cell-type labels instead of pseudotime quantiles
sqd.tl.fit_bridge(adata,
                   groupby  = 'condition',
                   src_group= 'young',
                   tgt_group= 'old')

# Option D: supply a custom TF-target network
import pandas as pd
my_net = pd.read_csv('my_network.csv')   # columns: source, target, weight
sqd.tl.infer_regulators(adata, network=my_net, organism='human')
```

---

## Parameter reference

Key parameters for each workflow — see [`API.md`](API.md) for the full list.

### `sqd.tl.fit_drift`

| Parameter | Default | Notes |
|---|---|---|
| `n_archetypes` | 5 | Number of operator archetypes K |
| `n_epochs` | 5000 | Training iterations (50,000 on GPU for publication) |
| `vel_scale` | 2.0 | Pseudotime velocity prior strength |
| `vel_time_mode` | `'flat'` | Gate shape: flat / root / rise / mid |
| `n_windows` | 100 | Pseudotime windows for Jacobian tensor |
| `hidden` | 256 | Network hidden dimension |
| `depth` | 4 | Network depth |

### `sqd.tl.fit_bridge`

| Parameter | Default | Notes |
|---|---|---|
| `src_quantile` | 0.20 | Bottom pseudotime fraction = source |
| `tgt_quantile` | 0.80 | Top pseudotime fraction = target |
| `n_archetypes` | 4 | Archetypes per bridge direction |
| `epsilon` | 0.5 | OT regularization (higher = smoother) |
| `t_steps` | 30 | Bridge time steps for Jacobian analysis |

### `sqd.tl.infer_regulators`

| Parameter | Default | Notes |
|---|---|---|
| `organism` | `'mouse'` | `'mouse'` or `'human'` |
| `network_source` | `'auto'` | CollecTRI → TRRUST → built-in |
| `min_targets` | 3 | Min shared targets to include a TF |
| `n_top` | 20 | Regulators to return |
| `direction` | `'forward'` | For bridge: forward / backward / both |
