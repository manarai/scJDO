# scJDO Quick-Start

Copy-paste any of these blocks to run a complete analysis.

> **Using scVI, Palantir, Harmony, or Slingshot?**
> scJDO accepts any latent space or pseudotime from any tool —
> see [`INTEROPERABILITY.md`](INTEROPERABILITY.md) for drop-in patterns.

---

## Drift field analysis (Paul15 hematopoiesis)

```python
import scanpy as sc
import scjdo as sjd

# 1. Load data
adata = sc.datasets.paul15()

# 2. Preprocess — normalize, HVG, PCA, DPT pseudotime
sjd.pp.prepare_trajectory(
    adata,
    groupby = 'paul15_clusters',
    root    = '7MEP',          # progenitor root cluster
    n_hvg   = 2000,
    n_pcs   = 50,
)

# 3. Fit drift field + archetype decomposition
sjd.tl.fit_drift(
    adata,
    n_archetypes = 5,
    n_epochs     = 5000,       # use 50000 + GPU for publication quality
)

# 4. Generate all figures
sjd.pl.summary_figure(adata, save='results/figure3.pdf')

# 5. Extract instability genes + regulators
table = sjd.tl.get_instability_genes(adata, n_genes=20)
table.to_csv('results/instability_genes.csv', index=False)

df_reg = sjd.tl.infer_regulators(adata, organism='mouse')
df_reg.to_csv('results/regulators.csv', index=False)

sjd.pl.regulator_summary(adata, save='results/regulator_summary.pdf')
sjd.pl.regulator_network(adata, save='results/regulator_network.pdf')
```

---

## Schrödinger Bridge analysis (Paul15)

```python
import scanpy as sc
import scjdo as sjd

adata = sc.datasets.paul15()
sjd.pp.prepare_trajectory(adata, groupby='paul15_clusters', root='7MEP')

# Train bridge: bottom 20% pseudotime → top 20% pseudotime
sjd.tl.fit_bridge(
    adata,
    src_quantile = 0.20,       # progenitors
    tgt_quantile = 0.80,       # committed cells
    n_archetypes = 4,
)

# Summary figure (7 panels)
sjd.pl.bridge_summary(adata, save='results/bridge_summary.pdf')

# Forward and backward instability genes
df_fwd, df_bwd = sjd.tl.get_bridge_instability_genes(adata)
df_fwd.to_csv('results/genes_forward.csv',  index=False)
df_bwd.to_csv('results/genes_backward.csv', index=False)

# Regulators — scored separately per direction
sjd.tl.infer_regulators(adata, key='scjdo_bridge', direction='forward',
                          key_added='scjdo_regulators_fwd')
sjd.tl.infer_regulators(adata, key='scjdo_bridge', direction='backward',
                          key_added='scjdo_regulators_bwd')
sjd.pl.regulator_network(adata, key='scjdo_regulators_fwd',
                           scjdo_key='scjdo_bridge',
                           save='results/regulator_network_forward.pdf')
```

---

## Individual figure panels

Every panel in the summary figures is also callable standalone:

```python
# Drift field panels
sjd.pl.drift_field(adata,    save='drift_field.pdf')
sjd.pl.sensitivity(adata,    save='sensitivity.pdf')
sjd.pl.archetypes(adata,     save='archetypes.pdf')
sjd.pl.coordination(adata,   save='coordination.pdf')
sjd.pl.instability_genes(adata, n_genes=15, save='instability_genes.pdf')

# Bridge panels
sjd.pl.bridge_trajectories(adata, direction='forward',  save='traj_fwd.pdf')
sjd.pl.bridge_trajectories(adata, direction='backward', save='traj_bwd.pdf')
sjd.pl.bridge_instability(adata, save='instability.pdf')
sjd.pl.bridge_archetypes(adata,  save='archetypes.pdf')
sjd.pl.bridge_genes(adata,       save='genes.pdf')

# Regulator panels
sjd.pl.regulator_barplot(adata,  save='reg_bar.pdf')
sjd.pl.regulator_heatmap(adata,  save='reg_heatmap.pdf')
sjd.pl.regulator_scatter(adata,  save='reg_scatter.pdf')
sjd.pl.regulator_profiles(adata, save='reg_profiles.pdf')
sjd.pl.regulator_network(adata,  save='reg_network.pdf')
```

---

## CLI — complete pipeline in one command

```bash
# Drift field analysis
scjdo drift paul15.h5ad \
  --groupby paul15_clusters \
  --root    7MEP \
  --n-hvg   2000 \
  --n-archetypes 5 \
  --n-epochs     5000 \
  --out results/paul15_drift/

# Schrödinger Bridge analysis
scjdo bridge paul15.h5ad \
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
import scjdo as sjd

adata = ad.read_h5ad('my_data.h5ad')

# Option A: define root by cluster name
sjd.pp.prepare_trajectory(adata, groupby='cell_type', root='HSC')

# Option B: supply pre-computed pseudotime and skip DPT
#   adata.obs['pseudotime'] already in [0, 1]  →  just run fit_drift
sjd.tl.fit_drift(adata, time_key='pseudotime', n_archetypes=5)

# Option C: bridge with cell-type labels instead of pseudotime quantiles
sjd.tl.fit_bridge(adata,
                   groupby  = 'condition',
                   src_group= 'young',
                   tgt_group= 'old')

# Option D: supply a custom TF-target network
import pandas as pd
my_net = pd.read_csv('my_network.csv')   # columns: source, target, weight
sjd.tl.infer_regulators(adata, network=my_net, organism='human')
```

---

## Parameter reference

Key parameters for each workflow — see [`API.md`](API.md) for the full list.

### `sjd.tl.fit_drift`

| Parameter | Default | Notes |
|---|---|---|
| `n_archetypes` | 5 | Number of operator archetypes K |
| `n_epochs` | 5000 | Training iterations (50,000 on GPU for publication) |
| `vel_scale` | 2.0 | Pseudotime velocity prior strength |
| `vel_time_mode` | `'flat'` | Gate shape: flat / root / rise / mid |
| `windowing` | `'kernel'` | Temporal aggregation: `'kernel'` (default) or `'fixed'` (legacy) |
| `bandwidth` | `'auto'` | Kernel bandwidth: `'auto'` picks $h^*$, or pass a float (e.g. `0.05`) |
| `grid_size` | 200 | Pseudotime evaluation grid (kernel mode) |
| `adaptive` | `False` | Use kNN-adaptive $h(\tau)$ for very non-uniform pseudotime |
| `n_windows` | 100 | Pseudotime windows when `windowing='fixed'` (legacy only) |
| `hidden` | 256 | Network hidden dimension |
| `depth` | 4 | Network depth |

### `sjd.tl.fit_bridge`

| Parameter | Default | Notes |
|---|---|---|
| `src_quantile` | 0.20 | Bottom pseudotime fraction = source |
| `tgt_quantile` | 0.80 | Top pseudotime fraction = target |
| `n_archetypes` | 4 | Archetypes per bridge direction |
| `epsilon` | 0.5 | OT regularization (higher = smoother) |
| `t_steps` | 30 | Bridge time steps for Jacobian analysis |

### `sjd.tl.infer_regulators`

| Parameter | Default | Notes |
|---|---|---|
| `organism` | `'mouse'` | `'mouse'` or `'human'` |
| `network_source` | `'auto'` | CollecTRI → TRRUST → built-in |
| `min_targets` | 3 | Min shared targets to include a TF |
| `n_top` | 20 | Regulators to return |
| `direction` | `'forward'` | For bridge: forward / backward / both |

---

## Using external tools (scVI, Palantir, Harmony)

scJDO accepts any latent representation or pseudotime from any tool.
The three key parameters are:

```python
sjd.tl.fit_drift(
    adata,
    rep        = "X_scvi",          # any adata.obsm key  (default: X_pca)
    time_key   = "palantir_pseudo", # any adata.obs column (default: pseudotime)
    branch_key = "branch_probs",    # obsm probability matrix or obs float column
)
```

**Quick patterns:**

```python
# scVI latent space (run scVI first, store in adata.obsm["X_scvi"])
sjd.tl.fit_drift(adata, rep="X_scvi")

# Palantir pseudotime (run Palantir first, store in adata.obs)
sjd.tl.fit_drift(adata, time_key="palantir_pseudotime",
                  branch_key="branch_probs")

# Branch-separated (runs fit_drift independently per lineage)
models = sjd.tl.fit_drift_branches(
    adata,
    branch_key   = "branch_probs",   # from Palantir, or obs label column
    branch_names = ["erythroid", "myeloid"],
)
# Results: adata.uns["scjdo_erythroid"], adata.uns["scjdo_myeloid"]

# Harmony batch-corrected embedding
sjd.tl.fit_drift(adata, rep="X_harmony")
```

Full worked examples for each tool → [`INTEROPERABILITY.md`](INTEROPERABILITY.md)
