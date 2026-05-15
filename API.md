# scQDiff API Reference

Complete parameter documentation for all public functions.
For copy-pasteable examples see [`QUICKSTART.md`](QUICKSTART.md).

---

## `sqd.pp` — Preprocessing

### `prepare_trajectory`

```python
sqd.pp.prepare_trajectory(
    adata,
    *,
    groupby       = None,        # adata.obs column for cluster labels
    root          = None,        # root cluster name for DPT
    n_hvg         = 2000,        # highly variable genes
    n_pcs         = 50,          # PCA components
    n_neighbors   = 15,          # kNN graph neighbours
    time_key      = 'pseudotime',# key to store normalized pseudotime [0,1]
    compute_umap  = True,        # also compute UMAP embedding
    flavor        = 'seurat',    # HVG flavor
    copy          = False,       # return copy instead of in-place
)
```

Runs: normalize (10k CPM) → log1p → HVG selection → PCA → kNN graph → UMAP → DPT.
Stores normalized pseudotime in `adata.obs[time_key]`.

---

## `sqd.tl` — Analysis

### `fit_drift`

```python
model = sqd.tl.fit_drift(
    adata,
    *,
    time_key      = 'pseudotime',
    rep           = 'X_pca',
    # Architecture
    hidden        = 256,
    depth         = 4,
    beta          = 0.1,          # score network weight
    use_spectral_norm = True,
    # Velocity prior (auto-computed from pseudotime gradient)
    vel_scale     = 2.0,          # prior strength; increase if drift looks noisy
    vel_time_mode = 'flat',       # 'flat' | 'root' | 'rise' | 'mid'
    # Training
    n_epochs      = 5000,         # 50,000 + GPU for publication quality
    batch_size    = 512,
    lr            = 2e-4,
    weight_decay  = 1e-4,
    sigma         = 0.1,          # DSM noise level
    use_local_sigma = False,      # adaptive per-cell sigma
    seed          = 42,
    device        = None,         # auto: cuda if available
    # Jacobian tensor
    n_windows     = 100,          # pseudotime windows
    overlap       = 0.80,         # window overlap fraction
    smooth_sigma  = 1.5,          # Gaussian smoothing along time axis
    # Archetype decomposition
    n_archetypes  = 5,
    n_restarts    = 5,            # semi-NMF restarts
    # Output
    key_added     = 'scqdiff',
    verbose       = True,
)
```

**Stores in `adata.uns['scqdiff']`:**

| Key | Shape | Description |
|---|---|---|
| `J_tensor` | (T, D, D) | Smoothed Jacobian tensor |
| `t_centers` | (T,) | Window pseudotime centers |
| `patterns` | (K, D, D) | Archetype Jacobian patterns |
| `activations` | (T, K) | Non-negative temporal activations |
| `act_norm` | (T, K) | Normalised activations for plotting |
| `max_real_eig` | (T,) | Max real eigenvalue per window |
| `instability_scores` | (T, n_genes) | Per-window gene instability scores |
| `corr_mat` | (K, K) | Archetype temporal correlation matrix |
| `gene_scores` | dict | Gene loadings per archetype |
| `top_genes` | dict | Top 50 genes per archetype |
| `top_instability_genes` | list | Globally ranked instability genes |
| `r2` | float | Semi-NMF reconstruction R² |

**Stores in `adata.obsm`:**

| Key | Description |
|---|---|
| `X_drift` | Per-cell model drift vectors (D-dim) |
| `X_velocity_pseudo` | Pseudotime-gradient velocity prior (D-dim) |

---

### `fit_bridge`

```python
bridge = sqd.tl.fit_bridge(
    adata,
    *,
    time_key      = 'pseudotime',
    rep           = 'X_pca',
    # Population selection (choose one)
    src_quantile  = 0.20,         # bottom fraction → source
    tgt_quantile  = 0.80,         # top fraction → target
    src_group     = None,         # OR: cluster label for source
    tgt_group     = None,         # cluster label for target
    groupby       = None,         # adata.obs column (if using src/tgt_group)
    # Architecture
    hidden        = 256,
    depth         = 4,
    epsilon       = 0.5,          # OT regularization
    n_score_steps = 500,
    max_iterations= 30,
    # Analysis
    n_archetypes  = 4,
    t_steps       = 30,           # bridge time steps for Jacobian analysis
    n_traj        = 80,           # trajectories to simulate
    steps         = 100,          # integration steps
    n_genes       = 20,
    seed          = 42,
    device        = None,
    key_added     = 'scqdiff_bridge',
    verbose       = True,
)
```

**Stores in `adata.uns['scqdiff_bridge']`:**
`src_mask`, `tgt_mask`, `t_vals`, `J_fwd`, `J_bwd`, `max_eig_fwd`, `max_eig_bwd`,
`evec_fwd`, `evec_bwd`, `pat_fwd`, `pat_bwd`, `act_fwd`, `act_bwd`,
`df_fwd`, `df_bwd` (gene tables), `fwd_traj_2d`, `bwd_traj_2d` (for plotting),
`history` (training curves), `gene_names`, `params`.

---

### `get_instability_genes`

```python
df = sqd.tl.get_instability_genes(
    adata,
    key           = 'scqdiff',
    n_genes       = 20,
    min_sensitivity = 0.05,       # min Re(λ) to include a window
    top_archetypes  = None,       # restrict to K most instable archetypes
)
```

Returns a DataFrame: `rank, gene, mean_instability_score, peak_pseudotime, primary_archetype`.

---

### `get_bridge_instability_genes`

```python
df_fwd, df_bwd = sqd.tl.get_bridge_instability_genes(
    adata,
    key           = 'scqdiff_bridge',
    n_genes       = None,
    top_archetypes= None,
)
```

Returns two DataFrames (forward and backward): `archetype, peak_t, mean_sensitivity, gene, instability_score, rank`.

---

### `infer_regulators`

```python
df = sqd.tl.infer_regulators(
    adata,
    key               = 'scqdiff',     # or 'scqdiff_bridge'
    direction         = 'forward',     # 'forward' | 'backward' | 'both' (bridge only)
    network           = None,          # custom DataFrame [source, target, weight]
    network_source    = 'auto',        # 'auto' | 'collectri' | 'trrust' | 'builtin'
    organism          = 'mouse',       # 'mouse' | 'human'
    min_targets       = 3,
    n_top             = 20,
    compute_pseudotime_lead = False,   # adds pseudotime_lead column (slower)
    denovo_n_top      = 15,            # genes per window for de novo edge inference
    key_added         = 'scqdiff_regulators',
    verbose           = True,
)
```

**Six scoring metrics:**

| Metric | Column | What it captures |
|---|---|---|
| Weighted out-degree | `weighted_score` | Total instability explained by TF targets |
| Mean target instability | `mean_instability` | Quality: few sharp targets > many weak ones |
| Regulon enrichment | `enrichment_score` | -log10 hypergeometric p-value |
| Branch specificity | `branch_specificity` | Entropy-based archetype preference (0–1) |
| Database confidence | `db_confidence` | Mean edge weight in source network |
| Pseudotime lead | `pseudotime_lead` | TF peak before target peak (optional) |

**Network sources (tried in order with `auto`):**
1. CollecTRI via `decoupler` — broadest, most curated
2. TRRUST v2 — downloaded from web, signed edges
3. Built-in hematopoiesis network — always available, mouse-specific

---

### `load_network`

```python
net = sqd.tl.load_network(
    organism = 'mouse',     # 'mouse' | 'human'
    source   = 'auto',      # 'auto' | 'collectri' | 'trrust' | 'builtin'
    custom   = None,        # your own DataFrame [source, target, weight]
)
```

Returns a DataFrame with columns `[source, target, weight]` for use as `network=` argument to `infer_regulators`.

---

## `sqd.pl` — Figures

All plotting functions accept `ax=None` (creates standalone figure) or an existing
`matplotlib.Axes` object for embedding in a larger layout. All accept `save=None` or
a file path string.

---

### Drift field panels

#### `summary_figure`
Four-panel layout: drift field | sensitivity | archetype profiles | coordination heatmap.
```python
sqd.pl.summary_figure(adata, key='scqdiff', basis='X_pca', save=None)
```

#### `drift_field`
Streamplot (default) or quiver arrows on the PCA/UMAP embedding.
```python
sqd.pl.drift_field(
    adata,
    key           = 'scqdiff',
    basis         = 'X_pca',
    color         = 'pseudotime',   # or any adata.obs column
    velocity_key  = None,           # auto: X_velocity_pseudo for PCA, X_drift otherwise
    stream        = True,           # False for quiver arrows
    stream_density= 1.2,
    n_grid        = 30,
    save          = None,
)
```

#### `sensitivity`
Max real eigenvalue (local instability) across pseudotime.
```python
sqd.pl.sensitivity(adata, key='scqdiff', save=None)
```

#### `archetypes`
Temporal activation profiles for K archetypes.
```python
sqd.pl.archetypes(adata, key='scqdiff', save=None)
```

#### `coordination`
Pairwise temporal correlation heatmap + coordination timeline.
```python
sqd.pl.coordination(adata, key='scqdiff', save=None)
```

#### `instability_genes`
Three-panel: sensitivity curve | top genes across pseudotime | gene × archetype heatmap.
```python
sqd.pl.instability_genes(
    adata,
    key                   = 'scqdiff',
    n_genes               = 10,
    sensitivity_threshold = 0.05,
    per_archetype         = True,
    save                  = None,
)
```

Returns a pandas DataFrame (ranked gene table).

---

### Schrödinger Bridge panels

#### `bridge_summary`
Seven-panel summary: forward trajectories | backward trajectories | instability curves |
asymmetry | training convergence | forward archetypes | backward archetypes.
```python
sqd.pl.bridge_summary(adata, key='scqdiff_bridge', basis='X_pca', save=None)
```

#### `bridge_trajectories`
PCA scatter colored by pseudotime with trajectory paths overlaid.
```python
sqd.pl.bridge_trajectories(
    adata,
    key       = 'scqdiff_bridge',
    basis     = 'X_pca',
    direction = 'both',     # 'forward' | 'backward' | 'both'
    n_show    = 30,
    color     = 'pseudotime',
    save      = None,
)
```

#### `bridge_instability`
Forward vs backward instability curves.
```python
sqd.pl.bridge_instability(adata, key='scqdiff_bridge', save=None)
```

#### `bridge_archetypes`
Archetype activation profiles for both directions.
```python
sqd.pl.bridge_archetypes(adata, key='scqdiff_bridge', save=None)
```

#### `bridge_genes`
Gene × archetype heatmaps for forward and backward.
```python
sqd.pl.bridge_genes(adata, key='scqdiff_bridge', n_genes=15, save=None)
```

#### `bridge_gene_comparison`
Gene heatmaps + forward/backward unique gene lists printed to stdout.
```python
sqd.pl.bridge_gene_comparison(adata, key='scqdiff_bridge', n_genes=15, save=None)
```

---

### Regulatory network panels

#### `regulator_summary`
Four-panel: bar chart | TF×archetype heatmap | scatter | target profiles.
```python
sqd.pl.regulator_summary(
    adata,
    key         = 'scqdiff_regulators',
    scqdiff_key = None,    # 'scqdiff' or 'scqdiff_bridge' (auto-detected)
    direction   = None,
    n_show      = 15,
    save        = None,
)
```

#### `regulator_barplot`
Horizontal bars: length = weighted_score, color = mean_instability, dot = n_targets.
```python
sqd.pl.regulator_barplot(adata, key='scqdiff_regulators', n_show=20, save=None)
```

#### `regulator_heatmap`
TF × archetype instability heatmap.
```python
sqd.pl.regulator_heatmap(adata, key='scqdiff_regulators', n_show=15, save=None)
```

#### `regulator_scatter`
Quality vs quantity scatter: X=n_targets, Y=mean_instability, size=weighted_score.
```python
sqd.pl.regulator_scatter(adata, key='scqdiff_regulators', n_label=10, save=None)
```

#### `regulator_profiles`
Target instability across pseudotime for the top N regulators.
```python
sqd.pl.regulator_profiles(adata, key='scqdiff_regulators', n_tfs=3, save=None)
```

#### `regulator_network`
Hybrid graph: solid edges = database-confirmed, dashed = de novo co-instability.
```python
sqd.pl.regulator_network(
    adata,
    key         = 'scqdiff_regulators',
    scqdiff_key = None,
    direction   = None,
    n_tfs       = 5,       # TF nodes to show
    n_targets   = 6,       # target gene nodes per TF
    n_denovo    = 4,       # de novo edges to show
    save        = None,
)
```

---

## CLI reference

```
usage: scqdiff COMMAND [options]

Commands:
  drift     Drift field + archetype + instability genes + regulators
  bridge    Schrödinger Bridge + forward/backward instability + regulators

scqdiff drift INPUT [options]
  INPUT                  Path to input .h5ad file
  --groupby COLUMN       adata.obs column for cluster labels (required)
  --root    CLUSTER      Root cluster for DPT pseudotime (required)
  --n-hvg   N            Highly variable genes [default: 2000]
  --n-pcs   N            PCA components [default: 50]
  --n-neighbors N        kNN graph neighbours [default: 15]
  --n-archetypes K       Operator archetypes [default: 5]
  --n-epochs N           Training iterations [default: 5000]
  --n-windows N          Pseudotime windows [default: 100]
  --vel-scale F          Velocity prior strength [default: 2.0]
  --vel-time-mode MODE   flat|mid|root|rise [default: flat]
  --n-genes N            Instability genes to report [default: 20]
  --seed N               Random seed [default: 42]
  --out DIR              Output directory [default: drift_results/]

scqdiff bridge INPUT [options]
  INPUT                  Path to input .h5ad file
  --groupby COLUMN       adata.obs column (required)
  --root    CLUSTER      Root cluster for DPT (required)
  --src-quantile Q       Source population lower pseudotime bound [default: 0.20]
  --tgt-quantile Q       Target population upper pseudotime bound [default: 0.80]
  --src-group NAME       OR: source cluster label (overrides --src-quantile)
  --tgt-group NAME       Target cluster label (overrides --tgt-quantile)
  --n-archetypes K       Archetypes per direction [default: 4]
  --epsilon F            OT regularization [default: 0.5]
  --t-steps N            Bridge time steps [default: 30]
  --n-genes N            Instability genes to report [default: 20]
  --seed N               Random seed [default: 42]
  --out DIR              Output directory [default: bridge_results/]
```
