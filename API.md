# scJDO API Reference

Complete parameter documentation for all public functions.
For copy-pasteable examples see [`QUICKSTART.md`](QUICKSTART.md).

---

## `sjd.pp` — Preprocessing

### `prepare_trajectory`

```python
sjd.pp.prepare_trajectory(
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

## `sjd.tl` — Analysis

### `fit_drift`

```python
model = sjd.tl.fit_drift(
    adata,
    *,
    time_key      = 'pseudotime',  # any adata.obs column — use Palantir/Slingshot output here
    rep           = 'X_pca',       # any adata.obsm key  — use 'X_scvi', 'X_harmony', etc.
    branch_key    = None,          # optional: obsm probability matrix or obs float column
                                   # weights each Jacobian window by branch probability
                                   # (Palantir: adata.obsm['branch_probs'])
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
    # Temporal windowing (default: adaptive Gaussian kernel)
    windowing     = 'kernel',     # 'kernel' (default) | 'fixed'
    bandwidth     = 'auto',       # float | 'auto' — auto picks h* by S=R·C·L
    bandwidth_grid = (0.01, 0.02, 0.03, 0.05, 0.08, 0.10),  # auto sweep
    n_eff_min     = 30.0,         # effective sample-size floor
    adaptive      = False,        # kNN-adaptive h(τ) instead of global h
    knn_k         = 80,           # neighbours for adaptive bandwidth
    grid_size     = 200,          # kernel-mode eval grid size
    n_boot        = 20,           # bootstraps for bandwidth scoring
    # Legacy fixed-window scheme (used only when windowing='fixed')
    n_windows     = 100,          # pseudotime windows
    overlap       = 0.80,         # window overlap fraction
    smooth_sigma  = 1.5,          # Gaussian smoothing along time axis
    # Archetype decomposition
    n_archetypes  = 5,
    n_restarts    = 5,            # semi-NMF restarts
    # Output
    key_added     = 'scjdo',
    verbose       = True,
)
```

**Temporal aggregation.** By default `fit_drift` uses **adaptive Gaussian
kernel windowing** on a 200-point pseudotime grid, with the bandwidth $h^*$
selected automatically by maximising $S(h) = R(h)\cdot C(h)\cdot L(h)$
(bootstrap reproducibility × peak contrast × peak localisation), subject to
an effective-sample-size floor $n_\mathrm{eff}\ge$ `n_eff_min` over the
interior of the trajectory. The legacy 100-fixed-window / 80%-overlap
scheme is opt-in via `windowing='fixed'`. See
[`Manuscript/adaptive_kernel_windowing.ipynb`](Manuscript/adaptive_kernel_windowing.ipynb)
for the derivation, validation, and a side-by-side comparison.

**Common recipes**

```python
# Default — adaptive kernel, h selected from the data
sjd.tl.fit_drift(adata, n_epochs=5000)

# Pin the bandwidth (skip the auto sweep — faster on large data)
sjd.tl.fit_drift(adata, n_epochs=5000, bandwidth=0.05)

# Locally-adaptive bandwidth (recommended for very non-uniform pseudotime)
sjd.tl.fit_drift(adata, n_epochs=5000, adaptive=True, knn_k=80)

# Reproduce the pre-v0.4 fixed-window scheme exactly
sjd.tl.fit_drift(adata, n_epochs=5000, windowing='fixed',
                 n_windows=100, overlap=0.80, smooth_sigma=1.5)
```

**Stores in `adata.uns['scjdo']`:**

| Key | Shape | Description |
|---|---|---|
| `J_tensor` | (T, D, D) | Temporal Jacobian operator (T=`grid_size` in kernel mode, `n_windows` in fixed mode) |
| `t_centers` | (T,) | Grid pseudotime centers |
| `patterns` | (K, D, D) | Archetype Jacobian patterns |
| `activations` | (T, K) | Non-negative temporal activations |
| `act_norm` | (T, K) | Normalised activations for plotting |
| `max_real_eig` | (T,) | Max real eigenvalue per grid point |
| `instability_scores` | (T, n_genes) | Per-time gene instability scores |
| `corr_mat` | (K, K) | Archetype temporal correlation matrix |
| `gene_scores` | dict | Gene loadings per archetype |
| `top_genes` | dict | Top 50 genes per archetype |
| `top_instability_genes` | list | Globally ranked instability genes |
| `r2` | float | Semi-NMF reconstruction R² |
| `windowing` | str | `'kernel'` or `'fixed'` |
| `bandwidth` | float or (T,) | Selected $h^*$ (kernel mode) — `None` in fixed mode |
| `n_eff` | (T,) | Effective sample size per grid point (kernel mode) |
| `kernel_score` | dict | `{R, C, L, S}` for the selected bandwidth |
| `kernel_sweep` | list[dict] | Per-bandwidth scores (kernel + `bandwidth='auto'` only) |
| `lam_bootstrap` | (n_boot, T) | Bootstrap $\lambda_{\max}$ curves at $h^*$ |

**Stores in `adata.obsm`:**

| Key | Description |
|---|---|
| `X_drift` | Per-cell model drift vectors (D-dim) |
| `X_velocity_pseudo` | Pseudotime-gradient velocity prior (D-dim) |

---

### `fit_bridge`

```python
bridge = sjd.tl.fit_bridge(
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
    key_added     = 'scjdo_bridge',
    verbose       = True,
)
```

**Stores in `adata.uns['scjdo_bridge']`:**
`src_mask`, `tgt_mask`, `t_vals`, `J_fwd`, `J_bwd`, `max_eig_fwd`, `max_eig_bwd`,
`evec_fwd`, `evec_bwd`, `pat_fwd`, `pat_bwd`, `act_fwd`, `act_bwd`,
`df_fwd`, `df_bwd` (gene tables), `fwd_traj_2d`, `bwd_traj_2d` (for plotting),
`history` (training curves), `gene_names`, `params`.

---

### `fit_bridge_branches`

Run `fit_bridge` independently for each (source, target) pair sharing a single
control population — the natural API for snapshot perturb-seq, paired
treatment/control studies, or any one-source-vs-many-targets design. Each
target gets its own forward + backward bridge, Jacobian tensors, archetypes,
and gene tables.

```python
bridges = sjd.tl.fit_bridge_branches(
    adata,
    groupby     = 'target',           # obs column with the group labels
    src_group   = 'Non-Targeting',    # the shared source / control
    tgt_groups  = ['PVT1', 'MALAT1', 'PSMA3-AS1'],
    rep         = 'X_fa',
    time_key    = 'bridge_t',         # auto-created (0=src, 1=tgt) if missing
    key_prefix  = 'scjdo_bridge',     # results in adata.uns['scjdo_bridge_PVT1'], etc.
    # … any kwargs forwarded to fit_bridge (epsilon, max_iterations, …)
)
# bridges = {'PVT1': <Bridge>, 'MALAT1': <Bridge>, 'PSMA3-AS1': <Bridge>}
```

Returns `{tgt: Bridge}`. Each target's full results live in
`adata.uns[f'{key_prefix}_{tgt}']` and follow the same layout as
`fit_bridge` (see above). Missing targets warn-and-skip rather than raise.
Compose with `sjd.tl.infer_regulators_branches(direction='both')` for the
per-target forward + backward TF analysis.

---

### `fit_drift_branches`

Run `fit_drift` independently per lineage arm — eliminates branch-mixing in
pseudotime windows. Each branch gets its own Jacobian tensor and archetypes.

```python
models = sjd.tl.fit_drift_branches(
    adata,
    branch_key      = 'branch_probs',  # obsm matrix (Palantir) or obs label column
    branch_names    = ['erythroid', 'myeloid'],  # None = auto from uns
    branch_threshold= 0.5,             # min prob to include a cell (obsm mode)
    time_key        = 'pseudotime',
    n_archetypes    = 5,
    n_epochs        = 5000,
    key_prefix      = 'scjdo',       # results in adata.uns['scjdo_erythroid'], etc.
)
# Access: adata.uns['scjdo_erythroid'], adata.uns['scjdo_myeloid'], …
```

Accepts both:
- `obsm` key → probability matrix, e.g. Palantir `branch_probs` (N, n_branches)
- `obs` key → categorical labels, e.g. `adata.obs['lineage'] = 'erythroid'`

See [`INTEROPERABILITY.md`](INTEROPERABILITY.md) for full Palantir workflow.

---

### `branch_drift_analysis`

End-to-end per-branch wrapper around `fit_drift_branches`'s output: for each
branch, plots the instability genes, infers TF regulators, copies the
regulator entry back onto the full `AnnData` (so plotters can find it), and
writes per-branch `instab_{branch}.pdf` + `instability_genes_{branch}.csv` +
`regulators_{branch}.csv` artifacts.

```python
models                = sjd.tl.fit_drift_branches(adata, branch_key='branch_masks')
df_genes, df_regs     = sjd.tl.branch_drift_analysis(
    adata, models,
    key_prefix       = 'scjdo',
    n_genes          = 15,
    organism         = 'human',
    min_targets      = 1,
    n_top_regulators = 15,
    save_dir         = 'results/figure3/',
)
# df_genes / df_regs : {branch -> DataFrame}; regulators with no qualifying
# TFs fall back to an empty DataFrame with the canonical column set.
```

---

### `get_instability_genes`

```python
df = sjd.tl.get_instability_genes(
    adata,
    key           = 'scjdo',
    n_genes       = 20,
    min_sensitivity = 0.05,       # min Re(λ) to include a window
    top_archetypes  = None,       # restrict to K most instable archetypes
)
```

Returns a DataFrame: `rank, gene, mean_instability_score, peak_pseudotime, primary_archetype`.

---

### `get_bridge_instability_genes`

```python
df_fwd, df_bwd = sjd.tl.get_bridge_instability_genes(
    adata,
    key           = 'scjdo_bridge',
    n_genes       = None,
    top_archetypes= None,
)
```

Returns two DataFrames (forward and backward): `archetype, peak_t, mean_sensitivity, gene, instability_score, rank`.

---

### `infer_regulators`

```python
df = sjd.tl.infer_regulators(
    adata,
    key               = 'scjdo',     # or 'scjdo_bridge'
    direction         = 'forward',     # 'forward' | 'backward' | 'both' (bridge only)
    network           = None,          # custom DataFrame [source, target, weight]
    network_source    = 'auto',        # 'auto' | 'collectri' | 'trrust' | 'builtin'
    organism          = 'mouse',       # 'mouse' | 'human'
    min_targets       = 3,
    n_top             = 20,
    compute_pseudotime_lead = False,   # adds pseudotime_lead column (slower)
    denovo_n_top      = 15,            # genes per window for de novo edge inference
    key_added         = 'scjdo_regulators',
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

### `infer_regulators_branches`

Per-branch / per-perturbation regulator inference in one call. Loops
`infer_regulators` over the keys of a `branch_models` dict, handles the
subset → uns copy-back roundtrip that was previously hand-coded (and
silently bug-prone), and optionally writes per-branch CSVs. Works for
**both** drift branches (the standard `direction='primary'`) and bridges
(`direction='both'` returns separate forward + backward tables).

```python
# Drift case
models = sjd.tl.fit_drift_branches(adata, branch_key='branch_masks')
regs   = sjd.tl.infer_regulators_branches(
    adata, models,
    organism      = 'human',
    min_targets   = 1,
    n_top         = 15,
    save_csv_dir  = 'results/figure3/',   # writes results/figure3/{branch}/regulators_*.csv
)

# Bridge case — same call, both directions
bridges = sjd.tl.fit_bridge_branches(adata, groupby='target',
              src_group='Non-Targeting', tgt_groups=['PVT1','MALAT1'])
regs    = sjd.tl.infer_regulators_branches(
    adata, bridges,
    direction         = 'both',
    key_prefix        = 'scjdo_bridge',
    regulators_prefix = 'scjdo_regulators_bridge',
    organism          = 'human', min_targets=1, n_top=15,
)
```

Returns `{branch_name: primary_regulator_df}`. Full per-direction tables for
bridges are accessible via `adata.uns[f'{regulators_prefix}_{name}']['tables']`.

---

### `load_network`

```python
net = sjd.tl.load_network(
    organism = 'mouse',     # 'mouse' | 'human'
    source   = 'auto',      # 'auto' | 'collectri' | 'trrust' | 'builtin'
    custom   = None,        # your own DataFrame [source, target, weight]
)
```

Returns a DataFrame with columns `[source, target, weight]` for use as `network=` argument to `infer_regulators`.

---

## `sjd.pl` — Figures

All plotting functions accept `ax=None` (creates standalone figure) or an existing
`matplotlib.Axes` object for embedding in a larger layout. All accept `save=None` or
a file path string.

---

### Drift field panels

#### `summary_figure`
Four-panel layout: drift field | sensitivity | archetype profiles | coordination heatmap.
```python
sjd.pl.summary_figure(adata, key='scjdo', basis='X_pca', save=None)
```

#### `drift_field`
Streamplot (default) or quiver arrows on the PCA/UMAP embedding.
```python
sjd.pl.drift_field(
    adata,
    key           = 'scjdo',
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
sjd.pl.sensitivity(adata, key='scjdo', save=None)
```

#### `archetypes`
Temporal activation profiles for K archetypes.
```python
sjd.pl.archetypes(adata, key='scjdo', save=None)
```

#### `coordination`
Pairwise temporal correlation heatmap + coordination timeline.
```python
sjd.pl.coordination(adata, key='scjdo', save=None)
```

#### `instability_genes`
Three-panel: sensitivity curve | top genes across pseudotime | gene × archetype heatmap.
```python
sjd.pl.instability_genes(
    adata,
    key                   = 'scjdo',
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
sjd.pl.bridge_summary(adata, key='scjdo_bridge', basis='X_pca', save=None)
```

#### `bridge_trajectories`
PCA scatter colored by pseudotime with trajectory paths overlaid.
```python
sjd.pl.bridge_trajectories(
    adata,
    key       = 'scjdo_bridge',
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
sjd.pl.bridge_instability(adata, key='scjdo_bridge', save=None)
```

#### `bridge_archetypes`
Archetype activation profiles for both directions.
```python
sjd.pl.bridge_archetypes(adata, key='scjdo_bridge', save=None)
```

#### `bridge_genes`
Gene × archetype heatmaps for forward and backward.
```python
sjd.pl.bridge_genes(adata, key='scjdo_bridge', n_genes=15, save=None)
```

#### `bridge_gene_comparison`
Gene heatmaps + forward/backward unique gene lists printed to stdout.
```python
sjd.pl.bridge_gene_comparison(adata, key='scjdo_bridge', n_genes=15, save=None)
```

---

### Regulatory network panels

#### `regulator_summary`
Four-panel: bar chart | TF×archetype heatmap | scatter | target profiles.
```python
sjd.pl.regulator_summary(
    adata,
    key         = 'scjdo_regulators',
    scjdo_key = None,    # 'scjdo' or 'scjdo_bridge' (auto-detected)
    direction   = None,
    n_show      = 15,
    save        = None,
)
```

#### `regulator_barplot`
Horizontal bars: length = weighted_score, color = mean_instability, dot = n_targets.
```python
sjd.pl.regulator_barplot(adata, key='scjdo_regulators', n_show=20, save=None)
```

#### `regulator_heatmap`
TF × archetype instability heatmap.
```python
sjd.pl.regulator_heatmap(adata, key='scjdo_regulators', n_show=15, save=None)
```

#### `regulator_scatter`
Quality vs quantity scatter: X=n_targets, Y=mean_instability, size=weighted_score.
```python
sjd.pl.regulator_scatter(adata, key='scjdo_regulators', n_label=10, save=None)
```

#### `regulator_profiles`
Target instability across pseudotime for the top N regulators.
```python
sjd.pl.regulator_profiles(adata, key='scjdo_regulators', n_tfs=3, save=None)
```

#### `regulator_network`
Hybrid graph: solid edges = database-confirmed, dashed = de novo co-instability.
```python
sjd.pl.regulator_network(
    adata,
    key         = 'scjdo_regulators',
    scjdo_key = None,
    direction   = None,
    n_tfs       = 5,       # TF nodes to show
    n_targets   = 6,       # target gene nodes per TF
    n_denovo    = 4,       # de novo edges to show
    save        = None,
)
```

#### `branch_regulator_panels`
Per-branch / per-perturbation plot suite — writes the six regulator panels
above into per-branch subdirectories under `outdir/{branch}/`, with one
try/except harness per panel so a failure on one panel doesn't abort the rest.
Works for both drift branches and bridge results (pass `direction='forward'`
or `'backward'` for bridges).
```python
sjd.pl.branch_regulator_panels(
    adata, branch_models, outdir,
    panels            = ('barplot', 'heatmap', 'scatter',
                         'profiles', 'summary', 'network'),
    direction         = None,                   # 'forward' / 'backward' for bridges
    key_prefix        = 'scjdo',                # drift source-key prefix
    regulators_prefix = 'scjdo_regulators',     # regulator-key prefix
    file_ext          = 'pdf',
    panel_kwargs      = {                       # optional per-panel overrides
        'barplot':  {'n_show': 20},
        'network':  {'n_tfs': 5, 'n_targets': 6, 'n_denovo': 4},
    },
)
# Returns: {branch_name: list_of_panels_that_succeeded}
```

---

## CLI reference

```
usage: scjdo COMMAND [options]

Commands:
  drift     Drift field + archetype + instability genes + regulators
  bridge    Schrödinger Bridge + forward/backward instability + regulators

scjdo drift INPUT [options]
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

scjdo bridge INPUT [options]
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
