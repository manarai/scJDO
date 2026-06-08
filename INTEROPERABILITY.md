# scJDO — Interoperability Guide

scJDO accepts any latent representation stored in `adata.obsm` and any
pseudotime stored in `adata.obs`. This means you can use **any preprocessing
tool** — scVI, Palantir, Harmony, Scanorama, etc. — and hand the results
to scJDO with one parameter change.

---

## Pattern

```
Step 1  Run your preferred tool          (scVI / Palantir / Harmony / …)
Step 2  Store results in adata           (obsm / obs — standard AnnData slots)
Step 3  Pass keys to scJDO            (rep= / time_key= / branch_key=)
```

scJDO reads from standard AnnData slots and does not need to know which
tool produced them.

---

## scVI latent space

scVI produces a nonlinear, probabilistic latent embedding that handles dropout
and batch effects better than PCA for many datasets.

**When to prefer scVI over PCA:**
- Dataset has >20% zero entries (high dropout)
- Multiple batches that distort the manifold
- Subtle cell states not separated by top PCs
- Cells form a curved manifold rather than a linear one

```python
# Step 1 — install and run scVI
# pip install scvi-tools

import scvi
import scanpy as sc
import scjdo as sjd

adata = sc.read_h5ad("my_data.h5ad")

# Standard scRNA-seq preprocessing
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")

# Store raw counts (scVI requires them)
adata.layers["counts"] = adata.raw.X.copy()   # or adata.X before log1p

# Train scVI
scvi.model.SCVI.setup_anndata(adata, layer="counts",
                               batch_key=None)  # set batch_key if needed
model = scvi.model.SCVI(adata, n_latent=20, n_layers=2)
model.train(max_epochs=400, early_stopping=True)

# Step 2 — store latent representation
adata.obsm["X_scvi"] = model.get_latent_representation()

# Build graph on scVI space (better than PCA space)
sc.pp.neighbors(adata, use_rep="X_scvi", n_neighbors=15)
sc.tl.umap(adata)

# Compute pseudotime in scVI space
sc.tl.diffmap(adata)
adata.uns["iroot"] = 0   # set root cell index
sc.tl.dpt(adata)
adata.obs["pseudotime"] = adata.obs["dpt_pseudotime"]

# Step 3 — hand off to scJDO with rep='X_scvi'
sjd.tl.fit_drift(adata, rep="X_scvi", time_key="pseudotime")
sjd.pl.summary_figure(adata)
```

**Key parameter:** `rep="X_scvi"` tells `fit_drift` to use the scVI embedding
instead of PCA. Everything downstream (Jacobians, archetypes, gene scores)
automatically uses this space.

---

## Palantir pseudotime + branch probabilities

Palantir is better than DPT for branching trajectories because it returns
**per-branch probabilities** — each cell gets a probability of going to each
terminal state. scJDO uses these to weight the Jacobian computation so that
the erythroid windows are dominated by erythroid-fated cells, not a mixture.

**When to prefer Palantir over DPT:**
- Dataset has multiple branches (bifurcation, trifurcation)
- You want to run scJDO separately per lineage arm
- DPT produces unstable ordering at branch points

```python
# Step 1 — install and run Palantir
# pip install palantir

import palantir
import scanpy as sc
import scjdo as sjd

adata = sc.read_h5ad("my_data.h5ad")

# Preprocessing (PCA or scVI)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
sc.tl.pca(adata, n_comps=50)
sc.pp.neighbors(adata, n_neighbors=15)
sc.tl.umap(adata)

# Run MAGIC imputation (recommended by Palantir)
import magic
magic_op = magic.MAGIC()
X_magic  = magic_op.fit_transform(adata.to_df())

# Run Palantir
start_cell = "CELL_BARCODE_OF_ROOT"   # replace with your root cell barcode
pr_res = palantir.core.run_palantir(
    X_magic, start_cell,
    num_waypoints=500,
    use_early_cell_as_start=True,
)

# Step 2 — store Palantir results in standard AnnData slots
adata.obs["pseudotime"]       = pr_res.pseudotime
adata.obs["palantir_entropy"] = pr_res.entropy
adata.obsm["branch_probs"]    = pr_res.branch_probs.values   # (N, n_branches)
adata.uns["branch_names"]     = list(pr_res.branch_probs.columns)

print("Branches:", adata.uns["branch_names"])
print("Branch probs shape:", adata.obsm["branch_probs"].shape)

# Step 3a — branch-probability-weighted Jacobians (soft weighting)
# Each pseudotime window is weighted by branch probability.
# Erythroid-fated cells contribute more to early erythroid Jacobians.
sjd.tl.fit_drift(
    adata,
    time_key   = "pseudotime",
    branch_key = "branch_probs",   # obsm key with branch probability matrix
)
sjd.pl.summary_figure(adata)

# Step 3b — branch-separated analysis (hard separation, recommended)
# Runs fit_drift independently on each branch.
# Eliminates mixing entirely.
models = sjd.tl.fit_drift_branches(
    adata,
    branch_key   = "branch_probs",
    branch_names = adata.uns["branch_names"],
    n_archetypes = 5,
    n_epochs     = 5000,
)
# Results stored as: adata.uns["scjdo_erythroid"], adata.uns["scjdo_myeloid"], …
for branch, model in models.items():
    r2 = adata.uns[f"scjdo_{branch}"]["r2"]
    print(f"{branch}: R²={r2:.3f}")
```

---

## scVI + Palantir (recommended for branching data with dropout)

Combine both for maximum accuracy:

```python
# 1. scVI latent space
scvi.model.SCVI.setup_anndata(adata, layer="counts")
model_scvi = scvi.model.SCVI(adata, n_latent=20)
model_scvi.train(max_epochs=400)
adata.obsm["X_scvi"] = model_scvi.get_latent_representation()

# 2. Palantir pseudotime on scVI space
sc.pp.neighbors(adata, use_rep="X_scvi")
pr_res = palantir.core.run_palantir(
    pd.DataFrame(adata.obsm["X_scvi"], index=adata.obs_names),
    start_cell,
)
adata.obs["pseudotime"]    = pr_res.pseudotime
adata.obsm["branch_probs"] = pr_res.branch_probs.values
adata.uns["branch_names"]  = list(pr_res.branch_probs.columns)

# 3. scJDO — branch-separated, scVI latent space
models = sjd.tl.fit_drift_branches(
    adata,
    branch_key   = "branch_probs",
    n_archetypes = 5,
    n_epochs     = 5000,
)
```

---

## Harmony batch correction

```python
# pip install harmonypy

import harmonypy as hm
import scanpy as sc
import scjdo as sjd

sc.tl.pca(adata, n_comps=50)

# Run Harmony on PCA embedding
ho = hm.run_harmony(adata.obsm["X_pca"], adata.obs, "batch")
adata.obsm["X_harmony"] = ho.Z_corr.T

# Step 3 — use Harmony embedding in scJDO
sc.pp.neighbors(adata, use_rep="X_harmony")
sc.tl.umap(adata)
sc.tl.diffmap(adata); sc.tl.dpt(adata)
adata.obs["pseudotime"] = adata.obs["dpt_pseudotime"]

sjd.tl.fit_drift(adata, rep="X_harmony", time_key="pseudotime")
```

---

## Slingshot lineage pseudotime (via R)

Slingshot produces lineage-specific pseudotime, one per branch. Export from R
and import into Python:

```r
# In R
library(slingshot)
sds <- slingshot(sce, clusterLabels="cell_type", start.clus="HSC")

# Export per-lineage pseudotimes
write.csv(slingPseudotime(sds), "slingshot_pseudotime.csv")
```

```python
# In Python
import pandas as pd
import scjdo as sjd

pt = pd.read_csv("slingshot_pseudotime.csv", index_col=0)

# pt has one column per lineage, e.g. "Lineage1", "Lineage2"
adata.obs["pseudotime_erythroid"] = pt["Lineage1"].values
adata.obs["pseudotime_myeloid"]   = pt["Lineage2"].values

# Run scJDO per lineage
sjd.tl.fit_drift(adata, time_key="pseudotime_erythroid",
                  key_added="scjdo_erythroid")
sjd.tl.fit_drift(adata, time_key="pseudotime_myeloid",
                  key_added="scjdo_myeloid")
```

---

## Custom pseudotime from any source

scJDO only requires pseudotime in `adata.obs` normalized to [0, 1]:

```python
# Any pseudotime source
adata.obs["my_pseudotime"] = your_pseudotime_array  # must be in [0, 1]

sjd.tl.fit_drift(adata, time_key="my_pseudotime")
```

---

## Parameter reference for interoperability

| `fit_drift` parameter | Purpose | Example |
|---|---|---|
| `rep` | Latent space key in `adata.obsm` | `rep="X_scvi"`, `rep="X_harmony"` |
| `time_key` | Pseudotime column in `adata.obs` | `time_key="palantir_pseudotime"` |
| `branch_key` | Branch weights: obsm (matrix) or obs (float) | `branch_key="branch_probs"` |

| `fit_drift_branches` parameter | Purpose | Example |
|---|---|---|
| `branch_key` | obsm matrix or obs label column | `branch_key="branch_probs"` |
| `branch_names` | Names for each branch | `branch_names=["Ery","Neu"]` |
| `branch_threshold` | Min probability to include a cell | `branch_threshold=0.5` |

---

## Decision guide

```
My data has:

  ├─ One lineage, clean data         → PCA + DPT (default)
  │
  ├─ One lineage, noisy/dropout      → scVI + DPT
  │
  ├─ Multiple branches, clean        → PCA + Palantir + fit_drift_branches
  │
  ├─ Multiple branches, noisy        → scVI + Palantir + fit_drift_branches
  │
  ├─ Multiple batches                → Harmony/scVI + DPT or Palantir
  │
  └─ Pre-computed pseudotime exists  → any rep + time_key="your_column"
```
