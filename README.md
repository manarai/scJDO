# scQDiff — Operator-level single-cell dynamics

**scQDiff** infers how local dynamical sensitivity evolves during cell fate transitions.
It learns a drift field from scRNA-seq data, computes temporal Jacobian operators along
a trajectory, decomposes them into recurrent regulatory archetypes, and identifies the
genes and transcription factors that drive instability at each transition point.

```python
import scanpy as sc
import scqdiff as sqd

adata = sc.datasets.paul15()
sqd.pp.prepare_trajectory(adata, groupby='paul15_clusters', root='7MEP')
sqd.tl.fit_drift(adata, n_archetypes=5, n_epochs=5000)
sqd.pl.summary_figure(adata, save='figure3.pdf')
```

```bash
# Same analysis from the command line
scqdiff drift paul15.h5ad --groupby paul15_clusters --root 7MEP --out results/
```

---

## Install

```bash
git clone https://github.com/manarai/scQDiff
cd scQDiff
pip install -e .
```

Requires Python ≥ 3.9, PyTorch ≥ 2.2.

**Optional extras:**

```bash
pip install decoupler          # CollecTRI regulatory network (recommended)
pip install networkx           # regulator_network() graph figure
pip install faiss-cpu          # faster kNN for velocity prior
```

---

## Three workflows

| Workflow | When to use | Key call |
|---|---|---|
| **Drift field** | Any scRNA-seq with pseudotime | `sqd.tl.fit_drift` |
| **Schrödinger Bridge** | Two defined populations (young/old, treated/ctrl) | `sqd.tl.fit_bridge` |
| **Regulatory inference** | After either — link instability genes to TF regulators | `sqd.tl.infer_regulators` |

---

## API overview

### `sqd.pp` — Preprocessing

| Function | What it does |
|---|---|
| `prepare_trajectory(adata, groupby, root)` | Normalize → HVG → PCA → kNN → DPT pseudotime in one call |

### `sqd.tl` — Analysis

| Function | What it does |
|---|---|
| `fit_drift(adata, ...)` | Train drift field, compute Jacobian tensor, decompose into archetypes |
| `fit_bridge(adata, ...)` | Train Schrödinger Bridge between source/target populations |
| `get_instability_genes(adata)` | Extract top instability-driving genes per archetype (drift) |
| `get_bridge_instability_genes(adata)` | Same for forward and backward bridge directions |
| `infer_regulators(adata, ...)` | Link instability genes to upstream TF regulators via network database |

### `sqd.pl` — Figures

**Drift field:**

| Function | Figure |
|---|---|
| `summary_figure(adata)` | 4-panel: drift field, sensitivity, archetypes, coordination |
| `drift_field(adata)` | Streamplot on PCA embedding |
| `sensitivity(adata)` | Max Re(λ) across pseudotime |
| `archetypes(adata)` | Archetype activation profiles |
| `coordination(adata)` | Temporal correlation heatmap |
| `instability_genes(adata)` | Top instability genes across pseudotime + heatmap |

**Schrödinger Bridge:**

| Function | Figure |
|---|---|
| `bridge_summary(adata)` | 7-panel summary |
| `bridge_trajectories(adata)` | PCA + trajectory paths (forward / backward / both) |
| `bridge_instability(adata)` | Forward vs backward instability curves |
| `bridge_archetypes(adata)` | Archetype activation for both directions |
| `bridge_genes(adata)` | Gene × archetype heatmaps |
| `bridge_gene_comparison(adata)` | Forward vs backward unique gene lists |

**Regulatory network:**

| Function | Figure |
|---|---|
| `regulator_summary(adata)` | 4-panel: bar chart, heatmap, scatter, profiles |
| `regulator_barplot(adata)` | Ranked bar chart colored by mean instability |
| `regulator_heatmap(adata)` | TF × archetype instability heatmap |
| `regulator_scatter(adata)` | Quality vs quantity (n_targets vs mean_instability) |
| `regulator_profiles(adata)` | Target instability across pseudotime for top TFs |
| `regulator_network(adata)` | Hybrid graph: solid=reference, dashed=de novo |

### CLI

```bash
scqdiff drift  input.h5ad --groupby CLUSTER_COL --root ROOT_CLUSTER --out DIR/
scqdiff bridge input.h5ad --groupby CLUSTER_COL --root ROOT_CLUSTER --out DIR/
```

---

## What gets stored

Both `fit_drift` and `fit_bridge` store all results in `adata.uns` so every plotting
function can read directly without recomputing:

```
adata.uns['scqdiff']          ← drift results
adata.uns['scqdiff_bridge']   ← bridge results
adata.uns['scqdiff_regulators'] ← regulator inference results
adata.obsm['X_drift']         ← per-cell drift vectors
adata.obsm['X_velocity_pseudo'] ← pseudotime-gradient velocity prior
```

---

## Notebooks

Five end-to-end tutorials are in [`examples/`](examples/README.md):

| Notebook | Analysis |
|---|---|
| `01_paul15_hybrid_drift_tutorial` | Drift field + archetypes + instability genes + regulators |
| `02_paul15_fourier_tutorial` | Fourier-domain score network, spectral validation |
| `03_schrodinger_bridge_tutorial` | Bridge on 2D synthetic data, forward/backward instability |
| `04_paul15_schrodinger_tutorial` | Bridge on Paul15 PCA space, forward vs backward gene lists |
| `05_scopatlas_complete_workflow` | Operator atlas on pre-trained model |

Figure-generating notebooks for the manuscript are in [`Figures_notebook/`](Figures_notebook/).

---

## Mathematical background

For the full mathematical derivation see [`MATH.md`](MATH.md).

**Core idea:** model cell dynamics as a stochastic differential equation

$$dX_t = f_\theta(X_t, t)\,dt + \sigma\,dW_t$$

where the drift field $f_\theta$ is parameterized by a FiLM-conditioned neural network
trained via denoising score matching. Local Jacobians $J(x,t) = \nabla_x f_\theta$
are stacked across pseudotime into a tensor, then decomposed by semi-NMF into $K$
recurrent operator archetypes with non-negative temporal activation profiles.

---

## Citation

If you use scQDiff, please cite:

> Redd D., Green S., Terooatea T.W. (2026). scQDiff: Inferring time-varying dynamical
> operators from single-cell transcriptomic data. *[journal]*.

---

## Version

Current version: **0.3.0** — see [`CHANGELOG.md`](CHANGELOG.md) for what's new.
