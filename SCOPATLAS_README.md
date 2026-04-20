# scOpAtlas: Stable Operator Atlas

**scOpAtlas** (Stable Operator Atlas) is an analytical layer within the **scqdiff** framework that defines cellular states by their **local stability structure** rather than expression patterns alone.

## Overview

Traditional cell atlases define cell states based on gene expression patterns. scOpAtlas adds a **dynamical layer** by characterizing cells according to their **operator regimes**—the local stability properties of the regulatory dynamics governing their behavior.

### Key Concept

> Cell states can be defined by their local operator regime—**stable**, **plastic**, or **unstable**—rather than expression alone.

This reveals:
- **Commitment depth** (how resistant a cell is to perturbation)
- **Plasticity** (how many directions a cell can move)
- **Bifurcation points** (where cell fate decisions occur)
- **Dynamical changes** invisible to expression-based analysis

## Mathematical Foundation

From the learned drift field `f(x,t)` in scqdiff, we compute the temporal Jacobian:

```
J(x,t) = ∂f/∂x
```

At each cell location, we extract four key metrics from the eigenvalue spectrum `{λᵢ}` of `J(x,t)`:

### Operator Metrics

| Metric | Formula | Interpretation | Biological Meaning |
|--------|---------|----------------|-------------------|
| **Max Unstable Eigenvalue** | λ_max⁺ = max(λᵢ) | Detects bifurcations | How easily cell escapes current state |
| **Stability Depth** | λ_min⁻ = min(λᵢ) | Damping strength | Commitment depth, resistance to perturbation |
| **Plasticity Index** | P = #{\|λᵢ\| < ε} / d | Fraction of neutral modes | Number of accessible directions |
| **Stable Subspace Dimension** | S = #{λᵢ < 0} | Buffering capacity | Number of stable directions |

### Operator Regimes

| Regime | Criteria | Biological Examples |
|--------|----------|-------------------|
| **Stable** | λ_max⁺ ≤ 0, large S | Terminal differentiation, homeostasis |
| **Plastic** | λ_max⁺ ≈ 0, high P | Progenitor states, decision points |
| **Unstable** | λ_max⁺ > τ | Transition states, bifurcations |
| **Deeply Stable** | Very negative λ_min⁻ | Resistant states, locked-in fates |

## Installation

scOpAtlas is included in the scqdiff package:

```bash
# Clone repository
git clone https://github.com/manarai/scqdiff_test.git
cd scqdiff_test

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Dependencies

**Required:**
- `torch` (for autograd Jacobian computation)
- `numpy`
- `anndata`
- `scanpy` (for visualization)
- `matplotlib`, `seaborn`

**Optional:**
- `cellrank` (for trajectory analysis integration)
- `episcanpy` (for ATAC-seq overlay)

## Quick Start

### Python API

```python
import torch
import anndata as ad
from scqdiff.models.drift import DriftField
from scqdiff.atlas import StableOperatorAtlas

# Load data and trained drift model
adata = ad.read_h5ad("your_data.h5ad")
drift_model = torch.load("your_model.pt")

# Build Stable Operator Atlas
atlas = StableOperatorAtlas(adata, drift_model)
atlas.build()

# Access results
print(atlas.regimes)  # Operator regime labels
print(atlas.metrics)  # Operator metrics

# Validate non-redundancy with cell types
atlas.validate_nonredundancy(celltype_key='cell_type')

# Save results
atlas.save("atlas_results.h5ad")
```

### Command Line Interface

```bash
# Build atlas from trained model
python -m scqdiff.atlas.build_atlas_cli \
    --h5ad data.h5ad \
    --model my_model.pt \
    --pseudotime-key pseudotime \
    --celltype-key cell_type \
    --condition-key treatment \
    --out atlas_results.h5ad
```

## Workflow

### 1. Train scqdiff Drift Model

First, train a drift field model using scqdiff:

```bash
python -m scqdiff.pipeline.train_from_anndata \
    --h5ad your_data.h5ad \
    --use-velocity-prior \
    --ptime-key pseudotime \
    --epochs 200 \
    --out-prefix my_model
```

### 2. Build Operator Atlas

```python
from scqdiff.atlas import StableOperatorAtlas

atlas = StableOperatorAtlas(
    adata=adata,
    drift_model=drift_model,
    use_rep="X_pca",
    pseudotime_key="pseudotime"
)

atlas.build(
    epsilon=0.1,                    # Threshold for near-neutral modes
    threshold_unstable=0.1,         # Unstable regime threshold
    threshold_plastic=0.05,         # Plastic regime threshold
    threshold_deeply_stable=-1.0,   # Deeply stable threshold
    batch_size=32
)
```

### 3. Validate Non-Redundancy

Critical for demonstrating that operator regimes provide information beyond expression-based cell types:

```python
validation = atlas.validate_nonredundancy(
    celltype_key='cell_type',
    condition_key='condition'
)
```

This tests:
- **Same cell type → different operator regimes** (regime diversity)
- **Same cell type, different conditions → different regime distributions**

### 4. Visualize Results

```python
from scqdiff.atlas.visualization import (
    plot_operator_regimes,
    plot_stability_depth_map,
    plot_plasticity_map,
    plot_nonredundancy_comparison
)

# Operator regimes on UMAP
plot_operator_regimes(adata, basis="umap")

# Stability depth map
plot_stability_depth_map(adata, basis="umap")

# Plasticity map
plot_plasticity_map(adata, basis="umap")

# Non-redundancy comparison (critical figure)
plot_nonredundancy_comparison(
    adata,
    celltype_key='cell_type',
    condition_key='condition'
)
```

## Key Features

### 1. Operator Metrics Computation

Compute eigenvalue-derived metrics from the Jacobian of the drift field:

```python
from scqdiff.atlas import OperatorMetrics

metrics_computer = OperatorMetrics(drift_model, epsilon=0.1)
metrics = metrics_computer.compute_all_metrics(X, t)

# Access individual metrics
lambda_max = metrics['lambda_max_plus']      # Max unstable eigenvalue
lambda_min = metrics['lambda_min_minus']     # Stability depth
plasticity = metrics['plasticity']           # Plasticity index
stable_dim = metrics['stable_dim']           # Stable subspace dimension
```

### 2. Regime Classification

Classify cells into operator regimes:

```python
from scqdiff.atlas import OperatorRegimeClassifier

classifier = OperatorRegimeClassifier(
    threshold_unstable=0.1,
    threshold_plastic=0.05
)

regimes, regime_masks = classifier.classify(metrics)
```

### 3. Condition Comparison

Compare operator regimes across experimental conditions:

```python
comparison = atlas.compare_conditions(
    condition_key='treatment',
    celltype_key='cell_type'
)
```

### 4. Temporal Evolution

Analyze how operator metrics evolve along pseudotime:

```python
from scqdiff.atlas.visualization import plot_temporal_evolution

plot_temporal_evolution(
    adata,
    pseudotime_key='pseudotime',
    n_bins=20
)
```

## Biological Applications

### 1. Immune Aging

**Question:** How do immune cells change with age?

**Approach:**
- Compare young vs. old donor samples
- Same cell type (e.g., naïve T cells), different operator regimes
- Hypothesis: Aging deepens stability without major expression changes

```python
atlas.validate_nonredundancy(
    celltype_key='cell_type',
    condition_key='age_group'  # 'young' vs 'old'
)
```

### 2. Drug Resistance

**Question:** What makes cells resistant to treatment?

**Approach:**
- Compare sensitive vs. resistant cell lines
- Identify deeply stable regimes in resistant cells
- Target interventions to shift operator regimes

### 3. Differentiation Commitment

**Question:** When do cells commit to a fate?

**Approach:**
- Track operator metrics along pseudotime
- Identify transition from plastic → stable regimes
- Locate bifurcation points (high λ_max⁺)

### 4. Exhaustion vs. Activation

**Question:** Can exhausted T cells be reactivated?

**Approach:**
- Compare operator regimes of exhausted vs. activated T cells
- Deeply stable exhausted cells → hard to reactivate
- Plastic exhausted cells → reversible

## Output Structure

After building the atlas, results are stored in the AnnData object:

### `adata.obs` (per-cell annotations)

| Column | Description |
|--------|-------------|
| `operator_regime` | Regime label (stable/plastic/unstable/deeply_stable) |
| `lambda_max_plus` | Max unstable eigenvalue |
| `lambda_min_minus` | Stability depth |
| `plasticity` | Plasticity index |
| `stable_dim` | Stable subspace dimension |
| `regime_confidence` | Classification confidence score |

### `adata.uns` (global metadata)

| Key | Description |
|-----|-------------|
| `operator_eigenvalues` | Full eigenvalue spectra (n_cells, n_dims) |

## Visualization Gallery

### 1. Operator Regimes on UMAP

Color cells by operator regime to reveal dynamical structure.

```python
plot_operator_regimes(adata, basis="umap")
```

**Colors:**
- 🟢 Green: Stable
- 🟠 Orange: Plastic
- 🔴 Red: Unstable
- 🔵 Blue: Deeply Stable

### 2. Stability Depth Map

Continuous heatmap showing commitment depth.

```python
plot_stability_depth_map(adata, basis="umap")
```

**Interpretation:**
- Dark regions: Deeply committed states
- Light regions: Shallow commitment, easily perturbed

### 3. Plasticity Map

Highlights decision points and progenitor states.

```python
plot_plasticity_map(adata, basis="umap")
```

**Interpretation:**
- High plasticity: Many accessible directions
- Low plasticity: Constrained motion

### 4. Non-Redundancy Comparison

**Critical figure for publication.** Shows that operator regimes differ across conditions even when expression-based cell types are the same.

```python
plot_nonredundancy_comparison(
    adata,
    celltype_key='cell_type',
    condition_key='condition'
)
```

### 5. Temporal Evolution

Track operator metrics along pseudotime to identify bifurcations and commitment transitions.

```python
plot_temporal_evolution(adata, pseudotime_key='pseudotime')
```

## Language Discipline

When describing scOpAtlas in papers and presentations:

### ✅ Always Say:
- "operator regime"
- "stability structure"
- "maintenance"
- "scqdiff application"
- "operator-based state definition"
- "dynamical layer of cell identity"

### ❌ Never Say:
- "new cell types"
- "redefine cell types"
- "replacement for cell atlases"
- "separate tool"

## Paper Positioning

### For Nature Computational Science:

> "We use scqdiff to construct a Stable Operator Atlas that defines cellular states by their local stability structure, revealing a dynamical layer of cell identity that is invisible to expression-based atlases."

### For OT Grants:

> "The Stable Operator Atlas provides a reusable, population-scale framework for quantifying robustness, plasticity, and control across heterogeneous immune systems."

## Validation Strategy

### Non-Redundancy Tests

**Option A (Preferred):** Same cell type → different operator regimes
- Example: Naïve T cells from young vs. old donors
- Same markers, different stability depth

**Option B:** Different cell types → shared operator regime
- Example: Multiple terminal lineages share "maintenance" stability program

### Biological Anchoring

Choose ONE strong biological axis:
- **Immune aging** (recommended)
- Exhaustion vs. reversible activation
- Drug resistance vs. sensitivity
- Differentiation commitment

### Chromatin Overlay (Optional)

Use ATAC-seq data to validate:
- Are unstable modes accessible?
- Are stable modes epigenetically reinforced?
- Do resistant states show chromatin locking?

## Advanced Usage

### Custom Thresholds

Adjust classification thresholds based on your data:

```python
atlas.build(
    epsilon=0.15,                    # Wider neutral zone
    threshold_unstable=0.2,          # Higher unstable threshold
    threshold_deeply_stable=-2.0     # Deeper stability requirement
)
```

### Batch Processing

For large datasets, increase batch size:

```python
atlas.build(batch_size=128)  # Process 128 cells at a time
```

### GPU Acceleration

Use GPU for faster computation:

```python
atlas = StableOperatorAtlas(
    adata, drift_model,
    device="cuda"  # Use GPU
)
```

## Troubleshooting

### Issue: "Pseudotime not found"

**Solution:** Compute pseudotime first using Scanpy or CellRank:

```python
import scanpy as sc
sc.tl.diffmap(adata)
sc.tl.dpt(adata)
```

### Issue: "Representation not found"

**Solution:** Compute PCA:

```python
sc.pp.pca(adata, n_comps=50)
```

### Issue: Jacobian computation is slow

**Solutions:**
1. Reduce batch size: `batch_size=16`
2. Use GPU: `device="cuda"`
3. Reduce dimensionality: Use fewer PCs

### Issue: All cells classified as same regime

**Solution:** Adjust thresholds:

```python
atlas.build(
    threshold_unstable=0.05,    # Lower threshold
    threshold_plastic=0.02      # Lower threshold
)
```

## Examples

See `examples/tutorial_scopatlas.py` for a complete tutorial.

## Citation

If you use scOpAtlas in your research, please cite:

```bibtex
@article{scqdiff2025,
  title={scqdiff: Schrödinger Bridge Learning of Single-Cell Regulatory Dynamics},
  author={Your Name et al.},
  journal={Nature Computational Science},
  year={2025}
}
```

## Support

For questions and issues:
- GitHub Issues: [github.com/manarai/scqdiff_test/issues](https://github.com/manarai/scqdiff_test/issues)
- Email: your.email@institution.edu

## License

MIT License

## Acknowledgments

scOpAtlas builds on the scqdiff framework for learning continuous-time cellular dynamics from single-cell data.
