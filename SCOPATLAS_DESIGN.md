# scOpAtlas Design Document

## Overview

**scOpAtlas** (Stable Operator Atlas) is a new analytical layer within the scqdiff framework that defines cellular states by their **local stability structure** rather than expression patterns alone. It leverages scqdiff's core capabilities (drift field inference, Jacobian computation, eigenmode analysis) to construct an atlas of operator regimes that reveal the dynamical properties of cells.

## Conceptual Framework

### Positioning
- **scqdiff** = the engine (drift + Jacobians + eigenmodes)
- **scOpAtlas** = a product/application of scqdiff (operator-based state definition)
- This is a **layer**, not a separate tool

### Core Claim
> Cell states can be defined by their local operator regime—stable, plastic, or unstable—rather than expression alone.

## Mathematical Foundation

### 1. Operator Spectrum Analysis

From the learned drift field `f(x,t)`, we compute the temporal Jacobian:

```
J(x,t) = ∂f/∂x
```

At each cell location and time point, we compute the eigenvalue spectrum `{λᵢ}` of `J(x,t)`.

### 2. Operator-Derived State Metrics

Four key scalar metrics define the operator regime at each cell:

#### a) Max Unstable Eigenvalue
```
λ_max⁺ = max(λᵢ)
```
- **Interpretation**: Detects bifurcation points and control sensitivity
- **Biological meaning**: How easily the cell can escape its current state

#### b) Stability Depth
```
λ_min⁻ = min(λᵢ)
```
- **Interpretation**: How strongly deviations are damped
- **Biological meaning**: Commitment depth and resistance to perturbation

#### c) Plasticity Index
```
P = #{|λᵢ| < ε} / d
```
- **Interpretation**: Fraction of near-neutral modes
- **Biological meaning**: Number of accessible directions without strong restoring forces

#### d) Stable Subspace Dimension
```
S = #{λᵢ < 0}
```
- **Interpretation**: Buffering capacity
- **Biological meaning**: Number of stable directions

### 3. Operator Regime Classification

| Operator Regime | Criteria | Biological Interpretation |
|----------------|----------|---------------------------|
| **Stable** | λ_max⁺ ≤ 0, large S | Terminal differentiation, homeostasis |
| **Plastic** | λ_max⁺ ≈ 0, high P | Progenitor states, decision points |
| **Unstable** | λ_max⁺ > τ | Transition states, bifurcations |
| **Deeply Stable** | Very negative λ_min⁻ | Resistant states, locked-in fates |

## Implementation Architecture

### Module Structure

```
scqdiff/
├── atlas/                      # NEW: scOpAtlas module
│   ├── __init__.py
│   ├── operator_metrics.py     # Compute λ_max⁺, λ_min⁻, P, S
│   ├── regime_classifier.py   # Classify operator regimes
│   ├── atlas_builder.py       # Build and manage atlas objects
│   └── validation.py          # Non-redundancy tests
├── models/
│   └── drift.py               # EXTEND: Add Jacobian eigenvalue computation
└── viz/
    └── atlas_plots.py         # NEW: Visualization for operator regimes
```

### Key Classes

#### 1. `OperatorMetrics`
```python
class OperatorMetrics:
    """Compute operator-derived state metrics from Jacobian eigenvalues."""
    
    def __init__(self, jacobian_field, epsilon=0.1):
        self.jacobian_field = jacobian_field
        self.epsilon = epsilon
    
    def compute_eigenvalues(self, x, t):
        """Compute eigenvalues of J(x,t)."""
        pass
    
    def max_unstable_eigenvalue(self, x, t):
        """λ_max⁺ = max(λᵢ)"""
        pass
    
    def stability_depth(self, x, t):
        """λ_min⁻ = min(λᵢ)"""
        pass
    
    def plasticity_index(self, x, t):
        """P = #{|λᵢ| < ε} / d"""
        pass
    
    def stable_subspace_dim(self, x, t):
        """S = #{λᵢ < 0}"""
        pass
```

#### 2. `OperatorRegimeClassifier`
```python
class OperatorRegimeClassifier:
    """Classify cells into operator regimes."""
    
    def __init__(self, threshold_unstable=0.1, threshold_plastic=0.05):
        self.tau = threshold_unstable
        self.eps = threshold_plastic
    
    def classify(self, metrics):
        """
        Returns: regime labels ("stable", "plastic", "unstable", "deeply_stable")
        """
        pass
```

#### 3. `StableOperatorAtlas`
```python
class StableOperatorAtlas:
    """Main atlas object containing operator regime information."""
    
    def __init__(self, adata, drift_model):
        self.adata = adata
        self.drift_model = drift_model
        self.metrics = None
        self.regimes = None
    
    def build(self, time_points=None):
        """Compute metrics and classify regimes for all cells."""
        pass
    
    def compare_conditions(self, condition_key):
        """Compare operator regimes across conditions."""
        pass
    
    def validate_nonredundancy(self, celltype_key):
        """Test that operator regimes are not redundant with cell types."""
        pass
    
    def to_anndata(self):
        """Store atlas results in AnnData object."""
        pass
```

### Integration with Existing scqdiff

#### Extend `DriftField` class
```python
class DriftField(nn.Module):
    # ... existing code ...
    
    def compute_jacobian(self, x, t):
        """Compute Jacobian ∂f/∂x at (x,t)."""
        # Use torch.autograd to compute gradient
        pass
    
    def compute_jacobian_eigenvalues(self, x, t):
        """Compute eigenvalues of J(x,t)."""
        J = self.compute_jacobian(x, t)
        eigenvalues = torch.linalg.eigvals(J)
        return eigenvalues
```

## API Design

### Python API

```python
import anndata as ad
from scqdiff.models.drift import DriftField
from scqdiff.atlas import StableOperatorAtlas

# Load trained drift model
adata = ad.read_h5ad("data.h5ad")
drift_model = DriftField.load("my_model.pt")

# Build Stable Operator Atlas
atlas = StableOperatorAtlas(adata, drift_model)
atlas.build(time_points=None)  # Use pseudotime from adata

# Access operator metrics
print(atlas.metrics.keys())  # ['lambda_max_plus', 'lambda_min_minus', 'plasticity', 'stable_dim']

# Access regime classifications
print(atlas.regimes)  # Array of regime labels

# Validate non-redundancy with cell types
atlas.validate_nonredundancy(celltype_key='cell_type')

# Store results in AnnData
adata_with_atlas = atlas.to_anndata()
```

### Command Line Interface

```bash
# Build atlas from trained model
python -m scqdiff.atlas.build_atlas \
    --h5ad data.h5ad \
    --model my_model.pt \
    --ptime-key pseudotime \
    --out atlas_results.h5ad

# Compare regimes across conditions
python -m scqdiff.atlas.compare_conditions \
    --h5ad atlas_results.h5ad \
    --condition-key treatment \
    --celltype-key cell_type \
    --out comparison_report.pdf
```

## Validation Strategy

### 1. Non-Redundancy Tests

**Option A (Preferred)**: Same cell type → different operator regimes
- Example: Naïve T cells from young vs. old donors
- Same expression markers, different stability depth

**Option B**: Different cell types → shared operator regime
- Example: Multiple terminal lineages share "maintenance" stability program

### 2. Biological Anchoring

Choose ONE strong biological axis:
- **Immune aging** (recommended)
- Exhaustion vs. reversible activation
- Drug resistance vs. sensitivity
- Differentiation commitment

### 3. Overlay with Chromatin Accessibility (Optional)

Use ATAC-seq data to validate:
- Are unstable modes accessible?
- Are stable modes epigenetically reinforced?
- Do resistant states show chromatin locking?

## Visualization

### Key Plots

1. **Operator Regime UMAP**
   - Color cells by operator regime (stable/plastic/unstable/deeply_stable)
   - Show overlap with RNA-defined clusters

2. **Stability Depth Map**
   - Continuous heatmap of λ_min⁻ on UMAP
   - Reveals commitment gradients

3. **Plasticity Map**
   - Continuous heatmap of plasticity index P
   - Highlights decision points

4. **Non-Redundancy Figure** (Critical for publication)
   - Same cell type, different conditions
   - Show operator regime differences despite similar expression

5. **Temporal Evolution**
   - Operator metrics along pseudotime
   - Identify bifurcation points and commitment transitions

## Language Discipline

### Always Say:
- "operator regime"
- "stability structure"
- "maintenance"
- "scqdiff application"
- "operator-based state definition"

### Never Say:
- "new cell types"
- "redefine cell types"
- "replacement for cell atlases"
- "separate tool"

## Paper Positioning

### For Nature Computational Science:
> "We use scqdiff to construct a Stable Operator Atlas that defines cellular states by their local stability structure, revealing a dynamical layer of cell identity that is invisible to expression-based atlases."

### For OT Grants:
> "The Stable Operator Atlas provides a reusable, population-scale framework for quantifying robustness, plasticity, and control across heterogeneous immune systems."

## Implementation Phases

### Phase 1: Core Metrics (Week 1)
- Implement `OperatorMetrics` class
- Extend `DriftField` with Jacobian eigenvalue computation
- Unit tests for metric computation

### Phase 2: Regime Classification (Week 1)
- Implement `OperatorRegimeClassifier`
- Define thresholds for regime boundaries
- Validate on synthetic data

### Phase 3: Atlas Builder (Week 2)
- Implement `StableOperatorAtlas` class
- Integration with AnnData
- Command-line interface

### Phase 4: Visualization (Week 2)
- Operator regime plots
- Stability/plasticity maps
- Temporal evolution plots

### Phase 5: Validation (Week 3)
- Non-redundancy tests
- Biological anchoring examples
- Documentation and tutorials

## Dependencies

### Required:
- `torch` (for autograd Jacobian computation)
- `numpy`
- `scanpy` (for AnnData integration)
- `matplotlib`, `seaborn` (for visualization)

### Optional:
- `cellrank` (for trajectory analysis integration)
- `episcanpy` (for ATAC-seq overlay)


## Future Extensions

- **Multi-condition atlases**: Compare operator regimes across multiple experimental conditions
- **Perturbation prediction**: Use operator regimes to predict response to perturbations
- **Drug target identification**: Find interventions that shift operator regimes
- **Integration with CellRank**: Combine operator regimes with fate probabilities
