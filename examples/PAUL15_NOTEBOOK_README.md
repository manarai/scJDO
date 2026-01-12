# Paul15 Hematopoiesis Analysis with Hybrid Drift Field

## Overview

This Jupyter notebook demonstrates the **Hybrid Drift Field** (the default mode in scIDiff) for analyzing real single-cell RNA-seq data. It uses the Paul et al. (2015) hematopoiesis dataset to model myeloid differentiation trajectories.

## Dataset

**Paul15** contains:
- **~2,700 cells** from mouse bone marrow
- **~3,900 genes** measured via single-cell RNA-seq
- **Cell types**: Multipotent progenitors (MPP), erythroid, and myeloid lineages
- **Biological process**: Hematopoietic differentiation

## Hybrid Drift Field

The model combines three components:

```
f(x,t) = β·s_θ(x,t) + r_φ(x,t) + b(x,t)
```

where:
- **s_θ(x,t)**: Score network (denoising score matching)
- **r_φ(x,t)**: Residual correction (Neural ODE)
- **b(x,t)**: Velocity prior (RNA velocity guidance)

This hybrid approach:
1. ✅ Learns data-driven dynamics via score matching
2. ✅ Corrects for model misspecification via residual network
3. ✅ Incorporates biological priors via RNA velocity

---

## Notebook Contents

### **Part 1: Data Loading & Preprocessing (Sections 1-4)**
- Load Paul15 dataset from scanpy
- Standard preprocessing pipeline
- Highly variable gene selection
- PCA, neighborhood graph, UMAP
- Leiden clustering

### **Part 2: RNA Velocity Analysis (Sections 5-7)**
- Diffusion pseudotime computation
- Pseudotime-based velocity estimation
- Velocity field visualization on UMAP

### **Part 3: Model Training (Sections 8-12)**
- Data preparation (PCA + velocity)
- Hybrid Drift Field configuration
- Model initialization with velocity prior
- Training with score matching
- Training progress visualization

### **Part 4: Trajectory Analysis (Sections 13-14)**
- Select progenitor cells (low pseudotime)
- Simulate differentiation trajectories
- Project trajectories to UMAP
- Visualize paths on cell landscape

### **Part 5: Component Analysis (Sections 15-16)**
- Decompose drift into 3 components
- Analyze relative contributions
- Biological interpretation
- Summary and applications

---

## Running the Notebook

### Prerequisites

```bash
# Activate virtual environment
source scidiff_env/bin/activate

# Install required packages
pip install jupyter scanpy scvelo matplotlib seaborn scikit-learn

# Ensure scqdiff is installed
pip install -e .
```

### Launch Jupyter

```bash
cd examples/
jupyter notebook paul15_hybrid_drift_analysis.ipynb
```

Or use JupyterLab:

```bash
jupyter lab paul15_hybrid_drift_analysis.ipynb
```

### Run All Cells

In Jupyter:
- Click **Cell → Run All** to execute the entire notebook
- Or run cells individually with **Shift+Enter**

---

## Expected Runtime

- **Total runtime**: ~10-15 minutes (CPU)
- **Data loading & preprocessing**: ~2-3 minutes
- **Velocity computation**: ~1-2 minutes
- **Model training**: ~5-8 minutes (100 epochs)
- **Trajectory simulation**: ~1-2 minutes

---

## Key Outputs

### Visualizations

1. **UMAP plots**: Cell types and clusters
2. **Pseudotime**: Diffusion pseudotime on UMAP
3. **Velocity field**: Arrows showing differentiation direction
4. **Training curves**: Loss convergence over epochs
5. **Trajectories**: Differentiation paths from progenitors
6. **Component analysis**: Drift field decomposition

### Quantitative Results

- Training loss convergence
- Drift component magnitudes
- Trajectory displacements
- Relative contributions of each component

---

## Customization

### Data Parameters

```python
# Highly variable genes
sc.pp.highly_variable_genes(adata, n_top_genes=2000)

# PCA dimensions
sc.tl.pca(adata, n_comps=50)

# Neighborhood size
sc.pp.neighbors(adata, n_neighbors=30, n_pcs=30)
```

### Model Configuration

```python
cfg = DriftConfig(
    dim=30,                    # PCA dimensions
    hidden=256,                # Hidden layer size
    depth=4,                   # Network depth
    beta=0.1,                  # Diffusion coefficient
    
    # Velocity prior
    use_velocity_prior=True,   # Enable/disable velocity
    vel_k=16,                  # KNN for velocity interpolation
    vel_tau=1.0,               # Temperature for softmax
    vel_scale=1.0,             # Velocity scaling
    vel_schedule='mid',        # 'constant', 'early', 'mid', 'late'
    
    # Regularization
    laplacian_weight=0.01,     # Smoothness penalty
)
```

### Training Parameters

```python
n_epochs = 100        # Training iterations
batch_size = 256      # Batch size
lr = 1e-3             # Learning rate
```

### Trajectory Simulation

```python
n_trajectories = 10   # Number of paths to simulate
n_steps = 100         # Integration steps
dt = 0.01             # Time step size
```

---

## Troubleshooting

### Memory Issues

If you encounter memory errors:

```python
# Reduce data size
sc.pp.highly_variable_genes(adata, n_top_genes=1000)

# Reduce model size
cfg = DriftConfig(dim=20, hidden=128, depth=2)

# Reduce batch size
batch_size = 128
```

### Slow Training

To speed up training:

```python
# Reduce epochs
n_epochs = 50

# Disable Laplacian regularization
cfg = DriftConfig(..., laplacian_weight=0.0)

# Use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg = DriftConfig(..., device=device)
```

### Velocity Computation Issues

If pseudotime-based velocity fails:

```python
# Adjust root cell for DPT
adata.uns['iroot'] = np.flatnonzero(adata.obs['leiden'] == '0')[0]

# Or use a different cluster as root
adata.uns['iroot'] = np.flatnonzero(adata.obs['leiden'] == '2')[0]
```

### Trajectory Visualization Issues

If trajectories don't project well to UMAP:

```python
# Increase number of neighbors for projection
neighbors = np.argsort(distances)[:20]  # instead of 10

# Or use PCA space directly for visualization
# (skip UMAP projection)
```

---

## Biological Interpretation

### What the Model Learns

1. **Score Network**: Captures the overall manifold structure and density of cell states
2. **Residual Network**: Learns cell-type specific dynamics and transition rates
3. **Velocity Prior**: Incorporates biological direction from RNA velocity/pseudotime

### Trajectory Interpretation

- **Starting points** (blue circles): Progenitor cells with low pseudotime
- **Endpoints** (red squares): Differentiated cells with high pseudotime
- **Paths** (red/black lines): Most probable differentiation routes

### Component Contributions

Typical relative contributions:
- Score network: ~40-50% (manifold structure)
- Residual network: ~30-40% (dynamics correction)
- Velocity prior: ~10-20% (biological guidance)

---

## Comparison with Other Methods

### vs. RNA Velocity (scVelo)

| Feature | scVelo | Hybrid Drift Field |
|---------|--------|-------------------|
| Input | Spliced/unspliced | Any representation |
| Model | Linear dynamics | Neural network |
| Trajectories | Streamlines | Integrated paths |
| Perturbation | ❌ | ✅ |
| Control | ❌ | ✅ |

### vs. Optimal Transport (Waddington-OT)

| Feature | Waddington-OT | Hybrid Drift Field |
|---------|---------------|-------------------|
| Framework | Optimal transport | Score matching + ODE |
| Time | Requires timepoints | Single snapshot |
| Velocity | ❌ | ✅ |
| Continuous | ❌ | ✅ |

### vs. Pseudotime (Monocle, PAGA)

| Feature | Pseudotime | Hybrid Drift Field |
|---------|------------|-------------------|
| Output | 1D ordering | Full vector field |
| Dynamics | ❌ | ✅ |
| Simulation | ❌ | ✅ |
| Perturbation | ❌ | ✅ |

---

## Next Steps

After completing this notebook:

1. **Gene Analysis**: Track gene expression changes along trajectories
2. **Jacobian Analysis**: Compute gene-gene regulatory interactions
3. **Perturbation Prediction**: Model effects of gene knockouts
4. **Cell Fate Control**: Design optimal interventions
5. **Multi-condition**: Compare control vs treated samples

---

## Related Notebooks

- `schrodinger_bridge_synthetic_analysis.ipynb` - Schrödinger Bridge mode
- `01_synthetic_no_velocity.ipynb` - Drift field without velocity
- `02_synthetic_with_velocity.ipynb` - Drift field with velocity
- `tutorial_paul15_hematopoiesis.ipynb` - Alternative Paul15 analysis

---

## References

### Dataset
- Paul et al. (2015). "Transcriptional Heterogeneity and Lineage Commitment in Myeloid Progenitors." *Cell*.

### Methods
- **Score Matching**: Song & Ermon (2019), "Generative Modeling by Estimating Gradients of the Data Distribution"
- **RNA Velocity**: La Manno et al. (2018), Bergen et al. (2020)
- **Neural ODEs**: Chen et al. (2018), "Neural Ordinary Differential Equations"

### scIDiff
- Main documentation: `README.md`
- Mathematical overview: `math_overview.md`
- Implementation: `scqdiff/models/drift.py`

---

## Support

For questions or issues:
- Check `TEST_RESULTS.md` for verification
- Review `TESTING_GUIDE.md` for troubleshooting
- See main `README.md` for detailed documentation

---

**Created**: December 25, 2025  
**Version**: scIDiff V2  
**Status**: Ready to use ✅
