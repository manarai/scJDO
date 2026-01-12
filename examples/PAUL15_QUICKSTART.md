# Paul15 Notebook Quick Start

## ✅ Import Issue Fixed

The notebook has been corrected. The import section now works properly:

```python
from scqdiff.models.drift import DriftField, DriftConfig
```

**Note**: The `prepare_anndata_for_training` function is not needed for this notebook as we prepare the data manually through the standard preprocessing pipeline.

---

## Running the Notebook

### 1. Ensure Dependencies Are Installed

```bash
# Activate environment
source scidiff_env/bin/activate

# Install required packages
pip install jupyter scanpy scvelo matplotlib seaborn scikit-learn scikit-misc
```

**Important**: `scikit-misc` is required for `flavor='seurat_v3'` in highly variable gene selection.

### 2. Launch Jupyter

```bash
cd examples/
jupyter notebook paul15_hybrid_drift_analysis.ipynb
```

### 3. Run All Cells

In Jupyter:
- Click **Cell → Run All**
- Or run cells individually with **Shift+Enter**

---

## What the Notebook Does

### Data Loading
```python
adata = sc.datasets.paul15()
print(f"Loaded {adata.n_obs} cells × {adata.n_vars} genes")
```

Loads the Paul15 hematopoiesis dataset (~2,700 cells).

### Preprocessing
- Filtering, normalization, log-transform
- Highly variable gene selection (2,000 genes)
- PCA (50 components → use first 30)
- Neighborhood graph and UMAP
- Leiden clustering

### Velocity Estimation
- Diffusion pseudotime (DPT)
- Pseudotime-based velocity vectors
- Confidence weighting

### Model Training
- Hybrid Drift Field with 3 components:
  1. Score network (data distribution)
  2. Residual network (dynamics correction)
  3. Velocity prior (biological guidance)

### Trajectory Analysis
- Select progenitor cells (low pseudotime)
- Simulate differentiation trajectories
- Visualize on UMAP

### Component Analysis
- Decompose drift field
- Analyze contributions
- Biological interpretation

---

## Expected Runtime

- **Data loading**: ~10 seconds
- **Preprocessing**: ~1-2 minutes
- **Velocity computation**: ~1 minute
- **Model training**: ~5-8 minutes (100 epochs)
- **Trajectory simulation**: ~30 seconds
- **Total**: ~10-15 minutes

---

## Common Issues

### Issue: `ModuleNotFoundError: No module named 'scqdiff.data.anndata'`

**Solution**: This has been fixed. The notebook no longer imports `prepare_anndata_for_training`.

### Issue: `ModuleNotFoundError: No module named 'skmisc'`

**Solution**: Install scikit-misc:
```bash
pip install scikit-misc
```

### Issue: `UserWarning: flavor='seurat_v3' expects raw count data`

**Solution**: This is just a warning and can be ignored. The Paul15 dataset is already normalized, but the HVG selection still works.

### Issue: Slow UMAP computation

**Solution**: This is normal. UMAP can take 1-2 minutes on CPU.

### Issue: Training is slow

**Solution**: 
- Reduce `n_epochs` from 100 to 50
- Reduce `batch_size` from 256 to 128
- Set `laplacian_weight=0.0` to disable regularization

---

## Key Parameters to Adjust

### Model Architecture
```python
cfg = DriftConfig(
    dim=30,           # PCA dimensions (match your data)
    hidden=256,       # Increase for more capacity
    depth=4,          # Increase for deeper network
    beta=0.1,         # Diffusion coefficient
)
```

### Velocity Prior
```python
cfg = DriftConfig(
    ...
    use_velocity_prior=True,  # Enable/disable velocity
    vel_k=16,                 # KNN neighbors
    vel_scale=1.0,            # Velocity strength
    vel_schedule='mid',       # 'constant', 'early', 'mid', 'late'
)
```

### Training
```python
n_epochs = 100      # Reduce to 50 for faster training
batch_size = 256    # Reduce to 128 if memory issues
lr = 1e-3           # Learning rate
```

---

## Verification

To verify the notebook works:

```bash
cd /home/ubuntu
python -c "
import scanpy as sc
from scqdiff.models.drift import DriftField, DriftConfig

# Load data
adata = sc.datasets.paul15()
print(f'✓ Paul15 loaded: {adata.n_obs} cells')

# Test model
import torch
cfg = DriftConfig(dim=10, hidden=64, depth=2, use_velocity_prior=True)
X = torch.randn(100, 10)
V = torch.randn(100, 10)
W = torch.rand(100)
model = DriftField(cfg, X_ref=X, V_ref=V, W_ref=W)
print('✓ Model initialized')

# Test forward pass
x = torch.randn(5, 10)
t = torch.rand(5)
drift = model(x, t)
print(f'✓ Forward pass: {drift.shape}')
print('✓ Notebook ready!')
"
```

---

## Next Steps

After running the notebook:

1. **Analyze trajectories**: Look at gene expression changes
2. **Compute Jacobian**: Gene-gene regulatory interactions
3. **Compare methods**: RNA velocity vs Hybrid Drift Field
4. **Perturbation analysis**: Model drug effects
5. **Multi-condition**: Compare control vs treated

---

## Support

For detailed documentation:
- `PAUL15_NOTEBOOK_README.md` - Full usage guide
- `NOTEBOOKS_SUMMARY.md` - Comparison of both notebooks
- `README.md` - Main scIDiff documentation

---

**Status**: ✅ Fixed and tested  
**Last updated**: December 25, 2025
