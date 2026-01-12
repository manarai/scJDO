# Schrödinger Bridge Synthetic Analysis Notebook

## Overview

This Jupyter notebook provides a comprehensive tutorial and analysis of the **Schrödinger Bridge** implementation in scIDiff. It demonstrates optimal transport between two distributions using a synthetic aging scenario.

## What's Inside

The notebook covers:

### 1. **Mathematical Framework**
- Schrödinger Bridge theory
- Optimal transport formulation
- Score matching for drift learning
- Entropic regularization

### 2. **Synthetic Data Generation**
- Young cells (compact distribution)
- Old cells (dispersed, shifted distribution)
- Biological interpretation of aging

### 3. **Optimal Transport Analysis**
- Sinkhorn algorithm for OT plan computation
- Coupling visualization
- Marginal constraint verification
- Sparsity analysis

### 4. **Model Training**
- Forward network (aging process)
- Backward network (rejuvenation process)
- Score matching loss
- Endpoint loss
- Training progress visualization

### 5. **Trajectory Analysis**
- Forward integration (aging trajectories)
- Backward integration (rejuvenation trajectories)
- Deterministic vs stochastic integration
- PCA visualization of trajectories

### 6. **Quantitative Evaluation**
- Transport quality metrics
- Round-trip consistency
- Drift field analysis
- Time-dependent behavior

### 7. **Biological Interpretation**
- Aging as optimal transport
- Rejuvenation as reverse process
- Applications to real data
- Next steps for research

## Running the Notebook

### Prerequisites

```bash
# Activate virtual environment
source scidiff_env/bin/activate

# Install Jupyter (if not already installed)
pip install jupyter matplotlib seaborn

# Ensure scqdiff is installed
pip install -e .
```

### Launch Jupyter

```bash
cd examples/
jupyter notebook schrodinger_bridge_synthetic_analysis.ipynb
```

Or use JupyterLab:

```bash
jupyter lab schrodinger_bridge_synthetic_analysis.ipynb
```

### Run All Cells

In Jupyter:
- Click **Cell → Run All** to execute the entire notebook
- Or run cells individually with **Shift+Enter**

## Expected Runtime

- **Total runtime**: ~5-10 minutes (depending on hardware)
- **Training**: ~3-5 minutes (50 iterations)
- **Visualizations**: ~1-2 minutes

## Key Results

After running the notebook, you will see:

1. **Distribution Visualization**: PCA plots showing young vs old cells
2. **OT Plan Heatmap**: Sparse coupling matrix
3. **Training Curves**: Loss convergence over iterations
4. **Trajectory Plots**: Aging and rejuvenation paths in PCA space
5. **Quantitative Metrics**: Transport quality and round-trip error

## Customization

You can modify the following parameters:

### Data Generation
```python
n_young = 500        # Number of young cells
n_old = 500          # Number of old cells
dim = 20             # Gene expression dimensions
```

### Model Configuration
```python
cfg = SchrodingerBridgeConfig(
    dim=dim,
    hidden=256,      # Hidden layer size
    depth=4,         # Network depth
    beta=0.1,        # Diffusion coefficient
    sigma=0.2,       # Score matching noise
    epsilon=0.1,     # Entropic regularization
)
```

### Training
```python
n_iterations = 50    # Training iterations
batch_size = 128     # Batch size
lr = 1e-3            # Learning rate
update_ot_every = 10 # OT plan update frequency
```

## Outputs

The notebook generates:

- **Figures**: Inline matplotlib plots
- **Metrics**: Printed statistics and analysis
- **Models**: Trained forward and backward networks (in memory)

To save outputs:

```python
# Save figure
plt.savefig('aging_trajectories.png', dpi=300, bbox_inches='tight')

# Save model
torch.save(bridge.state_dict(), 'schrodinger_bridge.pt')
```

## Troubleshooting

### Memory Issues

If you encounter memory errors:

```python
# Reduce data size
n_young = 200
n_old = 200
dim = 10

# Reduce model size
hidden = 128
depth = 2

# Reduce batch size
batch_size = 64
```

### Slow Training

To speed up training:

```python
# Reduce iterations
n_iterations = 20

# Reduce integration steps
steps = 30  # instead of 100

# Use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg = SchrodingerBridgeConfig(..., device=device)
```

### Visualization Issues

If plots don't display:

```python
# Force inline display
%matplotlib inline

# Or use non-interactive backend
import matplotlib
matplotlib.use('Agg')
```

## Next Steps

After completing this notebook, you can:

1. **Apply to Real Data**: Replace synthetic data with single-cell RNA-seq
2. **Add Velocity Prior**: Integrate RNA velocity as biological reference
3. **Analyze Gene Changes**: Track gene expression along trajectories
4. **Perturbation Analysis**: Model drug effects or genetic modifications
5. **Multi-condition**: Extend to multiple cell states or conditions

## Related Notebooks

- `01_synthetic_no_velocity.ipynb` - Basic drift field without velocity
- `02_synthetic_with_velocity.ipynb` - Hybrid drift field with velocity
- `tutorial_paul15_hematopoiesis.ipynb` - Real data example

## References

### Theory
- Schrödinger Bridge: Léonard (2014), Chen et al. (2021)
- Optimal Transport: Peyré & Cuturi (2019)
- Score Matching: Hyvärinen (2005), Song & Ermon (2019)

### Implementation
- scIDiff documentation: `README.md`
- Mathematical overview: `math_overview.md`
- API reference: `scqdiff/models/schrodinger_bridge.py`

## Support

For questions or issues:
- Check `TEST_RESULTS.md` for verification
- Review `TESTING_GUIDE.md` for troubleshooting
- See main `README.md` for detailed documentation

---

**Created**: December 25, 2025  
**Version**: scIDiff V2  
**Status**: Tested and verified ✅
