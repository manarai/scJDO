# scIDiff Jupyter Notebooks Summary

This directory contains comprehensive Jupyter notebooks for analyzing single-cell data with scIDiff.

---

## Available Notebooks

### 1. **Schrödinger Bridge: Synthetic Aging Analysis**
**File**: `schrodinger_bridge_synthetic_analysis.ipynb`

**Mode**: Schrödinger Bridge (optional mode)

**Dataset**: Synthetic aging data (young vs old cells)

**Key Features**:
- Optimal transport between distributions
- Forward process (aging: young → old)
- Backward process (rejuvenation: old → young)
- Sinkhorn algorithm for OT plan
- Score matching for drift learning
- Stochastic and deterministic integration
- Round-trip consistency analysis

**Use Cases**:
- Aging studies
- Perturbation response modeling
- Cell state transitions
- Reprogramming design

**Runtime**: ~5-10 minutes

**Documentation**: `NOTEBOOK_README.md`

---

### 2. **Paul15 Hematopoiesis with Hybrid Drift Field**
**File**: `paul15_hybrid_drift_analysis.ipynb`

**Mode**: Hybrid Drift Field (default mode)

**Dataset**: Paul15 hematopoiesis (~2,700 cells)

**Key Features**:
- Real single-cell RNA-seq data
- Standard preprocessing pipeline
- Pseudotime-based velocity estimation
- Hybrid drift field (score + residual + velocity)
- Trajectory simulation from progenitors
- Component decomposition analysis
- UMAP visualization

**Use Cases**:
- Differentiation analysis
- Trajectory inference
- Cell fate prediction
- Gene regulatory analysis

**Runtime**: ~10-15 minutes

**Documentation**: `PAUL15_NOTEBOOK_README.md`

---

## Quick Start

### Setup Environment

```bash
# Activate virtual environment
source scidiff_env/bin/activate

# Install Jupyter and dependencies
pip install jupyter matplotlib seaborn scanpy scvelo scikit-learn

# Ensure scqdiff is installed
cd /home/ubuntu
pip install -e .
```

### Launch Notebooks

```bash
cd examples/

# Option 1: Jupyter Notebook
jupyter notebook

# Option 2: JupyterLab
jupyter lab
```

Then open either:
- `schrodinger_bridge_synthetic_analysis.ipynb`
- `paul15_hybrid_drift_analysis.ipynb`

---

## Comparison: Schrödinger Bridge vs Hybrid Drift Field

| Feature | Schrödinger Bridge | Hybrid Drift Field |
|---------|-------------------|-------------------|
| **Framework** | Optimal transport | Score matching + ODE |
| **Input** | Two distributions | Single dataset |
| **Velocity Prior** | ❌ Optional | ✅ Integrated |
| **Directionality** | Bidirectional (forward/backward) | Forward with velocity |
| **Training** | Alternating OT + score matching | Score matching + residual |
| **Use Case** | Condition A → B transitions | Trajectory inference |
| **Biological** | Aging, perturbation, reprogramming | Differentiation, development |
| **Complexity** | Higher (OT + 2 networks) | Moderate (1 hybrid network) |

---

## When to Use Which

### Use **Schrödinger Bridge** when:
- ✅ You have two distinct conditions (e.g., control vs treated)
- ✅ You want optimal transport between distributions
- ✅ You need bidirectional trajectories (forward and backward)
- ✅ You're studying aging, perturbation response, or reprogramming
- ✅ You want to model the most probable transition path

**Examples**:
- Young cells → Old cells (aging)
- Control → Drug-treated (perturbation)
- Differentiated → Stem cells (reprogramming)

### Use **Hybrid Drift Field** when:
- ✅ You have a single dataset with developmental/differentiation process
- ✅ You have RNA velocity or pseudotime information
- ✅ You want to incorporate biological priors
- ✅ You're studying differentiation or development
- ✅ You want to simulate trajectories from any starting point

**Examples**:
- Hematopoiesis (progenitor → mature cells)
- Neurogenesis (neural stem → neurons)
- Embryogenesis (early → late stages)

---

## Notebook Structure

Both notebooks follow a similar structure:

### Part 1: Data & Setup
- Import libraries
- Load/generate data
- Preprocessing (if applicable)
- Visualization

### Part 2: Model Configuration
- Configure model parameters
- Initialize networks
- Set up training

### Part 3: Training
- Training loop
- Loss computation
- Progress visualization

### Part 4: Analysis
- Trajectory simulation
- Component analysis
- Quantitative metrics

### Part 5: Interpretation
- Biological interpretation
- Summary of results
- Next steps

---

## Output Files

Both notebooks generate:

### Visualizations
- Distribution plots (PCA/UMAP)
- Training curves
- Trajectory paths
- Component analysis

### Metrics
- Training loss
- Transport quality (Schrödinger Bridge)
- Component magnitudes (Hybrid Drift Field)
- Trajectory displacements

### Models
- Trained networks (in memory)
- Can be saved with `torch.save()`

---

## Customization Guide

### Data Parameters

**Schrödinger Bridge**:
```python
n_young = 500        # Source distribution size
n_old = 500          # Target distribution size
dim = 20             # Dimensions
```

**Hybrid Drift Field**:
```python
n_top_genes = 2000   # Highly variable genes
n_pcs = 30           # PCA components
n_neighbors = 30     # KNN graph
```

### Model Architecture

**Schrödinger Bridge**:
```python
cfg = SchrodingerBridgeConfig(
    dim=20,
    hidden=256,
    depth=4,
    beta=0.1,
    sigma=0.2,
    epsilon=0.1
)
```

**Hybrid Drift Field**:
```python
cfg = DriftConfig(
    dim=30,
    hidden=256,
    depth=4,
    beta=0.1,
    use_velocity_prior=True,
    vel_k=16,
    vel_scale=1.0
)
```

### Training

Both support:
```python
n_epochs = 100       # Training iterations
batch_size = 256     # Batch size
lr = 1e-3            # Learning rate
```

### Trajectory Simulation

Both support:
```python
n_trajectories = 10  # Number of paths
n_steps = 100        # Integration steps
stochastic = False   # Deterministic vs stochastic
```

---

## Troubleshooting

### Common Issues

**Memory errors**:
- Reduce `batch_size`
- Reduce `hidden` size
- Reduce `n_top_genes` (Hybrid Drift Field)

**Slow training**:
- Reduce `n_epochs`
- Disable `laplacian_weight` (Hybrid Drift Field)
- Use GPU if available

**Poor trajectories**:
- Increase `n_epochs`
- Adjust `vel_scale` (Hybrid Drift Field)
- Adjust `epsilon` (Schrödinger Bridge)

**Visualization issues**:
- Check UMAP computation
- Verify PCA projection
- Increase number of neighbors for projection

---

## Advanced Usage

### Save Trained Models

```python
# Schrödinger Bridge
torch.save({
    'forward_net': bridge.forward_net.state_dict(),
    'backward_net': bridge.backward_net.state_dict(),
    'config': cfg
}, 'schrodinger_bridge.pt')

# Hybrid Drift Field
torch.save({
    'model': model.state_dict(),
    'config': cfg
}, 'hybrid_drift_field.pt')
```

### Load Trained Models

```python
# Schrödinger Bridge
checkpoint = torch.load('schrodinger_bridge.pt')
bridge = SchrodingerBridge(checkpoint['config'], X_0, X_1)
bridge.forward_net.load_state_dict(checkpoint['forward_net'])
bridge.backward_net.load_state_dict(checkpoint['backward_net'])

# Hybrid Drift Field
checkpoint = torch.load('hybrid_drift_field.pt')
model = DriftField(checkpoint['config'], X_ref, V_ref, W_ref)
model.load_state_dict(checkpoint['model'])
```

### Export Trajectories

```python
# Save trajectories as CSV
import pandas as pd

# Convert to DataFrame
traj_df = pd.DataFrame(
    trajectories.reshape(-1, dim).numpy(),
    columns=[f'PC{i+1}' for i in range(dim)]
)
traj_df['trajectory_id'] = np.repeat(range(n_trajectories), n_steps + 1)
traj_df['time_step'] = np.tile(range(n_steps + 1), n_trajectories)

# Save
traj_df.to_csv('trajectories.csv', index=False)
```

---

## Related Examples

### Other Tutorial Notebooks

- `00_anndata_quickstart.ipynb` - AnnData basics
- `01_synthetic_no_velocity.ipynb` - Drift field without velocity
- `02_synthetic_with_velocity.ipynb` - Drift field with velocity
- `03_archetypes_factorization.ipynb` - Archetype analysis
- `04_fate_conditioned_archetypes.ipynb` - Conditional archetypes
- `05_simulate_trajectories.ipynb` - Trajectory simulation
- `tutorial_synthetic_data.ipynb` - Synthetic data tutorial
- `tutorial_paul15_hematopoiesis.ipynb` - Alternative Paul15 analysis

### Python Scripts

- `train_aging_bridge.py` - Schrödinger Bridge training script
- `train_with_velocity.py` - Hybrid Drift Field training script
- `plot_archetypes.py` - Archetype visualization

---

## References

### Theory

**Schrödinger Bridge**:
- Léonard (2014). "A survey of the Schrödinger problem and some of its connections with optimal transport"
- Chen et al. (2021). "Likelihood Training of Schrödinger Bridge using Forward-Backward SDEs Theory"

**Hybrid Drift Field**:
- Song & Ermon (2019). "Generative Modeling by Estimating Gradients of the Data Distribution"
- Chen et al. (2018). "Neural Ordinary Differential Equations"
- La Manno et al. (2018). "RNA velocity of single cells"

### Implementation

- Main documentation: `/home/ubuntu/README.md`
- Mathematical overview: `/home/ubuntu/math_overview.md`
- Testing guide: `/home/ubuntu/TESTING_GUIDE.md`
- Test results: `/home/ubuntu/TEST_RESULTS.md`

---

## Support

For questions or issues:
1. Check the individual notebook READMEs
2. Review the main documentation
3. Check test results for verification
4. See troubleshooting sections above

---

## Citation

If you use these notebooks in your research, please cite:

```bibtex
@software{scidiff2025,
  title={scIDiff: Single-Cell Inference with Diffusion Models},
  author={[Authors]},
  year={2025},
  url={https://github.com/[repo]/scidiff}
}
```

---

**Created**: December 25, 2025  
**Version**: scIDiff V2  
**Status**: Production ready ✅

Both notebooks are fully tested and ready for biological analysis!
