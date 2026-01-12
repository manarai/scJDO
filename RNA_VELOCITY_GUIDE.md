# RNA Velocity Integration in scIDiff

## Overview

This implementation integrates **RNA velocity as a biological prior** into the scIDiff framework. RNA velocity provides a local directional field derived from splicing kinetics, offering a window into the cell's immediate transcriptional future. By incorporating this information, scIDiff learns trajectories that balance global optimality (Schrödinger Bridge) with local biological realism (RNA velocity).

## Biological Motivation

The Schrödinger Bridge finds the most probable stochastic path between two cell populations, minimizing an energy functional. However, this path represents a **mathematical optimum**—the straightest line through a high-dimensional landscape. Biological processes rarely follow such idealized paths; they are constrained by the intricate machinery of gene regulation.

**RNA velocity acts as a biological compass**, providing direct readouts of transcriptional dynamics through the ratio of unspliced to spliced mRNA. By using velocity as a **reference drift**, we ensure the learned trajectories are not just mathematically probable but also **mechanistically plausible**.

This is particularly valuable for:
- **Drug response studies**: Detecting altered splicing patterns that indicate network rewiring
- **Differentiation modeling**: Ensuring trajectories follow biologically feasible paths
- **Perturbation analysis**: Identifying how interventions alter transcriptional dynamics

## Mathematical Framework

The drift field is computed as:

```
f(x,t) = b(x,t) + u_θ(x,t)
```

where:
- **b(x,t)** is the velocity prior (reference drift)
- **u_θ(x,t)** is the learned correction (score + residual)

The velocity prior is defined as:

```
b(x,t) = vel_scale × g(t) × gate(x) × v(x)
```

where:
- **v(x)**: Interpolated velocity from reference data using soft k-nearest neighbors
- **gate(x)**: Confidence-based gating to downweight unreliable velocities
- **g(t)**: Time schedule controlling when velocity guidance is strongest
- **vel_scale**: Global scaling factor for velocity magnitude

### Time Schedule g(t)

Two modes are available:

**"mid" mode** (default): `g(t) = 4t(1-t)`
- Peaks at t=0.5, zero at endpoints
- Emphasizes velocity in the middle of trajectories
- Allows model freedom at boundaries to match endpoint distributions

**"flat" mode**: `g(t) = 1`
- Constant velocity contribution throughout
- Use when velocity is reliable across all time points

### Confidence Gating

RNA velocity estimates vary in reliability. The confidence gate:

```
gate(x) = conf(x)^vel_conf_power
```

downweights unreliable estimates. Higher `vel_conf_power` values create stronger gating.

## Implementation Details

### KNN Velocity Interpolator

The `KNNVelocity` module creates a smooth velocity field from discrete reference points:

1. For query point **x**, find k nearest reference cells
2. Compute distance-based softmax weights: `w = softmax(-d/τ)`
3. Interpolate velocity: `v(x) = Σ w_i × v_i`
4. Interpolate confidence: `conf(x) = Σ w_i × conf_i`

The temperature parameter **τ** controls smoothness:
- Lower τ → sharper transitions (more local)
- Higher τ → smoother field (more global)

### Architecture Changes

**DriftConfig** now includes:
```python
use_velocity_prior: bool = False    # Enable velocity integration
vel_k: int = 32                     # Neighbors for KNN
vel_tau: float = 1.0                # KNN temperature
vel_scale: float = 1.0              # Velocity magnitude scaling
vel_conf_power: float = 1.0         # Confidence gating strength
vel_time_mode: str = "mid"          # Time schedule type
```

**DriftField** constructor accepts:
```python
DriftField(cfg, laplacian=None, X_ref=None, V_ref=None, W_ref=None)
```

The forward signature `model(x,t) -> drift` remains unchanged for backward compatibility.

## Usage

### Basic Training with Velocity

```bash
python -m scqdiff.pipeline.train_from_anndata \
    --h5ad data.h5ad \
    --use-velocity-prior \
    --vel-layer velocity \
    --ptime-key latent_time \
    --epochs 200
```

### Advanced Configuration

```bash
python -m scqdiff.pipeline.train_from_anndata \
    --h5ad data.h5ad \
    --use-velocity-prior \
    --normalize-velocity \
    --vel-k 32 \
    --vel-tau 1.0 \
    --vel-scale 0.5 \
    --vel-conf-power 1.5 \
    --vel-time-mode mid \
    --epochs 200
```

### Python API

```python
import torch
import anndata as ad
from scqdiff.io.anndata import tensors_from_anndata
from scqdiff.models.drift import DriftField, DriftConfig

# Load data
adata = ad.read_h5ad("data.h5ad")
X, V, T = tensors_from_anndata(
    adata,
    vel_layer="velocity",
    pseudotime_key="latent_time"
)

# Optionally normalize velocity
if V is not None:
    V = V / (V.norm(dim=1, keepdim=True) + 1e-8)

# Create configuration
cfg = DriftConfig(
    dim=X.shape[1],
    beta=0.1,
    use_velocity_prior=True,
    vel_k=32,
    vel_tau=1.0,
    vel_scale=1.0,
    vel_conf_power=1.0,
    vel_time_mode="mid"
)

# Instantiate model
model = DriftField(cfg, X_ref=X, V_ref=V)

# Use model
drift = model(X, T)  # Compute drift at all cells
```

## Hyperparameter Tuning

### vel_scale

Controls the strength of velocity guidance:
- **Start with 1.0** if using normalized velocities
- **Start with 0.1-1.0** if using raw velocities
- Increase if trajectories deviate too much from expected biology
- Decrease if model struggles to match endpoint distributions

### vel_k

Number of neighbors for interpolation:
- **Default: 32** works well for most datasets
- Increase for smoother velocity fields (more global)
- Decrease for sharper transitions (more local)
- Should be smaller than typical cluster sizes

### vel_tau

Temperature for softmax weighting:
- **Default: 1.0** provides balanced smoothness
- Increase for smoother interpolation
- Decrease for more localized influence

### vel_conf_power

Confidence gating strength:
- **Default: 1.0** uses confidence linearly
- Increase (e.g., 1.5-2.0) to more strongly suppress unreliable velocities
- Set to 0.0 to disable confidence gating

### vel_time_mode

Time schedule for velocity contribution:
- **"mid"**: Emphasize velocity in middle of trajectories (recommended)
- **"flat"**: Constant velocity contribution throughout

## Velocity Scaling Strategies

### Option 1: Normalization (Recommended)

Normalize velocity vectors to unit length before training:

```bash
python -m scqdiff.pipeline.train_from_anndata \
    --h5ad data.h5ad \
    --use-velocity-prior \
    --normalize-velocity \
    --vel-scale 1.0
```

**Pros:**
- Velocity provides pure directional guidance
- Magnitude doesn't bias the model
- More stable across datasets

**Cons:**
- Loses information about velocity magnitude
- May need to tune vel_scale

### Option 2: Raw Velocity with Tuning

Keep raw velocities and tune `vel_scale`:

```bash
python -m scqdiff.pipeline.train_from_anndata \
    --h5ad data.h5ad \
    --use-velocity-prior \
    --vel-scale 0.5
```

**Pros:**
- Preserves velocity magnitude information
- Can capture confidence through magnitude

**Cons:**
- Requires dataset-specific tuning
- Magnitude units may not match latent space

## Validation

### Check Velocity Field Quality

```python
import matplotlib.pyplot as plt
from scqdiff.models.drift import KNNVelocity

# Create velocity interpolator
vel_interp = KNNVelocity(X, V, k=32, tau=1.0)

# Evaluate on grid
v_interp, conf_interp = vel_interp(X)

# Compare with reference
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(V[:, 0], V[:, 1], c=conf_interp, alpha=0.5)
plt.title("Reference Velocity")
plt.subplot(1, 2, 2)
plt.scatter(v_interp[:, 0], v_interp[:, 1], c=conf_interp, alpha=0.5)
plt.title("Interpolated Velocity")
plt.show()
```

### Visualize Learned Trajectories

```python
from scqdiff.simulate.integrate import integrate_sde

# Simulate trajectories
t_span = torch.linspace(0, 1, 100)
x0 = X[:10]  # Start from first 10 cells
trajectories = integrate_sde(model, x0, t_span)

# Plot
plt.figure(figsize=(8, 6))
for traj in trajectories:
    plt.plot(traj[:, 0], traj[:, 1], alpha=0.5)
plt.scatter(X[:, 0], X[:, 1], c=T, s=1, alpha=0.3)
plt.title("Learned Trajectories")
plt.show()
```

## Troubleshooting

### Velocity not affecting trajectories

- Increase `vel_scale`
- Check that velocity data is loaded correctly
- Verify `use_velocity_prior=True` in config
- Try `vel_time_mode="flat"` instead of "mid"

### Model not matching endpoints

- Decrease `vel_scale`
- Use `vel_time_mode="mid"` to give model freedom at boundaries
- Increase confidence gating with higher `vel_conf_power`

### Noisy or unstable trajectories

- Increase `vel_k` for smoother interpolation
- Increase `vel_tau` for more global averaging
- Enable velocity normalization with `--normalize-velocity`
- Increase Laplacian smoothing with `--laplacian-lambda`

### Velocity interpolation too slow

- Decrease `vel_k` (fewer neighbors to compute)
- Use GPU acceleration (ensure tensors are on CUDA device)
- Consider downsampling reference cells for very large datasets

## Comparison with Naive Approach

The **previous implementation** used a simple MSE loss:

```python
loss += 0.5 * ((u - V[idx])**2).mean()
```

This approach:
- Forces drift to match velocity exactly everywhere
- Ignores velocity confidence/reliability
- Doesn't account for time-dependent relevance
- Can conflict with endpoint matching

The **new implementation**:
- Uses velocity as a **reference drift**, not a target
- Incorporates confidence gating
- Applies time-dependent scheduling
- Allows learned correction to handle endpoint matching
- Provides smooth interpolation for continuous evaluation

## Biological Interpretation

The velocity-guided drift field represents a **hybrid model**:

**Local dynamics** (RNA velocity): Captures immediate transcriptional changes based on splicing kinetics. This ensures the model respects the cell's current regulatory state and follows mechanistically plausible directions.

**Global optimization** (Schrödinger Bridge): Ensures the overall trajectory efficiently transports probability mass from initial to final distributions while minimizing control energy.

The result is a model that produces trajectories which are:
- **Locally realistic**: Each step follows biologically plausible directions
- **Globally optimal**: The overall path efficiently connects endpoints
- **Mechanistically interpretable**: Deviations from velocity indicate where active control is needed

## References

1. **RNA Velocity**: La Manno et al. (2018) "RNA velocity of single cells" Nature
2. **Schrödinger Bridges**: Chen et al. (2021) "Likelihood Training of Schrödinger Bridge using Forward-Backward SDEs Theory"
3. **Optimal Transport**: Cuturi (2013) "Sinkhorn Distances: Lightspeed Computation of Optimal Transport"

## Citation

If you use this velocity integration feature, please cite:

```
[Your scIDiff paper citation here]
```

## Support

For questions or issues:
- Open an issue on GitHub: https://github.com/manarai/scIDiff_V2
- Check the examples in `examples/` directory
- Review the implementation in `scqdiff/models/drift.py`
