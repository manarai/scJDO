# Quick Start: RNA Velocity Integration

## Installation

No additional dependencies required beyond the existing scIDiff requirements. The velocity integration uses only PyTorch operations.

## Basic Usage

### Command Line

Train with RNA velocity as biological prior:

```bash
python -m scqdiff.pipeline.train_from_anndata \
    --h5ad your_data.h5ad \
    --use-velocity-prior \
    --normalize-velocity \
    --epochs 200 \
    --out-prefix my_model
```

### Python API

```python
import torch
import anndata as ad
from scqdiff.io.anndata import tensors_from_anndata
from scqdiff.models.drift import DriftField, DriftConfig

# Load data
adata = ad.read_h5ad("your_data.h5ad")
X, V, T = tensors_from_anndata(
    adata,
    vel_layer="velocity",
    pseudotime_key="latent_time"
)

# Normalize velocity (recommended)
if V is not None:
    V = V / (V.norm(dim=1, keepdim=True) + 1e-8)

# Configure model with velocity
cfg = DriftConfig(
    dim=X.shape[1],
    beta=0.1,
    use_velocity_prior=True,
    vel_scale=1.0,
    vel_time_mode="mid"
)

# Create model
model = DriftField(cfg, X_ref=X, V_ref=V)

# Use model
drift = model(X, T)
```

## Key Parameters

### Essential

- `--use-velocity-prior`: Enable velocity integration (required)
- `--normalize-velocity`: Normalize velocity to unit length (recommended)

### Tuning

- `--vel-scale 1.0`: Strength of velocity guidance (0.1-2.0)
- `--vel-k 32`: Number of neighbors for interpolation (16-64)
- `--vel-time-mode mid`: When velocity is strongest ("mid" or "flat")

## Validation

Check that velocity is being used:

```python
# Model should have velocity module
assert model.vel is not None, "Velocity not enabled"

# Test velocity interpolation
v, conf = model.vel(X[:10])
print(f"Velocity shape: {v.shape}")
print(f"Confidence range: [{conf.min():.3f}, {conf.max():.3f}]")
```

## Troubleshooting

**Velocity not affecting results?**
- Increase `--vel-scale` (try 2.0 or 5.0)
- Use `--vel-time-mode flat` instead of "mid"

**Model not matching endpoints?**
- Decrease `--vel-scale` (try 0.5 or 0.1)
- Keep `--vel-time-mode mid` (default)

**Training unstable?**
- Enable `--normalize-velocity`
- Increase `--vel-k` for smoother interpolation

## Next Steps

- Read [RNA_VELOCITY_GUIDE.md](RNA_VELOCITY_GUIDE.md) for detailed documentation
- See [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for technical details
- Check `examples/train_with_velocity.py` for complete example
