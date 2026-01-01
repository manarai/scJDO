# Changelog: RNA Velocity Integration

## Version: RNA Velocity Feature (December 2025)

### Added

#### Core Implementation

- **KNNVelocity Module** (`scqdiff/models/drift.py`)
  - Soft k-nearest neighbors interpolation for smooth velocity fields
  - Confidence propagation from reference cells
  - Efficient PyTorch implementation with automatic device management
  - Configurable number of neighbors and temperature parameter

- **Extended DriftConfig** (`scqdiff/models/drift.py`)
  - `use_velocity_prior`: Enable/disable velocity integration
  - `vel_k`: Number of neighbors for KNN interpolation
  - `vel_tau`: Temperature for softmax weighting
  - `vel_scale`: Global velocity magnitude scaling
  - `vel_conf_power`: Confidence gating exponent
  - `vel_time_mode`: Time schedule type ("mid" or "flat")

- **Enhanced DriftField** (`scqdiff/models/drift.py`)
  - Accepts optional velocity reference data (X_ref, V_ref, W_ref)
  - Computes drift as: `f(x,t) = b(x,t) + u_θ(x,t)`
  - Time-dependent velocity contribution via scheduling
  - Confidence-based gating for reliability weighting
  - Preserves backward compatibility (model works without velocity)

#### Training Pipeline

- **Rewritten train_from_anndata.py** (`scqdiff/pipeline/train_from_anndata.py`)
  - New command-line arguments for velocity configuration
  - Optional velocity normalization (`--normalize-velocity`)
  - Removed naive MSE velocity loss
  - Velocity integrated as reference drift in model architecture
  - Improved argument organization and documentation
  - Saves training arguments in checkpoint for reproducibility

#### Documentation

- **RNA_VELOCITY_GUIDE.md**: Comprehensive user guide
  - Biological motivation and mathematical framework
  - Implementation details and architecture
  - Usage examples (CLI and Python API)
  - Hyperparameter tuning guidelines
  - Validation and troubleshooting
  - Comparison with naive approach

- **IMPLEMENTATION_SUMMARY.md**: Technical summary
  - Overview of changes
  - Testing results
  - Migration guide for existing users
  - Design decisions and rationale

- **QUICKSTART_VELOCITY.md**: Quick start guide
  - Basic usage examples
  - Key parameters
  - Common troubleshooting

- **CHANGELOG_VELOCITY.md**: This file
  - Detailed list of changes

#### Examples

- **train_with_velocity.py** (`examples/train_with_velocity.py`)
  - Example script demonstrating velocity-guided training
  - Executable with proper shebang
  - Includes usage documentation

### Changed

#### scqdiff/models/drift.py

- **ResidualNet**: Fixed input concatenation
  - Changed from `expand(-1, x.shape[1])` to proper concatenation
  - Now correctly handles `(B, D+1)` input shape
  - Maintains compatibility with time embedding

- **DriftField.__init__**: Extended signature
  - Added optional parameters: `X_ref`, `V_ref`, `W_ref`
  - Instantiates `KNNVelocity` when velocity prior enabled
  - Maintains backward compatibility (works without velocity)

- **DriftField.forward**: Enhanced computation
  - Adds velocity prior term when enabled
  - Applies time schedule and confidence gating
  - Preserves Laplacian smoothing on learned term only
  - Returns same output shape as before

#### scqdiff/pipeline/train_from_anndata.py

- **Complete rewrite** with velocity integration
  - Removed: `loss += 0.5*((u-V[idx])**2).mean()` (naive approach)
  - Added: Velocity as reference drift in model architecture
  - New arguments for velocity configuration
  - Optional velocity normalization
  - Better code organization and documentation
  - Enhanced checkpoint saving with training arguments

### Fixed

- **ResidualNet dimension mismatch**: Corrected input shape handling in concatenation
- **Time embedding**: Ensured proper broadcasting of time values

### Deprecated

- **Naive MSE velocity loss**: Removed from training pipeline
  - Old approach forced exact velocity matching
  - New approach uses velocity as biological prior
  - More stable and biologically interpretable

### Testing

- **test_velocity_implementation.py**: Comprehensive test suite
  - Tests KNN velocity interpolation
  - Tests drift field with and without velocity
  - Tests time schedules
  - Tests Jacobian computation
  - Tests model signature compatibility
  - All tests pass (6/6)

### Backward Compatibility

- **Fully backward compatible**
  - Model signature unchanged: `model(x,t) -> drift`
  - Default behavior preserved (velocity disabled by default)
  - Existing code continues to work
  - Old checkpoints can still be loaded

### Performance

- **Computational overhead**: Negligible
  - KNN interpolation: O(B × N) distance + O(B × k × D) interpolation
  - Dominated by neural network forward pass
  - Efficient PyTorch operations

- **Memory usage**: Minimal
  - Additional: 2 × N × D × 4 bytes for reference data
  - Example: 10k cells × 100 features ≈ 8 MB per tensor

### Known Issues

None at this time.

### Future Work

Potential enhancements for future versions:
- Adaptive confidence estimation from data
- Hierarchical interpolation for large-scale datasets
- Velocity field regularization
- Multi-scale velocity integration
- Uncertainty quantification

### Contributors

- Implementation based on ChatGPT conversation guidance
- Follows Schrödinger Bridge and RNA velocity best practices

### References

1. La Manno et al. (2018) "RNA velocity of single cells" Nature
2. Chen et al. (2021) "Likelihood Training of Schrödinger Bridge"
3. Cuturi (2013) "Sinkhorn Distances"

---

**Date**: December 19, 2025  
**Version**: RNA Velocity Feature Release  
**Status**: Tested and validated
