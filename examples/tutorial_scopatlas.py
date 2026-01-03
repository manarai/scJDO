"""
Tutorial: Building a Stable Operator Atlas with scidiff

This tutorial demonstrates how to use scidiff to construct a Stable Operator Atlas
that defines cellular states by their local stability structure.

The Stable Operator Atlas provides a dynamical layer of cell identity that is
invisible to expression-based atlases.
"""

import torch
import numpy as np
import anndata as ad
import scanpy as sc

# Import scidiff modules
from scqdiff.models.drift import DriftField, DriftConfig
from scqdiff.atlas import StableOperatorAtlas
from scqdiff.atlas.visualization import (
    plot_operator_regimes,
    plot_stability_depth_map,
    plot_plasticity_map,
    plot_nonredundancy_comparison,
    plot_temporal_evolution,
    plot_combined_overview
)

# ============================================================================
# Step 1: Load your data and trained drift model
# ============================================================================

print("Step 1: Loading data and model...")

# Load AnnData with single-cell data
# Your data should have:
# - PCA representation in adata.obsm['X_pca']
# - Pseudotime in adata.obs['pseudotime']
# - Cell type annotations in adata.obs['cell_type']
# - Condition labels in adata.obs['condition'] (optional)

adata = ad.read_h5ad("your_data.h5ad")

# Load trained drift model
# This should be a model trained using scidiff's train_from_anndata pipeline
drift_model = torch.load("your_model.pt")

print(f"Loaded {adata.n_obs} cells with {adata.n_vars} genes")

# ============================================================================
# Step 2: Build the Stable Operator Atlas
# ============================================================================

print("\nStep 2: Building Stable Operator Atlas...")

# Initialize atlas
atlas = StableOperatorAtlas(
    adata=adata,
    drift_model=drift_model,
    use_rep="X_pca",              # Use PCA representation
    pseudotime_key="pseudotime",  # Use pseudotime for temporal analysis
    device="cpu"                  # Use "cuda" if GPU available
)

# Build atlas (compute operator metrics and classify regimes)
atlas.build(
    epsilon=0.1,                    # Threshold for near-neutral modes
    threshold_unstable=0.1,         # Threshold for unstable regime
    threshold_plastic=0.05,         # Threshold for plastic regime
    threshold_deeply_stable=-1.0,   # Threshold for deeply stable regime
    plasticity_threshold=0.3,       # Minimum plasticity for plastic regime
    batch_size=32,                  # Batch size for processing
    compute_confidence=True         # Compute confidence scores
)

# Results are now stored in adata.obs:
# - 'operator_regime': Regime labels (stable/plastic/unstable/deeply_stable)
# - 'lambda_max_plus': Max unstable eigenvalue
# - 'lambda_min_minus': Stability depth
# - 'plasticity': Plasticity index
# - 'stable_dim': Stable subspace dimension
# - 'regime_confidence': Confidence scores

print("\nOperator metrics computed and stored in adata.obs")

# ============================================================================
# Step 3: Validate non-redundancy with cell types
# ============================================================================

print("\nStep 3: Validating non-redundancy with expression-based cell types...")

# This is critical for demonstrating that operator regimes provide
# information beyond expression-based cell type annotations

validation = atlas.validate_nonredundancy(
    celltype_key='cell_type',
    condition_key='condition'  # Optional: compare across conditions
)

# The validation tests:
# 1. Same cell type → different operator regimes (regime diversity)
# 2. Same cell type, different conditions → different regime distributions

print("\nNon-redundancy validation complete!")

# ============================================================================
# Step 4: Visualize operator regimes
# ============================================================================

print("\nStep 4: Creating visualizations...")

# Plot 1: Operator regimes on UMAP
plot_operator_regimes(
    adata,
    basis="umap",
    save="operator_regimes_umap.png"
)

# Plot 2: Stability depth map
plot_stability_depth_map(
    adata,
    basis="umap",
    save="stability_depth_map.png"
)

# Plot 3: Plasticity map
plot_plasticity_map(
    adata,
    basis="umap",
    save="plasticity_map.png"
)

# Plot 4: Non-redundancy comparison (critical figure for publication)
plot_nonredundancy_comparison(
    adata,
    celltype_key='cell_type',
    condition_key='condition',
    save="nonredundancy_comparison.png"
)

# Plot 5: Temporal evolution of operator metrics
plot_temporal_evolution(
    adata,
    pseudotime_key='pseudotime',
    save="temporal_evolution.png"
)

# Plot 6: Combined overview
plot_combined_overview(
    adata,
    basis="umap",
    celltype_key='cell_type',
    save="combined_overview.png"
)

print("\nVisualizations saved!")

# ============================================================================
# Step 5: Compare regimes across conditions
# ============================================================================

print("\nStep 5: Comparing operator regimes across conditions...")

comparison = atlas.compare_conditions(
    condition_key='condition',
    celltype_key='cell_type'
)

# Print summary
print("\nRegime distribution by condition:")
for cond, stats in comparison.items():
    if cond != 'by_celltype':
        print(f"\n{cond}:")
        for regime, frac in stats['regime_fractions'].items():
            print(f"  {regime:15s}: {frac*100:5.1f}%")

# ============================================================================
# Step 6: Get regime statistics
# ============================================================================

print("\nStep 6: Computing regime statistics...")

regime_stats = atlas.get_regime_statistics()

print("\nOperator metrics by regime:")
for regime, stats in regime_stats.items():
    print(f"\n{regime}:")
    print(f"  Count: {stats['count']}")
    print(f"  λ_max⁺: {stats['lambda_max_mean']:.3f} ± {stats['lambda_max_std']:.3f}")
    print(f"  λ_min⁻: {stats['lambda_min_mean']:.3f} ± {stats['lambda_min_std']:.3f}")
    print(f"  Plasticity: {stats['plasticity_mean']:.3f} ± {stats['plasticity_std']:.3f}")
    print(f"  Stable dim: {stats['stable_dim_mean']:.1f} ± {stats['stable_dim_std']:.1f}")

# ============================================================================
# Step 7: Save atlas results
# ============================================================================

print("\nStep 7: Saving atlas results...")

atlas.save("atlas_results.h5ad")

print("\nAtlas saved to atlas_results.h5ad")

# ============================================================================
# Step 8: Biological interpretation
# ============================================================================

print("\n" + "="*70)
print("BIOLOGICAL INTERPRETATION GUIDE")
print("="*70)

print("""
Operator Regimes:

1. STABLE (Green)
   - λ_max⁺ ≤ 0, large stable subspace
   - Biological meaning: Terminal differentiation, homeostasis
   - Examples: Mature cell types, quiescent states

2. PLASTIC (Orange)
   - λ_max⁺ ≈ 0, high plasticity index
   - Biological meaning: Progenitor states, decision points
   - Examples: Stem cells, multipotent progenitors

3. UNSTABLE (Red)
   - λ_max⁺ > threshold
   - Biological meaning: Transition states, bifurcations
   - Examples: Cells undergoing differentiation, stress response

4. DEEPLY STABLE (Blue)
   - Very negative λ_min⁻
   - Biological meaning: Resistant states, locked-in fates
   - Examples: Senescent cells, exhausted immune cells

Key Insights:

- Same cell type can have different operator regimes across conditions
  → Reveals dynamical changes invisible to expression analysis

- Operator regimes predict:
  * Response to perturbations
  * Aging-related changes
  * Drug resistance
  * Reprogramming potential

- Use operator regimes to:
  * Identify intervention targets
  * Predict cell fate decisions
  * Understand commitment mechanisms
""")

print("\nTutorial complete!")
print("\nNext steps:")
print("1. Anchor findings to biological axis (e.g., immune aging)")
print("2. Overlay with ATAC-seq for chromatin validation")
print("3. Test perturbation predictions")
print("4. Integrate with CellRank for fate analysis")
