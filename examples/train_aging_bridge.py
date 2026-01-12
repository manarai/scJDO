"""
Example: Train Schrödinger Bridge on aging data.
"""

import torch
import numpy as np
import anndata as ad
import scanpy as sc
from scqdiff.models.schrodinger_bridge import SchrodingerBridge, SchrodingerBridgeConfig
from scqdiff.pipeline.train_bridge import train_bridge_from_anndata


def create_synthetic_aging_data(n_young=1000, n_old=1000, n_genes=50):
    """Create synthetic aging data for testing."""
    
    # Young cells: cluster around origin
    X_young = np.random.randn(n_young, n_genes) * 0.5
    
    # Old cells: shifted and more dispersed
    X_old = np.random.randn(n_old, n_genes) * 1.0 + 2.0
    
    # Combine
    X = np.vstack([X_young, X_old])
    age_labels = ['young'] * n_young + ['old'] * n_old
    
    # Create AnnData
    adata = ad.AnnData(X)
    adata.obs['age_group'] = age_labels
    
    return adata


def main():
    print("Creating synthetic aging data...")
    adata = create_synthetic_aging_data(n_young=1000, n_old=1000, n_genes=50)
    
    print("\nTraining Schrödinger Bridge...")
    cfg = SchrodingerBridgeConfig(
        dim=50,
        hidden=128,
        depth=3,
        beta=0.1,
        sigma=0.2,
        epsilon=0.1,
        device='cpu'
    )
    
    model, history = train_bridge_from_anndata(
        adata,
        source_key='age_group',
        target_key='age_group',
        source_value='young',
        target_value='old',
        cfg=cfg,
        n_iterations=5,
        n_epochs_per_iter=50,
        batch_size=256,
        lr=1e-3,
        out_prefix='synthetic_aging_bridge'
    )
    
    print("\nTesting forward integration (aging)...")
    young_sample = torch.tensor(adata[adata.obs['age_group'] == 'young'].X[:10], dtype=torch.float32)
    aging_trajectory = model.forward_integrate(young_sample, steps=50)
    print(f"Aging trajectory shape: {aging_trajectory.shape}")
    
    print("\nTesting backward integration (rejuvenation)...")
    old_sample = torch.tensor(adata[adata.obs['age_group'] == 'old'].X[:10], dtype=torch.float32)
    rejuvenation_trajectory = model.backward_integrate(old_sample, steps=50)
    print(f"Rejuvenation trajectory shape: {rejuvenation_trajectory.shape}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
