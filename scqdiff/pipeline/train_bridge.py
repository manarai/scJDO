"""
Training pipeline for Schrödinger Bridge.
"""

import argparse
import torch
import numpy as np
import anndata as ad
from pathlib import Path
from tqdm import trange
import json

from scqdiff.models.schrodinger_bridge import SchrodingerBridge, SchrodingerBridgeConfig
from scqdiff.io.anndata import tensors_from_anndata


def train_bridge_from_anndata(
    adata: ad.AnnData,
    source_key: str,
    target_key: str,
    source_value: str,
    target_value: str,
    cfg: SchrodingerBridgeConfig,
    n_iterations: int = 10,
    n_epochs_per_iter: int = 100,
    batch_size: int = 512,
    lr: float = 1e-3,
    out_prefix: str = 'sb_model'
):
    """
    Train Schrödinger Bridge from AnnData.
    
    Args:
        adata: AnnData object
        source_key: Obs key for source/target labels (e.g., 'age_group')
        target_key: Same as source_key (or different if needed)
        source_value: Value for source distribution (e.g., 'young')
        target_value: Value for target distribution (e.g., 'old')
        cfg: Bridge configuration
        n_iterations: Number of outer iterations (OT + training)
        n_epochs_per_iter: Epochs per iteration
        batch_size: Batch size
        lr: Learning rate
        out_prefix: Output file prefix
    """
    print("="*80)
    print("Training Schrödinger Bridge")
    print("="*80)
    
    # Extract data
    print(f"\nExtracting data...")
    X, _, _ = tensors_from_anndata(adata, use_raw=False)
    
    # Filter source and target
    source_mask = adata.obs[source_key] == source_value
    target_mask = adata.obs[target_key] == target_value
    
    X_0 = X[source_mask]
    X_1 = X[target_mask]
    
    print(f"Source ({source_value}): {X_0.shape[0]} cells")
    print(f"Target ({target_value}): {X_1.shape[0]} cells")
    print(f"Dimension: {X_0.shape[1]} features")
    
    # Create model
    print(f"\nCreating Schrödinger Bridge model...")
    model = SchrodingerBridge(cfg, X_0, X_1)
    print(f"Forward parameters: {sum(p.numel() for p in model.forward_net.parameters()):,}")
    print(f"Backward parameters: {sum(p.numel() for p in model.backward_net.parameters()):,}")
    
    # Optimizers
    opt_forward = torch.optim.Adam(model.forward_net.parameters(), lr=lr)
    opt_backward = torch.optim.Adam(model.backward_net.parameters(), lr=lr)
    
    # Training loop
    print(f"\nTraining for {n_iterations} iterations...")
    history = {
        'loss_forward': [],
        'loss_backward': [],
        'loss_endpoint': [],
        'loss_total': []
    }
    
    for iteration in range(n_iterations):
        print(f"\n{'='*80}")
        print(f"Iteration {iteration+1}/{n_iterations}")
        print(f"{'='*80}")
        
        # Update OT plan
        print("Computing optimal transport plan...")
        model.compute_ot_plan()
        ot_cost = (model.P * torch.cdist(X_0, X_1)**2).sum().item()
        print(f"OT cost: {ot_cost:.4f}")
        
        # Train for multiple epochs
        pbar = trange(n_epochs_per_iter, desc=f"Iter {iteration+1}")
        for epoch in pbar:
            # Get losses
            losses = model.train_step(batch_size, update_ot=False)
            
            # Update forward network
            opt_forward.zero_grad()
            losses['forward'].backward(retain_graph=True)
            opt_forward.step()
            
            # Update backward network
            opt_backward.zero_grad()
            losses['backward'].backward()
            opt_backward.step()
            
            # Log
            history['loss_forward'].append(losses['forward'].item())
            history['loss_backward'].append(losses['backward'].item())
            history['loss_endpoint'].append(losses['endpoint'].item())
            history['loss_total'].append(losses['total'].item())
            
            pbar.set_postfix({
                'fwd': f"{losses['forward'].item():.4f}",
                'bwd': f"{losses['backward'].item():.4f}",
                'end': f"{losses['endpoint'].item():.4f}"
            })
    
    # Save model
    print(f"\nSaving model to {out_prefix}.pt...")
    torch.save({
        'model_state': {
            'forward_net': model.forward_net.state_dict(),
            'backward_net': model.backward_net.state_dict()
        },
        'cfg': cfg.__dict__,
        'X_0': X_0,
        'X_1': X_1,
        'P': model.P,
        'f': model.f,
        'g': model.g,
        'history': history
    }, f"{out_prefix}.pt")
    
    # Save history
    print(f"Saving training history to {out_prefix}_history.json...")
    with open(f"{out_prefix}_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*80)
    print("Training complete!")
    print("="*80)
    
    return model, history


def main():
    ap = argparse.ArgumentParser(
        description="Train Schrödinger Bridge from AnnData"
    )
    
    # Data arguments
    ap.add_argument('--h5ad', required=True, help="Path to AnnData h5ad file")
    ap.add_argument('--source-key', required=True, help="Obs key for source/target")
    ap.add_argument('--source-value', required=True, help="Value for source (e.g., 'young')")
    ap.add_argument('--target-value', required=True, help="Value for target (e.g., 'old')")
    
    # Model arguments
    ap.add_argument('--hidden', type=int, default=256, help="Hidden dimension")
    ap.add_argument('--depth', type=int, default=4, help="Network depth")
    ap.add_argument('--beta', type=float, default=0.1, help="Diffusion coefficient")
    ap.add_argument('--sigma', type=float, default=0.2, help="Noise level")
    ap.add_argument('--epsilon', type=float, default=0.1, help="OT regularization")
    
    # Training arguments
    ap.add_argument('--n-iterations', type=int, default=10, help="Number of outer iterations")
    ap.add_argument('--n-epochs', type=int, default=100, help="Epochs per iteration")
    ap.add_argument('--batch-size', type=int, default=512, help="Batch size")
    ap.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    
    # Output arguments
    ap.add_argument('--out-prefix', type=str, default='sb_model', help="Output prefix")
    ap.add_argument('--device', type=str, default='cpu', help="Device (cpu/cuda)")
    
    args = ap.parse_args()
    
    # Load data
    print(f"Loading data from {args.h5ad}...")
    adata = ad.read_h5ad(args.h5ad)
    
    # Create config
    cfg = SchrodingerBridgeConfig(
        dim=adata.n_vars,
        hidden=args.hidden,
        depth=args.depth,
        beta=args.beta,
        sigma=args.sigma,
        epsilon=args.epsilon,
        device=args.device
    )
    
    # Train
    model, history = train_bridge_from_anndata(
        adata,
        source_key=args.source_key,
        target_key=args.source_key,
        source_value=args.source_value,
        target_value=args.target_value,
        cfg=cfg,
        n_iterations=args.n_iterations,
        n_epochs_per_iter=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        out_prefix=args.out_prefix
    )


if __name__ == '__main__':
    main()
