
import argparse
import torch
import json
import numpy as np
import anndata as ad
from scqdiff.io.anndata import tensors_from_anndata, laplacian_from_connectivities
from scqdiff.models.drift import DriftField, DriftConfig
from scqdiff.losses import (
    denoising_score_matching,
    control_energy,
    fp_residual_loss,
    laplacian_smooth_drift
)
from scqdiff.utils.graph import build_knn_graph, graph_laplacian
from scqdiff.archetypes.decompose import jacobian_modes
from scqdiff.archetypes.cellrank import (
    fate_probs_from_anndata,
    fate_conditioned_mean_jacobians
)
from tqdm import trange


def main():
    ap = argparse.ArgumentParser(
        description="Train scIDiff drift field from AnnData with optional RNA velocity prior"
    )
    
    # Data arguments
    ap.add_argument('--h5ad', required=True, help="Path to AnnData h5ad file")
    ap.add_argument('--vel-layer', default=None, help="Velocity layer name (auto-detected if None)")
    ap.add_argument('--ptime-key', default=None, help="Pseudotime key (auto-detected if None)")
    ap.add_argument('--use-raw', action='store_true', help="Use raw counts from adata.raw")
    ap.add_argument('--n-hvg', type=int, default=None, help="Number of highly variable genes")
    
    # Training arguments
    ap.add_argument('--epochs', type=int, default=200, help="Number of training epochs")
    ap.add_argument('--beta', type=float, default=0.1, help="Diffusion coefficient")
    ap.add_argument('--sigma', type=float, default=0.2, help="Noise level for score matching")
    ap.add_argument('--batch-size', type=int, default=2048, help="Batch size for training")
    ap.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    
    # Regularization arguments
    ap.add_argument('--laplacian-lambda', type=float, default=0.0,
                    help="Laplacian smoothing strength in drift")
    ap.add_argument('--laplacian-reg', type=float, default=1e-3,
                    help="Laplacian smoothing regularization weight")
    ap.add_argument('--k', type=int, default=15, help="Number of neighbors for KNN graph")
    
    # RNA velocity prior arguments
    ap.add_argument('--use-velocity-prior', action='store_true',
                    help="Use RNA velocity as biological prior")
    ap.add_argument('--vel-k', type=int, default=32,
                    help="Number of neighbors for velocity interpolation")
    ap.add_argument('--vel-tau', type=float, default=1.0,
                    help="Temperature for velocity KNN softmax")
    ap.add_argument('--vel-scale', type=float, default=1.0,
                    help="Global scaling factor for velocity magnitude")
    ap.add_argument('--vel-conf-power', type=float, default=1.0,
                    help="Confidence gating exponent")
    ap.add_argument('--vel-time-mode', type=str, default='mid', choices=['mid', 'flat'],
                    help="Time schedule for velocity contribution")
    ap.add_argument('--normalize-velocity', action='store_true',
                    help="Normalize velocity vectors to unit length")
    
    # Archetype analysis arguments
    ap.add_argument('--fate-index', type=int, default=None,
                    help="Fate index for archetype analysis")
    ap.add_argument('--nbins', type=int, default=10,
                    help="Number of bins for fate-conditioned analysis")
    ap.add_argument('--rank', type=int, default=3,
                    help="Rank for archetype decomposition")
    
    # Output arguments
    ap.add_argument('--out-prefix', type=str, default='scqdiff_from_anndata',
                    help="Prefix for output files")
    
    args = ap.parse_args()
    
    # Load data
    print(f"Loading data from {args.h5ad}...")
    adata = ad.read_h5ad(args.h5ad)
    X, V, T = tensors_from_anndata(
        adata,
        use_raw=args.use_raw,
        n_hvg=args.n_hvg,
        vel_layer=args.vel_layer,
        pseudotime_key=args.ptime_key
    )
    print(f"Loaded {X.shape[0]} cells with {X.shape[1]} features")
    
    # Handle velocity data
    X_ref, V_ref = None, None
    if V is not None:
        print(f"Velocity data found: {V.shape}")
        if args.use_velocity_prior:
            print("Velocity prior enabled")
            X_ref, V_ref = X, V
            
            # Optionally normalize velocity vectors
            if args.normalize_velocity:
                print("Normalizing velocity vectors to unit length...")
                v_norm = V.norm(dim=1, keepdim=True) + 1e-8
                V_ref = V / v_norm
        else:
            print("Velocity prior disabled (use --use-velocity-prior to enable)")
    else:
        print("No velocity data found")
        if args.use_velocity_prior:
            print("WARNING: --use-velocity-prior specified but no velocity data available")
            args.use_velocity_prior = False
    
    # Build or load graph Laplacian
    L = laplacian_from_connectivities(adata)
    if L is None:
        print(f"Building KNN graph with k={args.k}...")
        A = build_knn_graph(X.numpy(), k=args.k, mode='distance')
        L = graph_laplacian(A, normalized=True)
    else:
        print("Using pre-computed connectivities from AnnData")
    
    # Create model configuration
    cfg = DriftConfig(
        dim=X.shape[1],
        beta=args.beta,
        laplacian_lambda=args.laplacian_lambda,
        use_velocity_prior=args.use_velocity_prior,
        vel_k=args.vel_k,
        vel_tau=args.vel_tau,
        vel_scale=args.vel_scale,
        vel_conf_power=args.vel_conf_power,
        vel_time_mode=args.vel_time_mode
    )
    
    # Instantiate model
    print("Creating drift field model...")
    model = DriftField(cfg, laplacian=L, X_ref=X_ref, V_ref=V_ref)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    for ep in trange(args.epochs):
        # Sample mini-batch
        idx = np.random.choice(
            X.shape[0],
            size=min(args.batch_size, X.shape[0]),
            replace=True
        )
        xb, tb = X[idx], T[idx]
        
        # Compute losses
        # 1. Denoising score matching (main loss)
        loss = denoising_score_matching(model, xb, tb, sigma=args.sigma)
        
        # 2. Control energy regularization
        u = model(xb, tb)
        loss += 1e-3 * control_energy(u)
        
        # 3. Fokker-Planck residual loss
        loss += 1e-3 * fp_residual_loss(model, xb, tb, beta=args.beta)
        
        # 4. Laplacian smoothing regularization
        loss += args.laplacian_reg * laplacian_smooth_drift(u, model.L)
        
        # Update
        opt.zero_grad()
        loss.backward()
        opt.step()
    
    # Save model
    print(f"\nSaving model to {args.out_prefix}.pt...")
    torch.save({
        'model': model.state_dict(),
        'cfg': cfg,
        'args': vars(args)
    }, f"{args.out_prefix}.pt")
    
    # Archetype analysis (if requested)
    if args.fate_index is not None:
        print(f"\nPerforming fate-conditioned archetype analysis...")
        P, names = fate_probs_from_anndata(adata)
        if P is not None:
            J_bins, edges = fate_conditioned_mean_jacobians(
                model, X, T, P,
                nbins=args.nbins,
                fate_idx=args.fate_index
            )
            patterns, U_t, S = jacobian_modes(J_bins, rank=args.rank)
            
            # Save archetype results
            np.save(f"{args.out_prefix}.J_bins.npy", J_bins.cpu().numpy())
            np.save(f"{args.out_prefix}.patterns.npy", patterns.cpu().numpy())
            np.save(f"{args.out_prefix}.U_t.npy", U_t.cpu().numpy())
            np.save(f"{args.out_prefix}.S.npy", S.cpu().numpy())
            
            meta = {
                'edges': edges.tolist(),
                'fate_index': int(args.fate_index),
                'fate_name': (names[args.fate_index]
                             if names and args.fate_index < len(names)
                             else None)
            }
            with open(f"{args.out_prefix}.meta.json", 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2)
            
            print(f"Saved fate-conditioned archetypes to {args.out_prefix}.*")
        else:
            print("No CellRank fate probabilities found; skipping archetype analysis")
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
