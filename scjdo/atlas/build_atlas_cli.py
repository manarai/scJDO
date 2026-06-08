"""
Command-line interface for building Stable Operator Atlas
"""

import argparse
import torch
import anndata as ad
from pathlib import Path

from ..models.drift import DriftField
from .atlas_builder import StableOperatorAtlas


def main():
    parser = argparse.ArgumentParser(
        description="Build Stable Operator Atlas from trained scJDO model"
    )
    
    # Required arguments
    parser.add_argument(
        "--h5ad",
        type=str,
        required=True,
        help="Path to input AnnData file (.h5ad)"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained drift model (.pt)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--use-rep",
        type=str,
        default="X_pca",
        help="Key in adata.obsm for state representation (default: X_pca)"
    )
    parser.add_argument(
        "--pseudotime-key",
        type=str,
        default="pseudotime",
        help="Key in adata.obs for pseudotime (default: pseudotime)"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Threshold for near-neutral modes (default: 0.1)"
    )
    parser.add_argument(
        "--threshold-unstable",
        type=float,
        default=0.1,
        help="Threshold for unstable regime (default: 0.1)"
    )
    parser.add_argument(
        "--threshold-plastic",
        type=float,
        default=0.05,
        help="Threshold for plastic regime (default: 0.05)"
    )
    parser.add_argument(
        "--threshold-deeply-stable",
        type=float,
        default=-1.0,
        help="Threshold for deeply stable regime (default: -1.0)"
    )
    parser.add_argument(
        "--plasticity-threshold",
        type=float,
        default=0.3,
        help="Minimum plasticity index for plastic regime (default: 0.3)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing (default: 32)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for computation (default: cpu)"
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output path for atlas results (.h5ad)"
    )
    
    # Validation arguments
    parser.add_argument(
        "--celltype-key",
        type=str,
        default=None,
        help="Key in adata.obs for cell type labels (for validation)"
    )
    parser.add_argument(
        "--condition-key",
        type=str,
        default=None,
        help="Key in adata.obs for condition labels (for validation)"
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.h5ad}...")
    adata = ad.read_h5ad(args.h5ad)
    
    # Load model
    print(f"Loading drift model from {args.model}...")
    drift_model = torch.load(args.model, map_location=args.device)
    
    # Build atlas
    print("Initializing Stable Operator Atlas...")
    atlas = StableOperatorAtlas(
        adata=adata,
        drift_model=drift_model,
        use_rep=args.use_rep,
        pseudotime_key=args.pseudotime_key,
        device=args.device
    )
    
    atlas.build(
        epsilon=args.epsilon,
        threshold_unstable=args.threshold_unstable,
        threshold_plastic=args.threshold_plastic,
        threshold_deeply_stable=args.threshold_deeply_stable,
        plasticity_threshold=args.plasticity_threshold,
        batch_size=args.batch_size
    )
    
    # Validate non-redundancy if cell types provided
    if args.celltype_key:
        print("\nValidating non-redundancy with cell types...")
        validation = atlas.validate_nonredundancy(
            celltype_key=args.celltype_key,
            condition_key=args.condition_key
        )
    
    # Save results
    print(f"\nSaving atlas to {args.out}...")
    atlas.save(args.out)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
