"""
Training pipeline for Schrödinger Bridge.
"""

import argparse
import json

import anndata as ad
import torch

from scqdiff.io.anndata import tensors_from_anndata
from scqdiff.models.schrodinger_bridge import SchrodingerBridge, SchrodingerBridgeConfig


def train_bridge_from_anndata(
    adata: ad.AnnData,
    source_key: str,
    target_key: str,
    source_value: str,
    target_value: str,
    cfg: SchrodingerBridgeConfig,
    n_iterations: int = None,
    batch_size: int = 512,
    out_prefix: str = "sb_model",
    verbose: bool = True,
):
    """
    Train Schrödinger Bridge from AnnData.

    Args:
        adata: AnnData object
        source_key: Obs key for source labels (e.g., 'age_group')
        target_key: Obs key for target labels (typically same as source_key)
        source_value: Value for source distribution (e.g., 'young')
        target_value: Value for target distribution (e.g., 'old')
        cfg: Bridge configuration
        n_iterations: Max outer iterations (defaults to cfg.max_iterations)
        batch_size: Pairs sampled per iteration
        out_prefix: Output file prefix
        verbose: Print progress
    """
    print("=" * 80)
    print("Training Schrödinger Bridge")
    print("=" * 80)

    X, _, _ = tensors_from_anndata(adata)

    source_mask = (adata.obs[source_key] == source_value).values
    target_mask = (adata.obs[target_key] == target_value).values

    X_0 = X[source_mask]
    X_1 = X[target_mask]

    print(f"Source ({source_value}): {X_0.shape[0]} cells")
    print(f"Target ({target_value}): {X_1.shape[0]} cells")
    print(f"Dimension: {X_0.shape[1]} features")

    model = SchrodingerBridge(cfg, X_0, X_1)
    print(f"Forward parameters:  {sum(p.numel() for p in model.forward_net.parameters()):,}")
    print(f"Backward parameters: {sum(p.numel() for p in model.backward_net.parameters()):,}")

    history = model.train_bridge(
        n_iterations=n_iterations,
        batch_size=batch_size,
        verbose=verbose,
    )

    print(f"\nSaving model to {out_prefix}.pt ...")
    torch.save(
        {
            "model_state": {
                "forward_net": model.forward_net.state_dict(),
                "backward_net": model.backward_net.state_dict(),
            },
            "cfg": cfg.__dict__,
            "X_0": X_0,
            "X_1": X_1,
        },
        f"{out_prefix}.pt",
    )

    print(f"Saving training history to {out_prefix}_history.json ...")
    with open(f"{out_prefix}_history.json", "w") as f:
        serialisable = {
            k: (v if isinstance(v, (bool, int)) else [float(x) for x in v])
            for k, v in history.items()
        }
        json.dump(serialisable, f, indent=2)

    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)

    return model, history


def main():
    ap = argparse.ArgumentParser(description="Train Schrödinger Bridge from AnnData")

    ap.add_argument("--h5ad", required=True, help="Path to AnnData h5ad file")
    ap.add_argument("--source-key", required=True, help="Obs key for source/target labels")
    ap.add_argument("--source-value", required=True, help="Value for source distribution (e.g., 'young')")
    ap.add_argument("--target-value", required=True, help="Value for target distribution (e.g., 'old')")
    ap.add_argument("--hidden", type=int, default=256, help="Hidden dimension")
    ap.add_argument("--depth", type=int, default=4, help="Network depth")
    ap.add_argument("--beta", type=float, default=0.1, help="Diffusion coefficient")
    ap.add_argument("--epsilon", type=float, default=0.1, help="OT regularization (Sinkhorn ε)")
    ap.add_argument("--n-iterations", type=int, default=None, help="Max outer iterations")
    ap.add_argument("--batch-size", type=int, default=512, help="Batch size")
    ap.add_argument("--out-prefix", type=str, default="sb_model", help="Output prefix")

    args = ap.parse_args()

    adata = ad.read_h5ad(args.h5ad)

    cfg = SchrodingerBridgeConfig(
        dim=adata.n_vars,
        hidden=args.hidden,
        depth=args.depth,
        beta=args.beta,
        epsilon=args.epsilon,
    )

    train_bridge_from_anndata(
        adata,
        source_key=args.source_key,
        target_key=args.source_key,
        source_value=args.source_value,
        target_value=args.target_value,
        cfg=cfg,
        n_iterations=args.n_iterations,
        batch_size=args.batch_size,
        out_prefix=args.out_prefix,
    )


if __name__ == "__main__":
    main()
