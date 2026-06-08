"""
scjdo — unified command-line interface.

Usage
-----
scjdo drift  input.h5ad  [options]  --out results/
scjdo bridge input.h5ad  [options]  --out results/
"""
from __future__ import annotations
import argparse, os, sys


# ---------------------------------------------------------------------------
# Shared preprocessing args
# ---------------------------------------------------------------------------

def _add_pp_args(p):
    p.add_argument("--groupby",    required=True, help="adata.obs column for cell type labels")
    p.add_argument("--root",       required=True, help="Root cluster for DPT pseudotime")
    p.add_argument("--n-hvg",      type=int, default=2000, metavar="N", help="Highly variable genes (default 2000)")
    p.add_argument("--n-pcs",      type=int, default=50,   metavar="N", help="PCA components (default 50)")
    p.add_argument("--n-neighbors",type=int, default=15,   metavar="N", help="kNN graph neighbours (default 15)")


# ---------------------------------------------------------------------------
# scjdo drift
# ---------------------------------------------------------------------------

def _run_drift(args):
    import matplotlib; matplotlib.use("Agg")
    import scanpy as sc
    import scjdo as sjd

    os.makedirs(args.out, exist_ok=True)
    fig_dir = os.path.join(args.out, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print(f"[scjdo drift] Loading {args.input}")
    adata = sc.read_h5ad(args.input)

    print("[scjdo drift] Preprocessing...")
    sjd.pp.prepare_trajectory(
        adata,
        groupby    = args.groupby,
        root       = args.root,
        n_hvg      = args.n_hvg,
        n_pcs      = args.n_pcs,
        n_neighbors= args.n_neighbors,
    )

    print(f"[scjdo drift] Training drift field ({args.n_epochs} epochs)...")
    sjd.tl.fit_drift(
        adata,
        n_archetypes = args.n_archetypes,
        n_epochs     = args.n_epochs,
        vel_scale    = args.vel_scale,
        vel_time_mode= args.vel_time_mode,
        n_windows    = args.n_windows,
        seed         = args.seed,
    )

    print("[scjdo drift] Generating figures...")
    sjd.pl.summary_figure(adata, basis="X_pca",
                           save=os.path.join(fig_dir, "drift_summary.pdf"))
    sjd.pl.drift_field(adata,  save=os.path.join(fig_dir, "drift_field.pdf"))
    sjd.pl.sensitivity(adata,  save=os.path.join(fig_dir, "sensitivity.pdf"))
    sjd.pl.archetypes(adata,   save=os.path.join(fig_dir, "archetypes.pdf"))
    sjd.pl.coordination(adata, save=os.path.join(fig_dir, "coordination.pdf"))
    sjd.pl.instability_genes(adata, n_genes=args.n_genes,
                              save=os.path.join(fig_dir, "instability_genes.pdf"))

    print("[scjdo drift] Saving outputs...")
    import pandas as pd
    table = sjd.tl.get_instability_genes(adata, n_genes=args.n_genes)
    table.to_csv(os.path.join(args.out, "instability_genes.csv"), index=False)

    adata.write_h5ad(os.path.join(args.out, "adata_drift.h5ad"))

    print(f"\n[scjdo drift] Done. Results in: {args.out}/")
    print(f"  adata_drift.h5ad")
    print(f"  instability_genes.csv")
    print(f"  figures/drift_summary.pdf  (+ 5 individual panels)")


# ---------------------------------------------------------------------------
# scjdo bridge
# ---------------------------------------------------------------------------

def _run_bridge(args):
    import matplotlib; matplotlib.use("Agg")
    import scanpy as sc
    import scjdo as sjd

    os.makedirs(args.out, exist_ok=True)
    fig_dir = os.path.join(args.out, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print(f"[scjdo bridge] Loading {args.input}")
    adata = sc.read_h5ad(args.input)

    print("[scjdo bridge] Preprocessing...")
    sjd.pp.prepare_trajectory(
        adata,
        groupby    = args.groupby,
        root       = args.root,
        n_hvg      = args.n_hvg,
        n_pcs      = args.n_pcs,
        n_neighbors= args.n_neighbors,
    )

    print("[scjdo bridge] Training Schrödinger Bridge...")
    sjd.tl.fit_bridge(
        adata,
        src_quantile = args.src_quantile,
        tgt_quantile = args.tgt_quantile,
        src_group    = args.src_group,
        tgt_group    = args.tgt_group,
        groupby      = args.groupby if (args.src_group or args.tgt_group) else None,
        n_archetypes = args.n_archetypes,
        epsilon      = args.epsilon,
        t_steps      = args.t_steps,
        n_genes      = args.n_genes,
        seed         = args.seed,
    )

    print("[scjdo bridge] Generating figures...")
    sjd.pl.bridge_summary(adata, save=os.path.join(fig_dir, "bridge_summary.pdf"))
    sjd.pl.bridge_gene_comparison(adata, n_genes=args.n_genes,
                                   save=os.path.join(fig_dir, "bridge_genes.pdf"))

    print("[scjdo bridge] Saving outputs...")
    df_fwd, df_bwd = sjd.tl.get_bridge_instability_genes(adata)
    df_fwd.to_csv(os.path.join(args.out, "instability_genes_forward.csv"),  index=False)
    df_bwd.to_csv(os.path.join(args.out, "instability_genes_backward.csv"), index=False)

    adata.write_h5ad(os.path.join(args.out, "adata_bridge.h5ad"))

    print(f"\n[scjdo bridge] Done. Results in: {args.out}/")
    print(f"  adata_bridge.h5ad")
    print(f"  instability_genes_forward.csv")
    print(f"  instability_genes_backward.csv")
    print(f"  figures/bridge_summary.pdf")
    print(f"  figures/bridge_genes.pdf")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="scjdo",
        description="scJDO — single-cell Jacobian drift operators",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ── scjdo drift ──────────────────────────────────────────────────────
    dp = sub.add_parser("drift", help="Drift field + archetype analysis",
                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    dp.add_argument("input", help="Input AnnData (.h5ad)")
    _add_pp_args(dp)
    dp.add_argument("--n-archetypes", type=int,   default=5,       metavar="K")
    dp.add_argument("--n-epochs",     type=int,   default=5000,    metavar="N")
    dp.add_argument("--n-windows",    type=int,   default=100,     metavar="N")
    dp.add_argument("--vel-scale",    type=float, default=2.0,     metavar="F")
    dp.add_argument("--vel-time-mode",            default="flat",
                    choices=["flat", "mid", "root", "rise"])
    dp.add_argument("--n-genes",      type=int,   default=20,      metavar="N")
    dp.add_argument("--seed",         type=int,   default=42)
    dp.add_argument("--out",                      default="drift_results", metavar="DIR")

    # ── scjdo bridge ─────────────────────────────────────────────────────
    bp = sub.add_parser("bridge", help="Schrödinger Bridge + instability genes",
                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    bp.add_argument("input", help="Input AnnData (.h5ad)")
    _add_pp_args(bp)
    bp.add_argument("--src-quantile", type=float, default=0.20, metavar="Q",
                    help="Bottom pseudotime fraction → source")
    bp.add_argument("--tgt-quantile", type=float, default=0.80, metavar="Q",
                    help="Top pseudotime fraction → target")
    bp.add_argument("--src-group",                default=None,
                    help="Source cluster name (overrides --src-quantile)")
    bp.add_argument("--tgt-group",                default=None,
                    help="Target cluster name (overrides --tgt-quantile)")
    bp.add_argument("--n-archetypes", type=int,   default=4,    metavar="K")
    bp.add_argument("--epsilon",      type=float, default=0.5,  metavar="F",
                    help="OT regularization")
    bp.add_argument("--t-steps",      type=int,   default=30,   metavar="N",
                    help="Bridge time steps for Jacobian analysis")
    bp.add_argument("--n-genes",      type=int,   default=20,   metavar="N")
    bp.add_argument("--seed",         type=int,   default=42)
    bp.add_argument("--out",                      default="bridge_results", metavar="DIR")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)
    elif args.command == "drift":
        _run_drift(args)
    elif args.command == "bridge":
        _run_bridge(args)


if __name__ == "__main__":
    main()
