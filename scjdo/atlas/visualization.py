"""
Visualization tools for Stable Operator Atlas

Provides plotting functions for operator regimes, stability maps, and comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List, Tuple
import warnings

try:
    import scanpy as sc
    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False
    warnings.warn("scanpy not available. Some visualization features will be limited.")


def plot_operator_regimes(
    adata,
    basis: str = "umap",
    regime_key: str = "operator_regime",
    color_map: Optional[Dict[str, str]] = None,
    figsize: Tuple[float, float] = (8, 6),
    save: Optional[str] = None
):
    """
    Plot operator regimes on a 2D embedding (UMAP, t-SNE, etc.).
    
    Args:
        adata: AnnData object with atlas results
        basis: Embedding to use (default: "umap")
        regime_key: Key in adata.obs for regime labels
        color_map: Optional custom color map for regimes
        figsize: Figure size
        save: Optional filename to save figure
    """
    if not SCANPY_AVAILABLE:
        raise ImportError("scanpy is required for this visualization")
    
    # Default color map
    if color_map is None:
        color_map = {
            'stable': '#2E7D32',        # Green
            'plastic': '#FFA726',       # Orange
            'unstable': '#E53935',      # Red
            'deeply_stable': '#1565C0'  # Blue
        }
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sc.pl.embedding(
        adata,
        basis=basis,
        color=regime_key,
        palette=color_map,
        ax=ax,
        show=False,
        title="Operator Regimes"
    )
    
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_stability_depth_map(
    adata,
    basis: str = "umap",
    metric_key: str = "lambda_min_minus",
    figsize: Tuple[float, float] = (8, 6),
    cmap: str = "viridis_r",
    save: Optional[str] = None
):
    """
    Plot stability depth (λ_min⁻) as a continuous heatmap.
    
    Args:
        adata: AnnData object with atlas results
        basis: Embedding to use
        metric_key: Key in adata.obs for stability depth
        figsize: Figure size
        cmap: Colormap
        save: Optional filename to save figure
    """
    if not SCANPY_AVAILABLE:
        raise ImportError("scanpy is required for this visualization")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sc.pl.embedding(
        adata,
        basis=basis,
        color=metric_key,
        cmap=cmap,
        ax=ax,
        show=False,
        title="Stability Depth (λ_min⁻)"
    )
    
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_plasticity_map(
    adata,
    basis: str = "umap",
    metric_key: str = "plasticity",
    figsize: Tuple[float, float] = (8, 6),
    cmap: str = "YlOrRd",
    save: Optional[str] = None
):
    """
    Plot plasticity index as a continuous heatmap.
    
    Args:
        adata: AnnData object with atlas results
        basis: Embedding to use
        metric_key: Key in adata.obs for plasticity
        figsize: Figure size
        cmap: Colormap
        save: Optional filename to save figure
    """
    if not SCANPY_AVAILABLE:
        raise ImportError("scanpy is required for this visualization")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sc.pl.embedding(
        adata,
        basis=basis,
        color=metric_key,
        cmap=cmap,
        ax=ax,
        show=False,
        title="Plasticity Index"
    )
    
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_nonredundancy_comparison(
    adata,
    celltype_key: str,
    condition_key: str,
    regime_key: str = "operator_regime",
    figsize: Tuple[float, float] = (12, 8),
    save: Optional[str] = None
):
    """
    Plot non-redundancy comparison: same cell type, different operator regimes.
    
    This is the critical figure for demonstrating that operator regimes provide
    information beyond expression-based cell types.
    
    Args:
        adata: AnnData object with atlas results
        celltype_key: Key in adata.obs for cell type labels
        condition_key: Key in adata.obs for condition labels
        regime_key: Key in adata.obs for regime labels
        figsize: Figure size
        save: Optional filename to save figure
    """
    celltypes = adata.obs[celltype_key].values
    conditions = adata.obs[condition_key].values
    regimes = adata.obs[regime_key].values
    
    unique_celltypes = np.unique(celltypes)
    unique_conditions = np.unique(conditions)
    
    fig, axes = plt.subplots(
        len(unique_celltypes),
        len(unique_conditions),
        figsize=figsize,
        squeeze=False
    )
    
    regime_order = ['stable', 'plastic', 'unstable', 'deeply_stable']
    colors = ['#2E7D32', '#FFA726', '#E53935', '#1565C0']
    
    for i, ct in enumerate(unique_celltypes):
        for j, cond in enumerate(unique_conditions):
            ax = axes[i, j]
            
            mask = (celltypes == ct) & (conditions == cond)
            if mask.sum() == 0:
                ax.axis('off')
                continue
            
            ct_cond_regimes = regimes[mask]
            
            # Count regimes
            regime_counts = []
            for regime in regime_order:
                count = (ct_cond_regimes == regime).sum()
                regime_counts.append(count)
            
            # Plot bar chart
            ax.bar(regime_order, regime_counts, color=colors)
            ax.set_title(f"{ct}\n{cond}", fontsize=10)
            ax.set_ylabel("Cell count")
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_temporal_evolution(
    adata,
    pseudotime_key: str = "pseudotime",
    metric_keys: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (12, 8),
    n_bins: int = 20,
    save: Optional[str] = None
):
    """
    Plot operator metrics along pseudotime.
    
    Identifies bifurcation points and commitment transitions.
    
    Args:
        adata: AnnData object with atlas results
        pseudotime_key: Key in adata.obs for pseudotime
        metric_keys: List of metric keys to plot
        figsize: Figure size
        n_bins: Number of pseudotime bins
        save: Optional filename to save figure
    """
    if metric_keys is None:
        metric_keys = ['lambda_max_plus', 'lambda_min_minus', 'plasticity', 'stable_dim']
    
    pseudotime = adata.obs[pseudotime_key].values
    
    # Bin pseudotime
    bins = np.linspace(pseudotime.min(), pseudotime.max(), n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    fig, axes = plt.subplots(len(metric_keys), 1, figsize=figsize, sharex=True)
    if len(metric_keys) == 1:
        axes = [axes]
    
    for ax, metric_key in zip(axes, metric_keys):
        metric_values = adata.obs[metric_key].values
        
        # Compute mean and std in each bin
        bin_means = []
        bin_stds = []
        
        for i in range(n_bins):
            mask = (pseudotime >= bins[i]) & (pseudotime < bins[i+1])
            if mask.sum() > 0:
                bin_means.append(metric_values[mask].mean())
                bin_stds.append(metric_values[mask].std())
            else:
                bin_means.append(np.nan)
                bin_stds.append(np.nan)
        
        bin_means = np.array(bin_means)
        bin_stds = np.array(bin_stds)
        
        # Plot
        ax.plot(bin_centers, bin_means, 'o-', linewidth=2)
        ax.fill_between(
            bin_centers,
            bin_means - bin_stds,
            bin_means + bin_stds,
            alpha=0.3
        )
        
        ax.set_ylabel(metric_key)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel("Pseudotime")
    plt.tight_layout()
    
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_regime_statistics(
    regime_stats: Dict,
    figsize: Tuple[float, float] = (10, 6),
    save: Optional[str] = None
):
    """
    Plot summary statistics for each operator regime.
    
    Args:
        regime_stats: Dictionary from atlas.get_regime_statistics()
        figsize: Figure size
        save: Optional filename to save figure
    """
    regimes = list(regime_stats.keys())
    metrics = ['lambda_max_mean', 'lambda_min_mean', 'plasticity_mean', 'stable_dim_mean']
    metric_labels = ['λ_max⁺', 'λ_min⁻', 'Plasticity', 'Stable Dim']
    
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    
    colors = {
        'stable': '#2E7D32',
        'plastic': '#FFA726',
        'unstable': '#E53935',
        'deeply_stable': '#1565C0'
    }
    
    for ax, metric, label in zip(axes, metrics, metric_labels):
        means = [regime_stats[r][metric] for r in regimes]
        stds = [regime_stats[r][metric.replace('_mean', '_std')] for r in regimes]
        
        bar_colors = [colors.get(r, 'gray') for r in regimes]
        
        ax.bar(regimes, means, yerr=stds, color=bar_colors, alpha=0.7)
        ax.set_ylabel(label)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_combined_overview(
    adata,
    basis: str = "umap",
    celltype_key: Optional[str] = None,
    figsize: Tuple[float, float] = (16, 12),
    save: Optional[str] = None
):
    """
    Create a combined overview figure with multiple panels.
    
    Panels:
    1. Operator regimes
    2. Cell types (if provided)
    3. Stability depth map
    4. Plasticity map
    
    Args:
        adata: AnnData object with atlas results
        basis: Embedding to use
        celltype_key: Optional key for cell type labels
        figsize: Figure size
        save: Optional filename to save figure
    """
    if not SCANPY_AVAILABLE:
        raise ImportError("scanpy is required for this visualization")
    
    n_panels = 4 if celltype_key else 3
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Panel 1: Operator regimes
    color_map = {
        'stable': '#2E7D32',
        'plastic': '#FFA726',
        'unstable': '#E53935',
        'deeply_stable': '#1565C0'
    }
    
    sc.pl.embedding(
        adata,
        basis=basis,
        color='operator_regime',
        palette=color_map,
        ax=axes[0],
        show=False,
        title="Operator Regimes"
    )
    
    # Panel 2: Cell types (if provided)
    if celltype_key:
        sc.pl.embedding(
            adata,
            basis=basis,
            color=celltype_key,
            ax=axes[1],
            show=False,
            title="Cell Types"
        )
    else:
        axes[1].axis('off')
    
    # Panel 3: Stability depth
    sc.pl.embedding(
        adata,
        basis=basis,
        color='lambda_min_minus',
        cmap='viridis_r',
        ax=axes[2],
        show=False,
        title="Stability Depth (λ_min⁻)"
    )
    
    # Panel 4: Plasticity
    sc.pl.embedding(
        adata,
        basis=basis,
        color='plasticity',
        cmap='YlOrRd',
        ax=axes[3],
        show=False,
        title="Plasticity Index"
    )
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    
    plt.show()
