"""
Standard figure panels for Schrödinger Bridge analysis.
All functions read from adata.uns[key] populated by sjd.tl.fit_bridge.
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


_ARCH_COLORS = ["#E63946", "#2A9D8F", "#E9C46A", "#457B9D",
                "#8338EC", "#F4A261", "#264653", "#E76F51"]


def _get(adata, key, field):
    if key not in adata.uns:
        raise KeyError(f"'{key}' not in adata.uns. Run sjd.tl.fit_bridge first.")
    return adata.uns[key][field]


# ---------------------------------------------------------------------------
# Panel: PCA + trajectories
# ---------------------------------------------------------------------------

def bridge_trajectories(
    adata,
    key: str = "scjdo_bridge",
    basis: str = "X_pca",
    direction: str = "both",
    n_show: int = 30,
    color: str = "pseudotime",
    ax=None,
    save: Optional[str] = None,
):
    """
    PCA scatter colored by pseudotime with forward and/or backward
    bridge trajectories overlaid as path lines.

    Parameters
    ----------
    direction : 'forward', 'backward', or 'both'.
    n_show    : Number of trajectory paths to draw (subset for clarity).
    color     : 'pseudotime' or any column in adata.obs for cell coloring.
    """
    src_mask    = _get(adata, key, "src_mask")
    tgt_mask    = _get(adata, key, "tgt_mask")
    fwd_traj_2d = _get(adata, key, "fwd_traj_2d")   # (n_traj, steps+1, 2)
    bwd_traj_2d = _get(adata, key, "bwd_traj_2d")

    if basis not in adata.obsm:
        raise KeyError(f"'{basis}' not in adata.obsm.")
    embed = adata.obsm[basis][:, :2]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 5))

    # ── Scatter cells ──────────────────────────────────────────────────────
    if color == "pseudotime" and "pseudotime" in adata.obs.columns:
        sc = ax.scatter(embed[:, 0], embed[:, 1],
                        c=adata.obs["pseudotime"].values, cmap="plasma",
                        s=7, alpha=0.75, linewidths=0, zorder=1)
        plt.colorbar(sc, ax=ax, label="Pseudotime", fraction=0.046, pad=0.04)
    elif color in adata.obs.columns:
        cats  = adata.obs[color].astype(str).values
        uniq  = sorted(set(cats))
        cmap  = plt.cm.get_cmap("tab20", len(uniq))
        ci    = {c: i for i, c in enumerate(uniq)}
        cols  = [cmap(ci[c]) for c in cats]
        ax.scatter(embed[:, 0], embed[:, 1], c=cols, s=7, alpha=0.75,
                   linewidths=0, zorder=1)
    else:
        ax.scatter(embed[:, 0], embed[:, 1], s=7, alpha=0.4, color="#aaa", zorder=1)

    # Highlight source and target
    ax.scatter(embed[src_mask, 0], embed[src_mask, 1], s=18, alpha=0.9,
               color="steelblue", edgecolors="white", lw=0.3,
               label=f"Source ({src_mask.sum()})", zorder=2)
    ax.scatter(embed[tgt_mask, 0], embed[tgt_mask, 1], s=18, alpha=0.9,
               color="tomato", edgecolors="white", lw=0.3,
               label=f"Target ({tgt_mask.sum()})", zorder=2)

    # ── Overlay trajectories ───────────────────────────────────────────────
    n_show = min(n_show, fwd_traj_2d.shape[0])

    if direction in ("forward", "both"):
        for t in range(n_show):
            path = fwd_traj_2d[t]              # (steps+1, 2)
            ax.plot(path[:, 0], path[:, 1],
                    lw=0.8, alpha=0.45, color="steelblue", zorder=3)
            ax.scatter(*path[-1], s=10, color="tomato",    zorder=4)

    if direction in ("backward", "both"):
        for t in range(n_show):
            path = bwd_traj_2d[t]
            ax.plot(path[:, 0], path[:, 1],
                    lw=0.8, alpha=0.45, color="tomato", zorder=3)
            ax.scatter(*path[-1], s=10, color="steelblue", zorder=4)

    xl = "PC 1" if basis == "X_pca" else "UMAP 1"
    yl = "PC 2" if basis == "X_pca" else "UMAP 2"
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    dir_label = {"forward": "Forward", "backward": "Backward", "both": "Fwd + Bwd"}
    ax.set_title(f"Bridge trajectories ({dir_label.get(direction, direction)})", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")

    if standalone:
        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=300, bbox_inches="tight")
        plt.show()
    return ax


# ---------------------------------------------------------------------------
# Individual panels
# ---------------------------------------------------------------------------

def bridge_source_target(adata, key="scjdo_bridge", basis="X_pca",
                          ax=None, save=None):
    """Scatter: source vs target vs transit cells."""
    src_mask = _get(adata, key, "src_mask")
    tgt_mask = _get(adata, key, "tgt_mask")
    embed    = adata.obsm[basis][:, :2]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))

    transit = ~src_mask & ~tgt_mask
    ax.scatter(embed[transit, 0], embed[transit, 1], s=5, alpha=0.2, color="#ccc", label="Transit")
    ax.scatter(embed[src_mask, 0], embed[src_mask, 1], s=12, alpha=0.8,
               color="steelblue", label=f"Source ({src_mask.sum()})")
    ax.scatter(embed[tgt_mask, 0], embed[tgt_mask, 1], s=12, alpha=0.8,
               color="tomato", label=f"Target ({tgt_mask.sum()})")
    xl = "PC 1" if basis == "X_pca" else "UMAP 1"
    yl = "PC 2" if basis == "X_pca" else "UMAP 2"
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    ax.set_title("Source vs target populations", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")

    if standalone:
        plt.tight_layout()
        if save: plt.savefig(save, dpi=300, bbox_inches="tight")
        plt.show()
    return ax


def bridge_instability(adata, key="scjdo_bridge", ax=None, save=None):
    """Instability curves: forward vs backward + asymmetry."""
    t_vals      = _get(adata, key, "t_vals")
    max_eig_fwd = _get(adata, key, "max_eig_fwd")
    max_eig_bwd = _get(adata, key, "max_eig_bwd")

    standalone = ax is None
    if standalone:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        ax0, ax1  = axes
    else:
        ax0 = ax
        ax1 = None

    ax0.plot(t_vals, max_eig_fwd, color="steelblue", lw=2.5, label="Forward (src→tgt)")
    ax0.plot(t_vals, max_eig_bwd, color="tomato",    lw=2.5, label="Backward (tgt→src)")
    ax0.axhline(0, color="gray", lw=0.8, ls="--")
    ax0.fill_between(t_vals, max_eig_fwd, 0, where=max_eig_fwd > 0, alpha=0.15, color="steelblue")
    ax0.fill_between(t_vals, max_eig_bwd, 0, where=max_eig_bwd > 0, alpha=0.15, color="tomato")
    ax0.set_xlabel("Bridge time t"); ax0.set_ylabel("Max Re(λ)")
    ax0.set_title("Local instability — forward vs backward"); ax0.legend(fontsize=9)

    if ax1 is not None:
        asym = max_eig_fwd - max_eig_bwd
        ax1.bar(t_vals, asym,
                color=["steelblue" if a > 0 else "tomato" for a in asym],
                alpha=0.7, width=(t_vals[1] - t_vals[0]) * 0.8)
        ax1.axhline(0, color="gray", lw=0.8)
        ax1.set_xlabel("Bridge time t")
        ax1.set_ylabel("Fwd − Bwd")
        ax1.set_title("Asymmetry (blue=fwd more sensitive)")

    if standalone:
        plt.tight_layout()
        if save: plt.savefig(save, dpi=300, bbox_inches="tight")
        plt.show()
    return ax0


def bridge_archetypes(adata, key="scjdo_bridge", axes=None, save=None):
    """Archetype activation profiles for both directions."""
    t_vals     = _get(adata, key, "t_vals")
    act_fwd    = _get(adata, key, "act_fwd")
    act_bwd    = _get(adata, key, "act_bwd")
    K          = act_fwd.shape[1]
    colors     = _ARCH_COLORS[:K]
    labels     = [f"A{k+1}" for k in range(K)]

    standalone = axes is None
    if standalone:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    for ax, act, title in [
        (axes[0], act_fwd, "Forward archetypes (src→tgt)"),
        (axes[1], act_bwd, "Backward archetypes (tgt→src)"),
    ]:
        for k in range(K):
            ax.plot(t_vals, act[:, k], color=colors[k], lw=2, label=labels[k])
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.set_xlabel("Bridge time t"); ax.set_ylabel("Norm. activation")
        ax.set_title(title, fontsize=10); ax.legend(ncol=min(K, 4), fontsize=8)

    if standalone:
        plt.tight_layout()
        if save: plt.savefig(save, dpi=300, bbox_inches="tight")
        plt.show()
    return axes


def bridge_genes(adata, key="scjdo_bridge", n_genes=15,
                 axes=None, save=None):
    """Gene × archetype heatmaps for forward and backward."""
    import pandas as pd
    res    = adata.uns[key]
    K      = res["act_fwd"].shape[1]

    standalone = axes is None
    if standalone:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, records, direction in [
        (axes[0], res["df_fwd"], "Forward"),
        (axes[1], res["df_bwd"], "Backward"),
    ]:
        df   = pd.DataFrame(records)
        if df.empty:
            ax.set_title(f"{direction} — no gene data"); continue
        top  = df.groupby("gene")["instability_score"].mean().nlargest(n_genes).index.tolist()
        mat  = np.zeros((len(top), K))
        for ki in range(K):
            sub = df[df["archetype"] == f"A{ki+1}"].set_index("gene")
            for gi, g in enumerate(top):
                if g in sub.index:
                    mat[gi, ki] = sub.loc[g, "instability_score"]
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(K)); ax.set_xticklabels([f"A{k+1}" for k in range(K)])
        ax.set_yticks(range(len(top))); ax.set_yticklabels(top, fontsize=8)
        ax.set_xlabel("Archetype")
        ax.set_title(f"{direction} instability genes", fontsize=10)
        plt.colorbar(im, ax=ax, label="Mean score", fraction=0.03, pad=0.02)

    if standalone:
        plt.tight_layout()
        if save: plt.savefig(save, dpi=300, bbox_inches="tight")
        plt.show()
    return axes


# ---------------------------------------------------------------------------
# Combined summary figure
# ---------------------------------------------------------------------------

def bridge_summary(
    adata,
    key: str = "scjdo_bridge",
    basis: str = "X_pca",
    n_genes: int = 12,
    save: Optional[str] = None,
):
    """
    Seven-panel summary figure for the Schrödinger Bridge analysis:

    (a) PCA + forward trajectories
    (b) PCA + backward trajectories
    (c) Instability curves forward vs backward
    (d) Instability asymmetry
    (e) Training convergence
    (f) Forward archetypes
    (g) Backward archetypes
    """
    fig = plt.figure(figsize=(16, 14))
    gs  = gridspec.GridSpec(4, 2, hspace=0.48, wspace=0.35,
                            left=0.07, right=0.97, top=0.93, bottom=0.06)

    ax_a = fig.add_subplot(gs[0, 0])   # fwd trajectory
    ax_b = fig.add_subplot(gs[0, 1])   # bwd trajectory
    ax_c = fig.add_subplot(gs[1, 0])   # instability curves
    ax_d = fig.add_subplot(gs[1, 1])   # asymmetry
    ax_e = fig.add_subplot(gs[2, 0])   # training curves
    ax_f = fig.add_subplot(gs[2, 1])   # archetype fwd
    ax_g = fig.add_subplot(gs[3, :])   # archetype bwd (full width)

    # (a) PCA + forward trajectories
    bridge_trajectories(adata, key=key, basis=basis, direction="forward",
                        color="pseudotime", ax=ax_a)
    ax_a.set_title("a  |  Forward trajectories (src→tgt)", fontweight="bold",
                   loc="left", fontsize=11)

    # (b) PCA + backward trajectories
    bridge_trajectories(adata, key=key, basis=basis, direction="backward",
                        color="pseudotime", ax=ax_b)
    ax_b.set_title("b  |  Backward trajectories (tgt→src)", fontweight="bold",
                   loc="left", fontsize=11)

    # (c) instability curves
    t_vals      = _get(adata, key, "t_vals")
    max_eig_fwd = _get(adata, key, "max_eig_fwd")
    max_eig_bwd = _get(adata, key, "max_eig_bwd")
    ax_c.plot(t_vals, max_eig_fwd, color="steelblue", lw=2.5, label="Forward")
    ax_c.plot(t_vals, max_eig_bwd, color="tomato",    lw=2.5, label="Backward")
    ax_c.axhline(0, color="gray", lw=0.8, ls="--")
    ax_c.fill_between(t_vals, max_eig_fwd, 0, where=max_eig_fwd > 0, alpha=0.15, color="steelblue")
    ax_c.fill_between(t_vals, max_eig_bwd, 0, where=max_eig_bwd > 0, alpha=0.15, color="tomato")
    ax_c.set_xlabel("Bridge time t"); ax_c.set_ylabel("Max Re(λ)")
    ax_c.legend(fontsize=9)
    ax_c.set_title("c  |  Instability — forward vs backward",
                   fontweight="bold", loc="left", fontsize=11)

    # (d) asymmetry
    asym = max_eig_fwd - max_eig_bwd
    dw   = (t_vals[1] - t_vals[0]) * 0.8
    ax_d.bar(t_vals, asym, color=["steelblue" if a > 0 else "tomato" for a in asym],
             alpha=0.7, width=dw)
    ax_d.axhline(0, color="gray", lw=0.8)
    ax_d.set_xlabel("Bridge time t"); ax_d.set_ylabel("Fwd − Bwd Re(λ)")
    ax_d.set_title("d  |  Instability asymmetry",
                   fontweight="bold", loc="left", fontsize=11)

    # (e) training curves
    history = _get(adata, key, "history")
    ax_e.plot(history["ot_costs"], color="black", lw=1.5)
    ax_e2 = ax_e.twinx()
    ax_e2.plot(history["forward_losses"],  color="steelblue", lw=1.2, ls="--", label="Fwd loss")
    ax_e2.plot(history["backward_losses"], color="tomato",    lw=1.2, ls="--", label="Bwd loss")
    ax_e.set_xlabel("Iteration"); ax_e.set_ylabel("OT cost")
    ax_e2.set_ylabel("Score loss"); ax_e2.legend(fontsize=7, loc="upper right")
    ax_e.set_title("e  |  Training convergence",
                   fontweight="bold", loc="left", fontsize=11)

    # (f/g) archetype profiles — side by side in bottom row
    K      = _get(adata, key, "act_fwd").shape[1]
    colors = _ARCH_COLORS[:K]
    labels = [f"A{k+1}" for k in range(K)]
    for act_key, ax_arch, label in [("act_fwd", ax_f, "f  |  Forward archetypes"),
                                     ("act_bwd", ax_g, "g  |  Backward archetypes")]:
        act = _get(adata, key, act_key)
        for k in range(K):
            ax_arch.plot(t_vals, act[:, k], color=colors[k], lw=2, label=labels[k])
        ax_arch.axhline(0, color="gray", lw=0.5, ls="--")
        ax_arch.set_xlabel("Bridge time t"); ax_arch.set_ylabel("Norm. activation")
        ax_arch.set_title(label, fontweight="bold", loc="left", fontsize=11)
        ax_arch.legend(ncol=min(K, 4), fontsize=8)

    fig.suptitle("Schrödinger Bridge — operator-level instability analysis",
                 fontsize=13, fontweight="bold")

    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")
        print(f"Saved: {save}")
    plt.show()
    return fig


def bridge_gene_comparison(
    adata,
    key: str = "scjdo_bridge",
    n_genes: int = 15,
    save: Optional[str] = None,
):
    """
    Two-panel gene heatmap (forward | backward) plus a Venn-style summary
    of shared vs unique instability drivers.
    """
    import pandas as pd
    res = adata.uns[key]
    df_fwd = pd.DataFrame(res["df_fwd"])
    df_bwd = pd.DataFrame(res["df_bwd"])

    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, n_genes * 0.5)))
    bridge_genes(adata, key=key, n_genes=n_genes, axes=axes)

    # Print comparison
    if not df_fwd.empty and not df_bwd.empty:
        fwd_set = set(df_fwd["gene"])
        bwd_set = set(df_bwd["gene"])
        print(f"Forward-only  ({len(fwd_set - bwd_set)}): "
              f"{', '.join(sorted(fwd_set - bwd_set)[:12])}")
        print(f"Backward-only ({len(bwd_set - fwd_set)}): "
              f"{', '.join(sorted(bwd_set - fwd_set)[:12])}")
        print(f"Shared        ({len(fwd_set & bwd_set)}): "
              f"{', '.join(sorted(fwd_set & bwd_set)[:12])}")

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
        print(f"Saved: {save}")
    plt.show()
    return fig
