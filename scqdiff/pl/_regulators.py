"""
Regulatory network figures for scQDiff instability analysis.

Five figures + one summary:
  1. regulator_barplot        — ranked bar chart, colored by mean_instability
  2. regulator_heatmap        — TF × archetype instability heatmap
  3. regulator_scatter        — quality vs quantity (n_targets vs mean_instability)
  4. regulator_profiles       — target instability across pseudotime for top TFs
  5. regulator_network        — hybrid graph: solid = reference, dashed = de novo
  0. regulator_summary        — 4-panel combined (panels 1-4)
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import warnings

def _first_not_none(d, *keys):
    """Return first key in d whose value is not None."""
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return None



_ARCH_COLORS = ["#E63946", "#2A9D8F", "#E9C46A", "#457B9D",
                "#8338EC", "#F4A261", "#264653", "#E76F51"]


def _get(adata, key, field, default=None):
    d = adata.uns.get(key, {})
    return d.get(field, default)


def _table(adata, key="scqdiff_regulators", direction=None) -> pd.DataFrame:
    tables = _get(adata, key, "tables", {})
    if not tables:
        raise KeyError(f"No regulator tables in adata.uns['{key}']. "
                       "Run sqd.tl.infer_regulators first.")
    if direction and direction in tables:
        return pd.DataFrame(tables[direction])
    # Return first available
    return pd.DataFrame(next(iter(tables.values())))


# ---------------------------------------------------------------------------
# Figure 1 — Ranked bar chart
# ---------------------------------------------------------------------------

def regulator_barplot(
    adata,
    key: str = "scqdiff_regulators",
    direction: Optional[str] = None,
    n_show: int = 20,
    ax=None,
    save: Optional[str] = None,
):
    """
    Horizontal bar chart of top regulators.

    Bar length = weighted_score (primary rank).
    Bar color  = mean_instability (darker = higher quality targets).
    Dot        = n_targets (secondary axis).
    """
    df = _table(adata, key, direction).head(n_show)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, max(4, n_show * 0.35)))

    # Color by mean_instability
    norm  = plt.Normalize(df["mean_instability"].min(), df["mean_instability"].max())
    cmap  = plt.cm.YlOrRd
    colors = [cmap(norm(v)) for v in df["mean_instability"]]

    y     = np.arange(len(df))[::-1]
    bars  = ax.barh(y, df["weighted_score"], color=colors, edgecolor="white", lw=0.5)

    # Dot for n_targets on twin axis
    ax2   = ax.twiny()
    ax2.scatter(df["n_targets"], y, color="steelblue", s=40, zorder=5,
                label="n_targets", marker="D", alpha=0.8)
    ax2.set_xlabel("n targets", color="steelblue", fontsize=9)
    ax2.tick_params(axis="x", colors="steelblue", labelsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(df["regulator"], fontsize=9)
    ax.set_xlabel("Weighted instability score", fontsize=9)
    ax.set_title("Top regulatory TFs by instability score", fontsize=10)

    # Annotate peak archetype
    for i, (_, row) in enumerate(df.iterrows()):
        arch = row.get("peak_archetype", "")
        if arch and arch != "—":
            arch_k = int(arch[1:]) - 1
            col    = _ARCH_COLORS[arch_k % len(_ARCH_COLORS)]
            ax.text(row["weighted_score"] * 0.02, y[i], arch,
                    va="center", fontsize=7, color=col, fontweight="bold")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Mean target instability", fraction=0.03, pad=0.12)

    if standalone:
        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=300, bbox_inches="tight")
        plt.show()
    return ax


# ---------------------------------------------------------------------------
# Figure 2 — TF × Archetype heatmap
# ---------------------------------------------------------------------------

def regulator_heatmap(
    adata,
    key: str = "scqdiff_regulators",
    scqdiff_key: Optional[str] = None,
    direction: Optional[str] = None,
    n_show: int = 15,
    ax=None,
    save: Optional[str] = None,
):
    """
    Heatmap: rows = top TFs, columns = archetypes,
    values = mean instability of TF targets when that archetype is active.
    """
    df       = _table(adata, key, direction).head(n_show)
    src_key  = scqdiff_key or _get(adata, key, "source_key", "scqdiff")
    res      = adata.uns.get(src_key, {})
    act_norm = _first_not_none(res, "act_norm", "act_fwd")
    net_rec  = _get(adata, key, "network", [])
    net      = pd.DataFrame(net_rec)
    gene_names = list(adata.var_names)

    if act_norm is None or net.empty:
        warnings.warn("Cannot build heatmap — missing archetype activations.")
        return ax

    K = act_norm.shape[1]
    instab_mat = (lambda v: v if v is not None else _build_instab_mat(res, adata))(res.get("instability_scores"))

    mat = np.zeros((len(df), K))
    for ki in range(K):
        a_mask = act_norm[:, ki] > 0.5
        if not a_mask.any() or instab_mat is None:
            continue
        for ri, (_, row) in enumerate(df.iterrows()):
            tgts = [t.strip() for t in row["top_targets"].split(",")]
            idxs = [i for i, g in enumerate(gene_names) if g in tgts]
            if idxs:
                mat[ri, ki] = float(np.abs(instab_mat[a_mask][:, idxs]).mean())

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(max(5, K * 1.2), max(4, n_show * 0.45)))

    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(K))
    ax.set_xticklabels([f"A{k+1}" for k in range(K)], fontsize=9)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["regulator"], fontsize=9)
    ax.set_xlabel("Archetype"); ax.set_title("TF instability score per archetype", fontsize=10)
    plt.colorbar(im, ax=ax, label="Mean |instability|", fraction=0.03, pad=0.02)

    if standalone:
        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=300, bbox_inches="tight")
        plt.show()
    return ax


def _build_instab_mat(res, adata):
    """Reconstruct instability matrix from bridge eigenvectors if needed."""
    pca_load = adata.varm.get("PCs", None)
    for prefix in ("fwd", "bwd", ""):
        evec_key = f"evec_{prefix}" if prefix else None
        evec = res.get(evec_key) if evec_key else None
        if evec is not None and pca_load is not None:
            return np.abs(pca_load.astype("float32") @ evec.T).T
    return None


# ---------------------------------------------------------------------------
# Figure 3 — Quality vs quantity scatter
# ---------------------------------------------------------------------------

def regulator_scatter(
    adata,
    key: str = "scqdiff_regulators",
    direction: Optional[str] = None,
    n_label: int = 10,
    ax=None,
    save: Optional[str] = None,
):
    """
    Scatter: n_targets (X) vs mean_instability (Y).
    Dot size = weighted_score. Dot color = peak archetype.
    Reveals whether top regulators earn their rank through breadth or quality.
    """
    df = _table(adata, key, direction)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 5))

    sizes = 80 + 400 * (df["weighted_score"] - df["weighted_score"].min()) / \
            (df["weighted_score"].max() - df["weighted_score"].min() + 1e-8)

    for _, row in df.iterrows():
        arch = row.get("peak_archetype", "A1")
        k    = (int(arch[1:]) - 1) if arch and arch != "—" else 0
        col  = _ARCH_COLORS[k % len(_ARCH_COLORS)]
        ax.scatter(row["n_targets"], row["mean_instability"],
                   s=float(sizes[df.index.get_loc(row.name)]),
                   color=col, alpha=0.7, edgecolors="white", lw=0.5)

    # Label top n_label regulators
    top = df.head(n_label)
    for _, row in top.iterrows():
        ax.annotate(row["regulator"],
                    xy=(row["n_targets"], row["mean_instability"]),
                    xytext=(4, 2), textcoords="offset points", fontsize=7)

    ax.set_xlabel("n targets in dataset"); ax.set_ylabel("Mean target instability")
    ax.set_title("Regulator quality vs breadth\n"
                 "(size = weighted score; color = peak archetype)", fontsize=10)

    # Archetype legend
    arch_labels = sorted({row.get("peak_archetype", "—")
                          for _, row in df.iterrows() if row.get("peak_archetype") != "—"})
    handles = [mpatches.Patch(color=_ARCH_COLORS[(int(a[1:])-1) % len(_ARCH_COLORS)], label=a)
               for a in arch_labels]
    if handles:
        ax.legend(handles=handles, title="Peak arch.", fontsize=7, loc="upper right")

    if standalone:
        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=300, bbox_inches="tight")
        plt.show()
    return ax


# ---------------------------------------------------------------------------
# Figure 4 — Target instability profiles across pseudotime
# ---------------------------------------------------------------------------

def regulator_profiles(
    adata,
    key: str = "scqdiff_regulators",
    scqdiff_key: Optional[str] = None,
    direction: Optional[str] = None,
    n_tfs: int = 3,
    ax=None,
    save: Optional[str] = None,
):
    """
    For each of the top n_tfs regulators: line plot of their top target genes'
    instability scores across pseudotime, plus a shaded band for archetype activation.
    """
    df       = _table(adata, key, direction).head(n_tfs)
    src_key  = scqdiff_key or _get(adata, key, "source_key", "scqdiff")
    res      = adata.uns.get(src_key, {})
    t_np     = _first_not_none(res, "t_centers", "t_vals")
    instab_mat = (lambda v: v if v is not None else _build_instab_mat(res, adata))(res.get("instability_scores"))
    act_norm   = _first_not_none(res, "act_norm", "act_fwd")
    gene_names = list(adata.var_names)

    if t_np is None or instab_mat is None:
        warnings.warn("Cannot build profiles — missing pseudotime or instability matrix.")
        return ax

    standalone = ax is None
    if standalone:
        fig, axes = plt.subplots(n_tfs, 1, figsize=(9, 3 * n_tfs),
                                 sharex=True, gridspec_kw={"hspace": 0.45})
        if n_tfs == 1:
            axes = [axes]
    else:
        axes = [ax]

    cmap_t = plt.cm.get_cmap("tab10", 8)

    for ti, (_, row) in enumerate(df.iterrows()):
        if ti >= len(axes):
            break
        axi   = axes[ti]
        tgts  = [t.strip() for t in row["top_targets"].split(",")][:6]

        for ci, gene in enumerate(tgts):
            if gene not in gene_names:
                continue
            gi     = gene_names.index(gene)
            scores = instab_mat[:, gi].copy()
            axi.plot(t_np, scores, color=cmap_t(ci), lw=1.8,
                     label=gene, alpha=0.85)

        # Shade archetype activation
        if act_norm is not None:
            arch = row.get("peak_archetype", "—")
            if arch and arch != "—":
                k   = int(arch[1:]) - 1
                col = _ARCH_COLORS[k % len(_ARCH_COLORS)]
                act = act_norm[:, k]
                axi.fill_between(t_np, 0, act * np.abs(instab_mat).max() * 0.3,
                                 alpha=0.12, color=col,
                                 label=f"{arch} active")

        axi.axhline(0, color="gray", lw=0.5, ls="--")
        axi.set_ylabel("Instability score")
        axi.set_title(f"{row['regulator']}  —  top targets  "
                      f"(weighted score={row['weighted_score']:.3f}, "
                      f"peak arch.={row.get('peak_archetype','—')})", fontsize=9)
        axi.legend(ncol=6, fontsize=7, loc="upper right")

    axes[-1].set_xlabel("Pseudotime")

    if standalone:
        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=300, bbox_inches="tight")
        plt.show()
    return axes


# ---------------------------------------------------------------------------
# Figure 5 — Hybrid network graph
# ---------------------------------------------------------------------------

def regulator_network(
    adata,
    key: str = "scqdiff_regulators",
    scqdiff_key: Optional[str] = None,
    direction: Optional[str] = None,
    n_tfs: int = 5,
    n_targets: int = 6,
    n_denovo: int = 4,
    ax=None,
    save: Optional[str] = None,
):
    """
    Hybrid regulatory network graph.

    Solid edges  = reference database confirms TF → target AND target is a top
                   instability gene (known biology, validated by scQDiff).
    Dashed edges = de novo co-instability edges inferred from Jacobian eigenvector
                   co-loading — not in any reference database (novel finding).

    Node color: TF nodes by peak archetype; target nodes by instability score.
    Node size:  TF by weighted_score; target by instability score.
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("pip install networkx to use regulator_network()")

    df       = _table(adata, key, direction).head(n_tfs)
    net_rec  = _get(adata, key, "network", [])
    dn_rec   = _get(adata, key, "denovo_edges", {})
    src_key  = scqdiff_key or _get(adata, key, "source_key", "scqdiff")
    res      = adata.uns.get(src_key, {})
    gene_names = list(adata.var_names)

    ref_net  = pd.DataFrame(net_rec)
    dn_dir   = direction or "forward"
    dn_net   = pd.DataFrame(dn_rec.get(dn_dir, dn_rec.get(list(dn_rec.keys())[0], [])) if dn_rec else [])

    # Collect TF and target nodes
    tf_list  = df["regulator"].tolist()
    tgt_set  = set()
    for _, row in df.iterrows():
        tgts = [t.strip() for t in row["top_targets"].split(",")][:n_targets]
        tgt_set.update(tgts)
    tgt_list = list(tgt_set)

    # Build instability score lookup
    instab_sc = {}
    instab_mat = (lambda v: v if v is not None else _build_instab_mat(res, adata))(res.get("instability_scores"))
    _dn_short  = {"forward": "fwd", "backward": "bwd"}.get(dn_dir, dn_dir)
    max_eig    = _first_not_none(res, "max_real_eig", f"max_eig_{_dn_short}")
    if instab_mat is not None and max_eig is not None:
        sens = max_eig > 0.05
        for i, g in enumerate(gene_names):
            if sens.any():
                instab_sc[g] = float(np.abs(instab_mat[sens, i]).mean())

    # Build graph
    G = nx.DiGraph()

    # Add TF nodes
    tf_scores = {row["regulator"]: row["weighted_score"]
                 for _, row in df.iterrows()}
    tf_arch   = {row["regulator"]: row.get("peak_archetype", "A1")
                 for _, row in df.iterrows()}
    for tf in tf_list:
        G.add_node(tf, node_type="tf", score=tf_scores.get(tf, 0.1),
                   arch=tf_arch.get(tf, "A1"))

    # Add target nodes
    for g in tgt_list:
        G.add_node(g, node_type="target", score=instab_sc.get(g, 0.01))

    # Reference edges (solid) — TF → target where both in graph
    ref_edges  = []
    if not ref_net.empty:
        ref_sub = ref_net[ref_net["source"].isin(tf_list) &
                          ref_net["target"].isin(tgt_list)]
        for _, row in ref_sub.iterrows():
            G.add_edge(row["source"], row["target"],
                       edge_type="reference", weight=float(row["weight"]))
            ref_edges.append((row["source"], row["target"]))

    # De novo edges (dashed) — co-instability pairs NOT in reference
    dn_edges = []
    if not dn_net.empty:
        ref_tgt_pairs = {(r["source"], r["target"]) for _, r in ref_net.iterrows()
                         } if not ref_net.empty else set()
        dn_sub = dn_net[
            (dn_net["source"].isin(tf_list + tgt_list)) &
            (dn_net["target"].isin(tf_list + tgt_list))
        ].head(n_denovo)
        for _, row in dn_sub.iterrows():
            if (row["source"], row["target"]) not in ref_tgt_pairs:
                if not G.has_edge(row["source"], row["target"]):
                    G.add_node(row["source"],
                               node_type="target",
                               score=instab_sc.get(row["source"], 0.01))
                    G.add_node(row["target"],
                               node_type="target",
                               score=instab_sc.get(row["target"], 0.01))
                    G.add_edge(row["source"], row["target"],
                               edge_type="denovo", weight=float(row["weight"]))
                    dn_edges.append((row["source"], row["target"]))

    # ── Layout ────────────────────────────────────────────────────────────
    # TFs on left, targets on right
    pos = {}
    for i, tf in enumerate(tf_list):
        pos[tf] = (-1.5, 1.0 - i * (2.0 / max(len(tf_list) - 1, 1)))
    all_tgts = [n for n in G.nodes() if G.nodes[n]["node_type"] == "target"]
    for i, tg in enumerate(all_tgts):
        pos[tg] = (1.5, 1.0 - i * (2.0 / max(len(all_tgts) - 1, 1)))

    # ── Draw ──────────────────────────────────────────────────────────────
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, max(6, len(all_tgts) * 0.5)))

    ax.set_xlim(-2.5, 2.8); ax.set_ylim(-1.4, 1.4); ax.axis("off")

    # TF nodes
    tf_node_sizes  = [max(300, tf_scores.get(n, 0.1) * 3000) for n in tf_list if n in pos]
    tf_node_colors = []
    for n in tf_list:
        if n not in pos:
            continue
        arch = tf_arch.get(n, "A1")
        k    = (int(arch[1:]) - 1) if arch and arch != "—" else 0
        tf_node_colors.append(_ARCH_COLORS[k % len(_ARCH_COLORS)])
    nx.draw_networkx_nodes(G, pos, nodelist=[n for n in tf_list if n in pos],
                           node_size=tf_node_sizes, node_color=tf_node_colors,
                           node_shape="s", ax=ax, alpha=0.9)

    # Target nodes (color = instability score)
    tgt_nodes  = [n for n in all_tgts if n in pos]
    tgt_scores = [instab_sc.get(n, 0.01) for n in tgt_nodes]
    sc_norm    = plt.Normalize(0, max(tgt_scores) if tgt_scores else 1)
    tgt_colors = [plt.cm.YlOrRd(sc_norm(s)) for s in tgt_scores]
    tgt_sizes  = [max(150, s * 2000) for s in tgt_scores]
    nx.draw_networkx_nodes(G, pos, nodelist=tgt_nodes,
                           node_size=tgt_sizes, node_color=tgt_colors,
                           ax=ax, alpha=0.9)

    # Labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold", ax=ax)

    # Reference edges — solid
    if ref_edges:
        nx.draw_networkx_edges(G, pos, edgelist=ref_edges,
                               edge_color="dimgray", width=1.8,
                               arrows=True, arrowsize=15,
                               connectionstyle="arc3,rad=0.1",
                               ax=ax, style="solid")

    # De novo edges — dashed
    if dn_edges:
        nx.draw_networkx_edges(G, pos, edgelist=dn_edges,
                               edge_color="#2196F3", width=1.5,
                               arrows=True, arrowsize=12,
                               connectionstyle="arc3,rad=-0.2",
                               ax=ax, style="dashed")

    # ── Legend ────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(color="dimgray",   label="Reference edge (solid)"),
        mpatches.Patch(color="#2196F3",   label="De novo co-instability (dashed)"),
        mpatches.Patch(color="lightgray", label="TF node (square, color = archetype)"),
        mpatches.Patch(color=plt.cm.YlOrRd(0.8), label="Target node (circle, color = instability)"),
    ]
    ax.legend(handles=legend_items, loc="lower left", fontsize=7, framealpha=0.9)
    ax.set_title("Regulatory network — reference (solid) + de novo (dashed)",
                 fontsize=10, pad=8)

    if standalone:
        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=300, bbox_inches="tight")
        plt.show()
    return ax


# ---------------------------------------------------------------------------
# Summary figure — 4 panels (bar + heatmap + scatter + profiles)
# ---------------------------------------------------------------------------

def regulator_summary(
    adata,
    key: str = "scqdiff_regulators",
    scqdiff_key: Optional[str] = None,
    direction: Optional[str] = None,
    n_show: int = 15,
    save: Optional[str] = None,
):
    """
    Four-panel summary figure:
    (a) Ranked bar chart       (b) TF × Archetype heatmap
    (c) Quality vs quantity    (d) Target instability profiles (top 3 TFs)
    """
    fig = plt.figure(figsize=(16, 13))
    gs  = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35,
                            left=0.07, right=0.97, top=0.93, bottom=0.07)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])

    # Panel d: 3 sub-rows for target profiles
    gs_d = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1, 1], hspace=0.5)
    axes_d = [fig.add_subplot(gs_d[i]) for i in range(3)]

    kw = dict(key=key, direction=direction)
    regulator_barplot(adata, **kw, n_show=n_show,         ax=ax_a)
    regulator_heatmap(adata, **kw, scqdiff_key=scqdiff_key,
                      n_show=n_show,                       ax=ax_b)
    regulator_scatter(adata, **kw, n_label=10,             ax=ax_c)
    regulator_profiles(adata, **kw, scqdiff_key=scqdiff_key,
                       n_tfs=3,                             ax=axes_d[0])

    for i, axi in enumerate(axes_d):
        axi.set_visible(i == 0)   # only first sub-row used by regulator_profiles
    # Manually call profiles for all 3 sub-rows
    df3 = _table(adata, key, direction).head(3)
    src_key  = scqdiff_key or _get(adata, key, "source_key", "scqdiff")
    res      = adata.uns.get(src_key, {})
    t_np     = _first_not_none(res, "t_centers", "t_vals")
    instab_mat = (lambda v: v if v is not None else _build_instab_mat(res, adata))(res.get("instability_scores"))
    act_norm   = _first_not_none(res, "act_norm", "act_fwd")
    gene_names = list(adata.var_names)
    cmap_t     = plt.cm.get_cmap("tab10", 8)

    for axi in axes_d:
        axi.set_visible(True)

    for ti, (_, row) in enumerate(df3.iterrows()):
        axi   = axes_d[ti]
        tgts  = [t.strip() for t in row["top_targets"].split(",")][:5]
        if t_np is not None and instab_mat is not None:
            for ci, gene in enumerate(tgts):
                if gene not in gene_names:
                    continue
                gi = gene_names.index(gene)
                axi.plot(t_np, instab_mat[:, gi], color=cmap_t(ci),
                         lw=1.5, label=gene, alpha=0.85)
            if act_norm is not None:
                arch = row.get("peak_archetype", "—")
                if arch and arch != "—":
                    k   = int(arch[1:]) - 1
                    col = _ARCH_COLORS[k % len(_ARCH_COLORS)]
                    act = act_norm[:, k]
                    scale = np.abs(instab_mat).max() * 0.3
                    axi.fill_between(t_np, 0, act * scale,
                                     alpha=0.1, color=col)
        axi.axhline(0, color="gray", lw=0.5, ls="--")
        axi.set_ylabel("Score", fontsize=7)
        axi.set_title(f"{row['regulator']} targets", fontsize=8)
        axi.legend(ncol=5, fontsize=6, loc="upper right")
        axi.tick_params(labelsize=7)
    axes_d[-1].set_xlabel("Pseudotime", fontsize=8)

    for ax, label in zip([ax_a, ax_b, ax_c], ["a", "b", "c"]):
        ax.set_title(f"{label}  |  " + ax.get_title(),
                     fontweight="bold", loc="left", fontsize=10)

    fig.suptitle("Regulatory network inference — instability-driven TF ranking",
                 fontsize=12, fontweight="bold")

    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")
        print(f"Saved: {save}")
    plt.show()
    return fig
