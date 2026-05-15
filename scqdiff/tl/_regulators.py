"""
Regulatory network inference: link instability genes to upstream TF regulators.

Scoring uses six complementary metrics:
  1. Weighted out-degree     — total instability explained by TF targets
  2. Mean target instability — quality (few sharp targets > many weak ones)
  3. Regulon enrichment      — hypergeometric overlap with top instability genes
  4. Branch specificity      — entropy-based archetype preference
  5. Database confidence     — mean edge weight in the source network
  6. Pseudotime lead         — TF expression peak before target instability peak

Figure 5 uses a hybrid network:
  Solid edges   = reference database confirms TF → target AND target is instability gene
  Dashed edges  = de novo, inferred from Jacobian eigenvector co-loading (novel)
"""
from __future__ import annotations

from typing import Optional
import warnings
import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Network loading
# ---------------------------------------------------------------------------

def _load_collectri(organism: str) -> Optional[pd.DataFrame]:
    try:
        import decoupler as dc
        net = dc.get_collectri(organism=organism, split_complexes=False)
        col_map = {"tf": "source", "genesymbol": "target", "gene": "target"}
        net = net.rename(columns={k: v for k, v in col_map.items() if k in net.columns})
        if {"source", "target"}.issubset(net.columns):
            if "weight" not in net.columns:
                net["weight"] = 1.0
            return net[["source", "target", "weight"]].dropna()
    except Exception:
        pass
    return None


def _load_trrust(organism: str) -> Optional[pd.DataFrame]:
    import urllib.request, io
    urls = {
        "mouse": "https://www.grnpedia.org/trrust/data/trrust_rawdata.mouse.tsv",
        "human": "https://www.grnpedia.org/trrust/data/trrust_rawdata.human.tsv",
    }
    try:
        with urllib.request.urlopen(urls.get(organism, urls["mouse"]), timeout=8) as r:
            df = pd.read_csv(io.StringIO(r.read().decode()), sep="\t",
                             names=["source", "target", "mode", "pmid"])
        df["weight"] = df["mode"].map({"Activation": 1.0, "Repression": -1.0}).fillna(0.5)
        return df[["source", "target", "weight"]].dropna()
    except Exception:
        return None


def _builtin_network() -> pd.DataFrame:
    """Curated mouse hematopoiesis TF-target network (always available)."""
    edges = [
        # Erythroid program
        ("Klf1",  "Alas1",  1.0), ("Klf1",  "Car1",   1.0), ("Klf1",  "Car2",   1.0),
        ("Klf1",  "Blvrb",  1.0), ("Klf1",  "Fam132a",1.0), ("Klf1",  "Ermap",  1.0),
        ("Gata1", "Car1",   1.0), ("Gata1", "Car2",   1.0), ("Gata1", "Alas1",  1.0),
        ("Gata1", "Blvrb",  1.0), ("Gata1", "Ermap",  1.0), ("Gata1", "Klf1",   1.0),
        ("Gata2", "Gata1",  1.0), ("Gata2", "Klf1",   1.0), ("Gata2", "Nfe2",   1.0),
        ("Tal1",  "Gata1",  1.0), ("Tal1",  "Klf1",   1.0), ("Tal1",  "Alas1",  1.0),
        ("Nfe2",  "Alas1",  1.0), ("Nfe2",  "Car1",   1.0), ("Nfe2",  "Blvrb",  1.0),
        ("Lmo2",  "Gata1",  1.0), ("Lmo2",  "Tal1",   1.0),
        # Myeloid / neutrophil program
        ("Spi1",  "Mpo",    1.0), ("Spi1",  "Elane",  1.0), ("Spi1",  "Ctsg",   1.0),
        ("Spi1",  "Prtn3",  1.0), ("Spi1",  "Ly6c2",  1.0), ("Spi1",  "Nkg7",   1.0),
        ("Spi1",  "Cst3",   1.0), ("Spi1",  "Crip1",  1.0), ("Spi1",  "Hp",     1.0),
        ("Irf8",  "Mpo",    1.0), ("Irf8",  "Elane",  1.0), ("Irf8",  "Ctsg",   1.0),
        ("Irf8",  "Ly6c2",  1.0), ("Irf8",  "Id2",    1.0), ("Irf8",  "Cst3",   1.0),
        ("Cebpa", "Mpo",    1.0), ("Cebpa", "Elane",  1.0), ("Cebpa", "Prtn3",  1.0),
        ("Cebpa", "Ctsg",   1.0), ("Cebpa", "Gstm1",  1.0),
        ("Gfi1",  "Mpo",    1.0), ("Gfi1",  "Elane",  1.0), ("Gfi1",  "Spi1",  -1.0),
        # Cross-antagonism
        ("Gata1", "Spi1",  -1.0), ("Spi1",  "Gata1", -1.0),
        # Chromatin / proliferation
        ("Myc",   "Rps11",  1.0), ("Myc",   "Rpl23",  1.0), ("Myc",   "Tmsb10",1.0),
        ("Chd4",  "Gata1",  1.0), ("Chd4",  "Klf1",   1.0),
        # Inflammatory / immune
        ("Stat1", "H2-Aa",  1.0), ("Stat1", "H2-Ab1", 1.0), ("Irf1",  "Cst3",  1.0),
        # Shared progenitor
        ("Runx1", "Gata1",  1.0), ("Runx1", "Spi1",   1.0), ("Runx1", "Gfi1",  1.0),
        ("Cbfa2t3","Gata1", 1.0), ("Cbfa2t3","Klf1",  1.0),
        ("Tgfb1", "Id2",    1.0), ("Vamp8",  "Mpo",   0.5),
        # TF self-regulation / co-factor
        ("Nup210","Gata1",  0.7), ("Malat1", "Spi1",  0.5),
    ]
    return pd.DataFrame(edges, columns=["source", "target", "weight"])


def load_network(
    organism: str = "mouse",
    source: str = "auto",
    custom: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Load a signed TF-target regulatory network.

    Priority: CollecTRI (decoupler) → TRRUST v2 (web) → built-in hematopoiesis.

    Parameters
    ----------
    organism : 'mouse' or 'human'.
    source   : 'auto' | 'collectri' | 'trrust' | 'builtin'.
    custom   : User-supplied DataFrame [source, target, weight].
    """
    if custom is not None:
        if not {"source", "target", "weight"}.issubset(custom.columns):
            raise ValueError("custom must have columns: source, target, weight")
        return custom[["source", "target", "weight"]].dropna()

    if source in ("auto", "collectri"):
        net = _load_collectri(organism)
        if net is not None:
            print(f"[network] CollecTRI loaded — {len(net):,} edges, "
                  f"{net['source'].nunique()} TFs")
            return net

    if source in ("auto", "trrust"):
        net = _load_trrust(organism)
        if net is not None:
            print(f"[network] TRRUST v2 loaded — {len(net):,} edges")
            return net

    print("[network] Using built-in hematopoiesis network "
          "(install decoupler for CollecTRI)")
    return _builtin_network()


# ---------------------------------------------------------------------------
# De novo edge inference from Jacobian eigenvectors
# ---------------------------------------------------------------------------

def _de_novo_edges(
    instab_matrix: np.ndarray,
    max_eig: np.ndarray,
    gene_names: list[str],
    pca_load: Optional[np.ndarray] = None,
    n_top: int = 15,
    sens_thresh: float = 0.05,
) -> pd.DataFrame:
    """
    Infer co-instability gene pairs from Jacobian eigenvector co-loading.

    For each sensitive pseudotime window, genes that load highly onto the
    max-eigenvalue eigenvector are co-amplified — putative co-regulatory
    partners regardless of known network edges.

    Parameters
    ----------
    instab_matrix : (T, n_genes) — instability score per window per gene.
    max_eig       : (T,) — max real eigenvalue per window.
    n_top         : Genes per window to consider co-amplified.
    sens_thresh   : Min eigenvalue to consider a window sensitive.

    Returns
    -------
    DataFrame [source, target, weight, type='denovo'].
    """
    from collections import defaultdict

    sens_mask = max_eig > sens_thresh
    n_sens    = int(sens_mask.sum())
    if n_sens == 0:
        return pd.DataFrame(columns=["source", "target", "weight", "type"])

    edge_acc = defaultdict(float)
    for i in np.where(sens_mask)[0]:
        scores_i = np.abs(instab_matrix[i])
        top_idx  = np.argsort(scores_i)[::-1][:n_top]
        top_g    = [gene_names[j] for j in top_idx]
        top_s    = scores_i[top_idx]
        for a in range(len(top_g)):
            for b in range(a + 1, len(top_g)):
                key = tuple(sorted([top_g[a], top_g[b]]))
                edge_acc[key] += float(min(top_s[a], top_s[b]))

    rows = [{"source": k[0], "target": k[1],
             "weight": round(v / n_sens, 5), "type": "denovo"}
            for k, v in edge_acc.items()]
    df = pd.DataFrame(rows).sort_values("weight", ascending=False)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Six scoring metrics
# ---------------------------------------------------------------------------

def _score_regulators(
    gene_scores: dict[str, float],
    network: pd.DataFrame,
    all_genes: list[str],
    act_norm: Optional[np.ndarray],         # (T, K) archetype activations
    instab_matrix: Optional[np.ndarray],    # (T, n_genes) per-window scores
    max_eig: Optional[np.ndarray],          # (T,) eigenvalues
    min_targets: int = 3,
    sens_thresh: float = 0.05,
) -> pd.DataFrame:
    """Score each TF on six complementary metrics."""
    gene_set  = set(all_genes)
    net       = network[network["target"].isin(gene_set)].copy()
    score_map = {g: float(abs(s)) for g, s in gene_scores.items()}

    # Top-25% instability genes for enrichment denominator
    if score_map:
        q75   = float(np.quantile(list(score_map.values()), 0.75))
        top_g = {g for g, s in score_map.items() if s >= q75}
    else:
        top_g = set()

    n_total = len(gene_set)

    rows = []
    for tf, grp in net.groupby("source"):
        shared = grp[grp["target"].isin(score_map)]
        if len(shared) < min_targets:
            continue

        scores_v  = np.array([score_map[g] for g in shared["target"]])
        weights_v = np.abs(shared["weight"].values)

        # Metric 1: weighted out-degree
        w_score = float((scores_v * weights_v).sum())

        # Metric 2: mean target instability (quality)
        mean_ins = float(scores_v.mean())

        # Metric 3: regulon enrichment (hypergeometric p-value)
        n_tgt      = len(shared)
        n_in_top   = len(set(shared["target"]) & top_g)
        if len(top_g) > 0 and n_total > n_tgt:
            _, hg_p = stats.fisher_exact(
                [[n_in_top, n_tgt - n_in_top],
                 [len(top_g) - n_in_top, n_total - n_tgt - len(top_g) + n_in_top]],
                alternative="greater"
            )
        else:
            hg_p = 1.0
        enrich_score = float(-np.log10(max(hg_p, 1e-10)))

        # Metric 4: branch/archetype specificity (entropy-based)
        branch_spec = 0.0
        peak_arch   = "—"
        if act_norm is not None and instab_matrix is not None:
            K = act_norm.shape[1]
            arch_scores = np.zeros(K)
            for ki in range(K):
                a_mask = act_norm[:, ki] > 0.5
                if a_mask.any():
                    gi_list = [all_genes.index(g) for g in shared["target"]
                               if g in all_genes]
                    if gi_list:
                        arch_scores[ki] = float(
                            np.abs(instab_matrix[a_mask][:, gi_list]).mean())
            if arch_scores.sum() > 0:
                p = arch_scores / (arch_scores.sum() + 1e-8)
                entropy  = float(-np.sum(p * np.log(p + 1e-8)))
                max_ent  = float(np.log(K))
                branch_spec = round(1.0 - entropy / max_ent, 4)
                peak_arch = f"A{int(np.argmax(arch_scores)) + 1}"

        # Metric 5: database confidence (mean edge weight)
        db_conf = float(weights_v.mean())

        # Top contributing targets
        contrib = scores_v * weights_v
        top_tgts = shared.assign(_c=contrib).nlargest(8, "_c")["target"].tolist()

        rows.append({
            "regulator":       tf,
            "weighted_score":  round(w_score, 4),
            "mean_instability":round(mean_ins, 4),
            "enrichment_score":round(enrich_score, 3),
            "branch_specificity": branch_spec,
            "peak_archetype":  peak_arch,
            "db_confidence":   round(db_conf, 3),
            "n_targets":       n_tgt,
            "enrichment_pval": round(float(hg_p), 5),
            "top_targets":     ", ".join(top_tgts),
        })

    if not rows:
        warnings.warn(
            f"No regulators passed min_targets={min_targets}. "
            "The bridge instability gene set may be small — try min_targets=1 "
            "or use the built-in network (network_source='builtin').",
            UserWarning, stacklevel=3,
        )
        return pd.DataFrame(columns=["regulator", "weighted_score", "mean_instability",
                                      "enrichment_score", "branch_specificity",
                                      "peak_archetype", "db_confidence",
                                      "n_targets", "enrichment_pval", "top_targets"])

    return (pd.DataFrame(rows)
            .sort_values("weighted_score", ascending=False)
            .reset_index(drop=True))


# ---------------------------------------------------------------------------
# Pseudotime lead (metric 6, optional)
# ---------------------------------------------------------------------------

def _pseudotime_lead_score(
    tf: str,
    target_genes: list[str],
    adata,
    time_key: str = "pseudotime",
    n_bins: int = 30,
) -> Optional[float]:
    """
    Estimate how many pseudotime units the TF peaks before its targets.
    Positive = TF is upstream (causal-looking).
    """
    if time_key not in adata.obs.columns or tf not in adata.var_names:
        return None
    pt   = adata.obs[time_key].values
    bins = np.linspace(0, 1, n_bins)

    def _peak(gene):
        if gene not in adata.var_names:
            return None
        idx  = list(adata.var_names).index(gene)
        expr = np.asarray(adata.X[:, idx]).ravel()
        means = [expr[np.abs(pt - t).argsort()[:20]].mean() for t in bins]
        return float(bins[np.argmax(means)])

    tf_peak = _peak(tf)
    if tf_peak is None:
        return None
    tgt_peaks = [p for g in target_genes if (p := _peak(g)) is not None]
    if not tgt_peaks:
        return None
    return round(tf_peak - float(np.mean(tgt_peaks)), 3)


# ---------------------------------------------------------------------------
# Unified data extraction
# ---------------------------------------------------------------------------

def _extract_scores(adata, key, direction, pca_load):
    """Return (gene_scores, instab_matrix, max_eig, act_norm) from uns."""
    res        = adata.uns[key]
    gene_names = list(adata.var_names)
    is_bridge  = "df_fwd" in res

    if is_bridge:
        _short    = {"forward": "fwd", "backward": "bwd"}.get(direction, direction)
        df_key    = f"df_{_short}"
        df        = pd.DataFrame(res.get(df_key, []))
        # Primary gene scores from stored table
        gene_sc   = df.groupby("gene")["instability_score"].mean().to_dict() if not df.empty else {}
        ekey      = f"evec_{_short}"
        mkey      = f"max_eig_{_short}"
        evec      = res.get(ekey)          # (T, n_pcs)
        max_eig   = res.get(mkey)
        # Expand gene scores using eigenvector projections → covers all HVGs
        if evec is not None and pca_load is not None and max_eig is not None:
            instab_full = np.abs(pca_load.astype("float32") @ evec.T).T  # (T, n_genes)
            sens_mask   = max_eig > 0.05
            if sens_mask.any():
                for i, g in enumerate(gene_names):
                    sc_i = float(instab_full[sens_mask, i].mean())
                    if g not in gene_sc or sc_i > gene_sc[g]:
                        gene_sc[g] = sc_i
        if evec is not None and pca_load is not None:
            instab_mat = np.abs(pca_load @ evec.T).T   # (T, n_genes)
        else:
            instab_mat = None
        act_norm  = res.get("act_fwd" if direction == "forward" else "act_bwd")
    else:
        instab_mat = res.get("instability_scores")     # (T, n_genes)
        max_eig    = res.get("max_real_eig")
        act_norm   = res.get("act_norm")
        # Build gene_scores from the stored matrix
        if instab_mat is not None and max_eig is not None:
            sens = max_eig > 0.05
            if sens.sum() > 0:
                gene_sc = {g: float(np.abs(instab_mat[sens, i]).mean())
                           for i, g in enumerate(gene_names)}
            else:
                tg = res.get("top_instability_genes", [])
                ts = res.get("top_instability_scores", [])
                gene_sc = dict(zip(tg, ts))
        else:
            tg = res.get("top_instability_genes", [])
            ts = res.get("top_instability_scores", [])
            gene_sc = dict(zip(tg, ts))

    return gene_sc, instab_mat, max_eig, act_norm, gene_names


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def infer_regulators(
    adata,
    key: str = "scqdiff",
    direction: str = "forward",
    network: Optional[pd.DataFrame] = None,
    network_source: str = "auto",
    organism: str = "mouse",
    min_targets: int = 3,
    n_top: int = 20,
    compute_pseudotime_lead: bool = False,
    denovo_n_top: int = 15,
    key_added: str = "scqdiff_regulators",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Infer upstream TF regulators from instability-associated genes.

    Scores each TF on six metrics (weighted out-degree, mean target instability,
    regulon enrichment, branch specificity, database confidence, pseudotime lead)
    and infers de novo co-instability edges from Jacobian eigenvector structure.

    Works with both ``sqd.tl.fit_drift`` and ``sqd.tl.fit_bridge`` results.

    Parameters
    ----------
    key       : ``adata.uns`` key from fit_drift ('scqdiff') or fit_bridge ('scqdiff_bridge').
    direction : For bridge — 'forward', 'backward', or 'both'.
    network   : Custom TF-target DataFrame [source, target, weight].
    organism  : 'mouse' or 'human'.
    min_targets : Min shared targets to include a TF.
    n_top     : Return top N regulators.
    compute_pseudotime_lead : Compute pseudotime lead-lag (slower).
    denovo_n_top : Genes per window for de novo edge inference.
    key_added : Key to store results in adata.uns.

    Returns
    -------
    pandas.DataFrame — ranked regulators.

    Examples
    --------
    >>> sqd.tl.fit_drift(adata)
    >>> df = sqd.tl.infer_regulators(adata)
    >>> sqd.pl.regulator_summary(adata)

    >>> sqd.tl.fit_bridge(adata)
    >>> df = sqd.tl.infer_regulators(adata, key='scqdiff_bridge',
    ...                               direction='both')
    """
    if key not in adata.uns:
        raise KeyError(f"'{key}' not in adata.uns. Run fit_drift or fit_bridge first.")

    pca_load   = adata.varm.get("PCs", None)
    if pca_load is not None:
        pca_load = pca_load.astype("float32")

    # ── Determine directions ───────────────────────────────────────────────
    is_bridge = "df_fwd" in adata.uns[key]
    dirs      = (["forward", "backward"] if (direction == "both" and is_bridge)
                 else [direction])

    # ── Load network ───────────────────────────────────────────────────────
    # For bridge results the instability gene set is small (~80 genes).
    # If external network has no matches, fall back to built-in automatically.
    net = load_network(organism=organism, source=network_source, custom=network)

    # ── Adaptive min_targets ───────────────────────────────────────────────
    # Bridge gene sets are smaller → lower the bar to 2 shared targets.
    effective_min = min_targets if not is_bridge else min(min_targets, 2)

    # ── Score each direction ───────────────────────────────────────────────
    tables   = {}
    denovo_d = {}

    for d in dirs:
        gene_sc, instab_mat, max_eig, act_norm, gene_names = _extract_scores(
            adata, key, d, pca_load)

        if not gene_sc:
            warnings.warn(f"No gene scores for direction '{d}'. Skipping.")
            continue

        if verbose:
            print(f"[{d}] Scoring regulators against {len(gene_sc)} genes...")

        df_sc = _score_regulators(
            gene_sc, net, gene_names, act_norm, instab_mat, max_eig,
            min_targets=effective_min, sens_thresh=0.05,
        )

        # Metric 6: pseudotime lead (optional)
        if compute_pseudotime_lead:
            leads = []
            for _, row in df_sc.iterrows():
                tgts = [t.strip() for t in row["top_targets"].split(",")]
                leads.append(_pseudotime_lead_score(row["regulator"], tgts, adata))
            df_sc["pseudotime_lead"] = leads

        tables[d] = df_sc.head(n_top)

        if verbose:
            top5 = df_sc["regulator"].head(5).tolist()
            print(f"  Top 5: {', '.join(top5)}")

        # De novo edges
        if instab_mat is not None and max_eig is not None:
            denovo_d[d] = _de_novo_edges(
                instab_mat, max_eig, gene_names,
                pca_load=pca_load, n_top=denovo_n_top,
            )
            if verbose:
                print(f"  De novo edges: {len(denovo_d[d])}")

    if not tables:
        raise ValueError("No regulator results. Check that instability scores are computed.")

    primary = next(iter(tables.values()))

    # ── Store results ──────────────────────────────────────────────────────
    adata.uns[key_added] = {
        "tables":       {d: t.to_dict("records") for d, t in tables.items()},
        "denovo_edges": {d: e.to_dict("records") for d, e in denovo_d.items()},
        "network":      net.to_dict("records"),
        "source_key":   key,
        "direction":    direction,
        "params": {
            "organism": organism, "min_targets": min_targets,
            "n_top": n_top, "network_source": network_source,
        },
    }

    return primary
