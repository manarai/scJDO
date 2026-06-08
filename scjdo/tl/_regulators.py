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

_NETWORK_CACHE: dict = {}   # module-level cache; survives across infer_regulators calls


def _load_collectri(organism: str) -> Optional[pd.DataFrame]:
    cache_key = f"collectri_{organism}"
    if cache_key in _NETWORK_CACHE:
        return _NETWORK_CACHE[cache_key]
    try:
        import decoupler as dc
        # decoupler renamed the public API between 1.x and 2.x. Try both so the
        # large CollecTRI network is actually loaded instead of silently falling
        # back to the much smaller TRRUST/builtin sources.
        loader = getattr(dc, "get_collectri", None)            # decoupler 1.x
        if loader is None:
            loader = getattr(getattr(dc, "op", None),          # decoupler 2.x
                             "collectri", None)
        if loader is None:
            raise AttributeError(
                "Installed decoupler exposes neither dc.get_collectri (1.x) "
                "nor dc.op.collectri (2.x). Upgrade decoupler or pass "
                "network_source='trrust'/'builtin'."
            )
        # decoupler 1.x: split_complexes ; 2.x: remove_complexes (default False
        # in both keeps complexes as single edges). Pass only what the installed
        # version accepts, otherwise the call dies with TypeError and the user
        # silently gets TRRUST/builtin instead of the full 43k-edge CollecTRI.
        import inspect
        sig = set(inspect.signature(loader).parameters)
        kwargs = {"organism": organism}
        if "split_complexes" in sig:
            kwargs["split_complexes"] = False
        elif "remove_complexes" in sig:
            kwargs["remove_complexes"] = False
        net = loader(**kwargs)
        col_map = {"tf": "source", "genesymbol": "target", "gene": "target"}
        net = net.rename(columns={k: v for k, v in col_map.items() if k in net.columns})
        if {"source", "target"}.issubset(net.columns):
            if "weight" not in net.columns:
                net["weight"] = 1.0
            result = net[["source", "target", "weight"]].dropna()
            _NETWORK_CACHE[cache_key] = result
            return result
    except Exception as e:
        # Surface what went wrong rather than silently falling back — silent
        # failure here is exactly what produced 'empty Ery/DC regulator tables'
        # while Mono squeaked through TRRUST.
        warnings.warn(
            f"[regulators] CollecTRI load via decoupler failed "
            f"({type(e).__name__}: {e}); falling back to next source.",
            UserWarning, stacklevel=3,
        )
    _NETWORK_CACHE[cache_key] = None
    return None


def _load_trrust(organism: str) -> Optional[pd.DataFrame]:
    cache_key = f"trrust_{organism}"
    if cache_key in _NETWORK_CACHE:
        return _NETWORK_CACHE[cache_key]
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
        result = df[["source", "target", "weight"]].dropna()
        _NETWORK_CACHE[cache_key] = result
        return result
    except Exception:
        _NETWORK_CACHE[cache_key] = None
        return None


def _builtin_network(organism: str = "mouse") -> pd.DataFrame:
    """Curated hematopoiesis TF-target network (always available).

    Edges are stored with mouse-cased symbols. For ``organism='human'`` they are
    mapped to human symbols so the network gene namespace matches human data
    (otherwise target overlap is zero and every regulator is filtered out).
    Most mouse→human orthologs differ only by case (Gata1→GATA1); the few that
    do not are listed in ``_HUMAN_ORTHOLOG`` below.
    """
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
    net = pd.DataFrame(edges, columns=["source", "target", "weight"])

    if organism == "human":
        # Symbols whose human ortholog is not simply the uppercased mouse name.
        _HUMAN_ORTHOLOG = {
            "H2-Aa": "HLA-DRA", "H2-Ab1": "HLA-DRB1", "Fam132a": "C1QTNF12",
        }
        to_human = lambda g: _HUMAN_ORTHOLOG.get(g, g.upper())
        net["source"] = net["source"].map(to_human)
        net["target"] = net["target"].map(to_human)

    return net


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
            if f"collectri_{organism}_printed" not in _NETWORK_CACHE:
                print(f"[network] CollecTRI loaded — {len(net):,} edges, "
                      f"{net['source'].nunique()} TFs")
                _NETWORK_CACHE[f"collectri_{organism}_printed"] = True
            return net

    if source in ("auto", "trrust"):
        net = _load_trrust(organism)
        if net is not None:
            if f"trrust_{organism}_printed" not in _NETWORK_CACHE:
                print(f"[network] TRRUST v2 loaded — {len(net):,} edges")
                _NETWORK_CACHE[f"trrust_{organism}_printed"] = True
            return net

    if "builtin_printed" not in _NETWORK_CACHE:
        print("[network] Using built-in hematopoiesis network "
              "(install decoupler for CollecTRI)")
        _NETWORK_CACHE["builtin_printed"] = True
    return _builtin_network(organism)


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

    # ── Shared gene universe ────────────────────────────────────────────────
    # Regulon enrichment must be evaluated on ONE consistent background: genes
    # that are both scored in this branch's instability table AND present as
    # targets in the regulator network. Counting targets, top genes, and the
    # total on mismatched universes — or counting duplicate target edges — is
    # what produced 2x2 Fisher tables with negative cells. Intersecting every
    # set with this universe (and treating targets as a set) guarantees a valid
    # contingency table without dropping regulators wholesale.
    universe = set(score_map) & set(net["target"])
    n_total  = len(universe)

    # Top-25% instability genes for the enrichment denominator, restricted to
    # the shared universe so the top set never references genes outside n_total.
    if score_map:
        q75   = float(np.quantile(list(score_map.values()), 0.75))
        top_g = {g for g, s in score_map.items() if s >= q75} & universe
    else:
        top_g = set()
    n_top_g = len(top_g)

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
        # Deduplicate this TF's targets and intersect with the shared universe,
        # then recompute every cell of the 2x2 table from those harmonized sets.
        tf_targets = set(shared["target"]) & universe
        n_tgt      = len(tf_targets)
        n_in_top   = len(tf_targets & top_g)

        a = n_in_top                              # in targets and in top
        b = n_tgt - n_in_top                      # in targets, not in top
        c = n_top_g - n_in_top                    # in top, not in targets
        d = n_total - n_tgt - n_top_g + n_in_top  # in neither

        if min(a, b, c, d) < 0:
            # Should not occur once the sets are harmonized; skip this single
            # regulator rather than silencing every regulator or clipping the
            # (signed) instability scores globally. Emit a debug trace with the
            # raw counts so a recurrence (e.g. Ery/DC) can be diagnosed directly.
            print(
                f"[regulators][skip] '{tf}' invalid 2x2 table — "
                f"n_total={n_total}, n_tgt={n_tgt}, len(top_g)={len(top_g)}, "
                f"n_in_top={n_in_top} (a={a}, b={b}, c={c}, d={d})"
            )
            warnings.warn(
                f"Regulator '{tf}' still yields an invalid 2x2 table after "
                f"universe harmonization (a={a}, b={b}, c={c}, d={d}); "
                f"skipping this regulator.",
                UserWarning, stacklevel=3,
            )
            continue

        if n_top_g > 0 and n_tgt > 0:
            _, hg_p = stats.fisher_exact([[a, b], [c, d]],
                                         alternative="greater")
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

        # Top contributing targets — ranked full list + capped display copy.
        # The capped copy ("top_targets", up to 8) is for human-readable CSVs;
        # the full list ("top_targets_full", all matched targets ranked by
        # contribution) is for downstream pipelines like chromatin validation
        # that should not be silently restricted to 8 targets per TF.
        contrib = scores_v * weights_v
        ranked  = (shared.assign(_c=contrib)
                          .sort_values("_c", ascending=False)["target"].tolist())
        top_tgts = ranked[:8]

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
            "top_targets_full":ranked,                    # NEW — uncapped
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
                                      "n_targets", "enrichment_pval",
                                      "top_targets", "top_targets_full"])

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
    key: str = "scjdo",
    direction: str = "forward",
    network: Optional[pd.DataFrame] = None,
    network_source: str = "auto",
    organism: str = "mouse",
    min_targets: int = 3,
    n_top: int = 20,
    compute_pseudotime_lead: bool = False,
    denovo_n_top: int = 15,
    key_added: str = "scjdo_regulators",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Infer upstream TF regulators from instability-associated genes.

    Scores each TF on six metrics (weighted out-degree, mean target instability,
    regulon enrichment, branch specificity, database confidence, pseudotime lead)
    and infers de novo co-instability edges from Jacobian eigenvector structure.

    Works with both ``sjd.tl.fit_drift`` and ``sjd.tl.fit_bridge`` results.

    Parameters
    ----------
    key       : ``adata.uns`` key from fit_drift ('scjdo') or fit_bridge ('scjdo_bridge').
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
    >>> sjd.tl.fit_drift(adata)
    >>> df = sjd.tl.infer_regulators(adata)
    >>> sjd.pl.regulator_summary(adata)

    >>> sjd.tl.fit_bridge(adata)
    >>> df = sjd.tl.infer_regulators(adata, key='scjdo_bridge',
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


# ---------------------------------------------------------------------------
# Per-branch / per-direction wrapper — encapsulates the loop + uns copy-back
# pattern that was repeated (and bug-prone) across Figure3_FA and Figure5.
# ---------------------------------------------------------------------------

def infer_regulators_branches(
    adata,
    branch_models: dict,
    *,
    direction: str = "primary",
    key_prefix: str = "scjdo",
    regulators_prefix: str = "scjdo_regulators",
    save_csv_dir: Optional[str] = None,
    verbose: bool = True,
    **infer_kwargs,
) -> dict:
    """
    Run ``infer_regulators`` for each branch / perturbation in one call.

    Wraps the per-branch idiom that was hand-written (and silently broken) in
    every prior notebook: subset the AnnData to each branch's cells, call
    ``infer_regulators`` on the subset, and **copy the resulting regulator
    entry back onto the full AnnData's uns** so downstream plotters can find
    it. The copy-back step is the one that was missing from Figure3_FA cells
    13/15 and produced empty regulator plots for every branch.

    Parameters
    ----------
    adata : AnnData
        The full AnnData object containing the per-branch drift / bridge
        results in ``adata.uns[f'{key_prefix}_{name}']``.
    branch_models : dict
        ``{name: model}`` mapping returned by ``fit_drift_branches`` or
        ``fit_bridge_branches``. Only the keys are used (branch names).
    direction : {'primary', 'both', 'forward', 'backward'}
        Which direction(s) to score. ``'primary'`` (default) calls
        ``infer_regulators`` without a direction argument — appropriate for
        drift-branch keys. Use ``'both'`` for bridge keys to get the
        forward+backward tables.
    key_prefix : str
        Prefix used by ``fit_drift_branches`` / ``fit_bridge_branches`` —
        each branch's source key is ``f'{key_prefix}_{name}'``.
    regulators_prefix : str
        Prefix for the regulator-result keys written to ``adata.uns`` —
        each branch's destination key is ``f'{regulators_prefix}_{name}'``.
    save_csv_dir : str, optional
        If set, also write ``{save_csv_dir}/{name}/regulators_{direction}.csv``
        per branch (helpful for downstream plotting / sharing).
    verbose : bool
        Print a one-line summary per branch.
    **infer_kwargs
        Forwarded to :func:`infer_regulators` (e.g. ``organism='human'``,
        ``min_targets=1``, ``n_top=15``, ``network_source='auto'``).

    Returns
    -------
    dict
        ``{branch_name: regulator_df}`` — the primary regulator table per
        branch (forward direction for bridges; the single direction for drift).
        Full per-direction tables are accessible via
        ``adata.uns[f'{regulators_prefix}_{name}']['tables']``.

    Examples
    --------
    Drift branches (one direction):

    >>> models = sjd.tl.fit_drift_branches(adata, branch_key='branch_masks')
    >>> regs   = sjd.tl.infer_regulators_branches(
    ...     adata, models, organism='human', min_targets=1, n_top=15,
    ...     save_csv_dir='results/figure3/')

    Bridge branches (both directions):

    >>> bridges = sjd.tl.fit_bridge_branches(adata, groupby='target',
    ...     src_group='Non-Targeting', tgt_groups=['PVT1','MALAT1'])
    >>> regs    = sjd.tl.infer_regulators_branches(
    ...     adata, bridges, direction='both',
    ...     organism='human', min_targets=1, n_top=15)
    """
    import os
    out = {}

    for name in branch_models:
        src_key = f"{key_prefix}_{name}"
        reg_key = f"{regulators_prefix}_{name}"

        if src_key not in adata.uns:
            warnings.warn(
                f"[{name}] source key '{src_key}' not in adata.uns; skipping. "
                f"Did fit_drift_branches/fit_bridge_branches store under a "
                f"different key_prefix?",
                UserWarning, stacklevel=2,
            )
            continue

        # ── Subset to this branch's cells (drift) or the bridge population
        # (which is already encoded by src_mask/tgt_mask in the bridge uns,
        # so for bridges we don't need to subset — pass the full adata).
        src_uns  = adata.uns[src_key]
        is_bridge = "df_fwd" in src_uns

        if is_bridge:
            # Bridges store the populations inside uns and read directly from
            # adata.varm['PCs'] + adata.var_names — no subset required.
            ad_for_call = adata
        else:
            # Drift branches: re-create the per-branch subset that
            # infer_regulators expects (gene_names / instab_mat aligned).
            import numpy as np
            cell_idx = np.asarray(src_uns.get("branch_cells", []))
            if len(cell_idx) == 0:
                warnings.warn(
                    f"[{name}] uns['{src_key}']['branch_cells'] is empty; "
                    f"running infer_regulators on the full AnnData.",
                    UserWarning, stacklevel=2,
                )
                ad_for_call = adata
            else:
                ad_for_call = adata[cell_idx].copy()
                ad_for_call.uns[src_key] = src_uns

        # ── Inference call ────────────────────────────────────────────────
        try:
            kwargs = dict(infer_kwargs)
            if direction != "primary":
                kwargs["direction"] = direction
            df_primary = infer_regulators(
                ad_for_call, key=src_key, key_added=reg_key, **kwargs,
            )
            # Copy uns back onto the *full* AnnData so plotters can find it.
            # This is the single line whose absence broke every prior notebook.
            if reg_key in ad_for_call.uns:
                adata.uns[reg_key] = ad_for_call.uns[reg_key]
            out[name] = df_primary
        except Exception as e:
            warnings.warn(
                f"[{name}] infer_regulators raised {type(e).__name__}: {e} "
                f"— branch will have no regulator table.",
                UserWarning, stacklevel=2,
            )
            out[name] = pd.DataFrame()

        # ── Optional CSV ──────────────────────────────────────────────────
        if save_csv_dir is not None and len(out[name]):
            branch_dir = os.path.join(save_csv_dir, name)
            os.makedirs(branch_dir, exist_ok=True)
            tables = adata.uns.get(reg_key, {}).get("tables", {})
            if tables:
                for d, recs in tables.items():
                    pd.DataFrame(recs).to_csv(
                        os.path.join(branch_dir, f"regulators_{d}.csv"),
                        index=False,
                    )
            else:
                out[name].to_csv(
                    os.path.join(branch_dir, "regulators_primary.csv"),
                    index=False,
                )

        if verbose:
            top5 = out[name]["regulator"].head(5).tolist() if len(out[name]) else []
            print(f"[{name}] {len(out[name])} regulators  top5={top5}")

    return out


# ---------------------------------------------------------------------------
# Post-hoc regulator filtering — drops promiscuous TFs and re-ranks by a
# composite of weighted_score × branch_specificity. Addresses the failure
# mode where CollecTRI's broadest TFs (SP1, TP53, MYC, NFKB, JUN — each with
# thousands of documented targets) win every enrichment ranking on every
# dataset, drowning out lineage-specific drivers.
# ---------------------------------------------------------------------------

def filter_regulators(
    df_reg: pd.DataFrame,
    *,
    organism: str = "human",
    network_source: str = "auto",
    custom_network: Optional[pd.DataFrame] = None,
    min_regulon_size: int = 50,
    max_regulon_size: int = 500,
    specificity_floor: float = 0.30,
    keep_tfs: Optional[set] = None,
    drop_tfs: Optional[set] = None,
    rank_by: str = "composite",
) -> pd.DataFrame:
    """
    Filter and re-rank a regulator table to surface lineage-specific drivers.

    The default behaviour of :func:`infer_regulators` ranks TFs by
    ``weighted_score`` alone — which is dominated by **promiscuous TFs** in
    any large reference network (SP1 has >2,000 documented targets in
    CollecTRI; TP53 has >1,500; NFKB family similar). On every Perturb-seq,
    every drift-branch, every dataset, these same names appear at the top
    because their target list intersects everything. The branch_specificity
    column is already computed but ignored by the default sort.

    This helper applies three filters in sequence:

    1. **Regulon-size filter.** Drops TFs whose reference-network regulon
       is outside ``[min_regulon_size, max_regulon_size]``. The lower bound
       cuts noise-level TFs with <50 targets; the upper bound cuts the
       promiscuous "regulates everything" set.
    2. **Branch-specificity floor.** Drops TFs with
       ``branch_specificity < specificity_floor``.
    3. **Composite re-ranking** by
       ``weighted_score × (specificity_floor + branch_specificity)``,
       so a TF needs both quantity *and* archetype-specific activation to
       move up the list.

    For tissue-specific analyses (e.g. neurogenesis), pass an allowlist via
    ``keep_tfs`` to restrict to known lineage drivers and avoid relying on
    the database's coverage of relevant TFs.

    Parameters
    ----------
    df_reg : pandas.DataFrame
        Output of :func:`infer_regulators` (one row per TF).
    organism, network_source, custom_network
        Forwarded to :func:`load_network` to fetch regulon sizes. Only used
        when ``min_regulon_size`` / ``max_regulon_size`` are not ``None``.
    min_regulon_size, max_regulon_size : int
        Regulon-size bounds. Pass ``None`` to disable either bound.
    specificity_floor : float
        Minimum ``branch_specificity`` to keep a TF, *and* the additive
        constant in the composite score (so TFs with specificity 0 still
        contribute their ``weighted_score`` × ``specificity_floor``).
    keep_tfs, drop_tfs : set of str, optional
        Allowlist / denylist applied first; case-insensitive match.
    rank_by : {'composite', 'weighted_score', 'branch_specificity'}
        Column to sort by. ``'composite'`` is the new default and what this
        function exists to compute.

    Returns
    -------
    pandas.DataFrame
        Filtered + re-ranked copy of ``df_reg`` with an added ``composite``
        column.

    Examples
    --------
    Generic filter for any single-cell dataset:

    >>> df = sjd.tl.infer_regulators(adata, organism='mouse')
    >>> df_clean = sjd.tl.filter_regulators(df, organism='mouse',
    ...     min_regulon_size=50, max_regulon_size=500,
    ...     specificity_floor=0.30)

    Neurogenesis allowlist (drops everything else):

    >>> NEURO_TFS = {'Pax6','Sox2','Hes1','Hes5','Eomes','Neurog1','Neurog2',
    ...     'Neurod1','Neurod2','Neurod6','Tbr1','Tbr2','Satb2','Bcl11b',
    ...     'Foxp2','Dlx1','Dlx2','Dlx5','Gad1','Gad2','Lhx5','Lhx6'}
    >>> df_neuro = sjd.tl.filter_regulators(df, keep_tfs=NEURO_TFS,
    ...     min_regulon_size=None, max_regulon_size=None,
    ...     specificity_floor=0.0)
    """
    df = df_reg.copy()
    if not len(df) or "regulator" not in df.columns:
        return df

    # ── (a) Allow/deny lists (case-insensitive on the regulator column) ───
    if keep_tfs is not None:
        keep_upper = {t.upper() for t in keep_tfs}
        df = df[df["regulator"].astype(str).str.upper().isin(keep_upper)]
    if drop_tfs is not None:
        drop_upper = {t.upper() for t in drop_tfs}
        df = df[~df["regulator"].astype(str).str.upper().isin(drop_upper)]

    # ── (b) Regulon-size filter — needs the reference network ─────────────
    if (min_regulon_size is not None) or (max_regulon_size is not None):
        net = load_network(organism=organism, source=network_source,
                           custom=custom_network)
        sizes = net.groupby("source")["target"].nunique()
        lo = min_regulon_size if min_regulon_size is not None else 0
        hi = max_regulon_size if max_regulon_size is not None else 10**9
        ok = set(sizes[(sizes >= lo) & (sizes <= hi)].index)
        df = df[df["regulator"].isin(ok)]

    # ── (c) Branch-specificity floor ──────────────────────────────────────
    if "branch_specificity" in df.columns and specificity_floor > 0:
        df = df[df["branch_specificity"] >= specificity_floor]

    if not len(df):
        # Preserve the schema callers expect (so `df[['regulator','composite']]`
        # works on an empty result without raising KeyError).
        df["composite"] = pd.Series(dtype=float)
        return df

    # ── (d) Composite re-ranking ─────────────────────────────────────────
    spec = df["branch_specificity"].fillna(0.0) if "branch_specificity" in df.columns \
           else pd.Series(0.0, index=df.index)
    df["composite"] = (df["weighted_score"].astype(float) *
                       (specificity_floor + spec))
    if rank_by == "composite":
        df = df.sort_values("composite", ascending=False)
    elif rank_by in df.columns:
        df = df.sort_values(rank_by, ascending=False)
    return df.reset_index(drop=True)
