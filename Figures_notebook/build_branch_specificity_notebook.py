"""
Build Figures_notebook/Figure6_multiome_branch_specificity_validation.ipynb.

Source of truth: edit cell content here and re-run to regenerate the .ipynb.

The notebook delivers two pieces:
  Piece 2 — Archetype-resolved peak reconciliation on scjdo_cc_ExcNeuron.
  Piece 1 — Branch-specificity test for neurogenic vs housekeeping regulons.
"""
from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf


CELLS: list = []


def md(text: str):
    CELLS.append(nbf.v4.new_markdown_cell(text))


def code(src: str):
    CELLS.append(nbf.v4.new_code_cell(src))


# ─────────────────────────────────────────────────────────────────────────
md(r"""# Figure 6 — Branch-specificity validation and archetype peak reconciliation

**Purpose.** Decide whether the multiome section can claim
"lineage-specific chromatin support for neurogenic commitment", or whether
the chromatin column should drop to a supporting role. Also resolve the
contradiction between branch-level `peak_t≈0.97` (a likely boundary
artefact) and the archetype-level neurogenic peak at interior pseudotime.

**Inputs.**

* `adata_multiome_drift_cc.h5ad` — produced by `Figure6_multiome_drift_cellcycle_regressed.ipynb`,
  containing CC-corrected `scjdo_cc_<branch>` results for at least
  ExcNeuron plus one non-neurogenic branch (InhNeuron, CR, or any other).
* `adata_multiome_fa_atac.h5ad` + `features.tsv.gz` — for the proximal
  peak → gene mapping and the log-TF-IDF accessibility track.

**Deliverables.**

1. **Piece 2 — archetype dissection** (Section A): per archetype $k$ in
   ExcNeuron, the activation-peak pseudotime, the effective sample size
   at that peak, and the top driving genes.  Flags archetypes whose peak
   sits in the trajectory boundary $\tau<0.05$ or $\tau>0.95$ with low
   $n_\mathrm{eff}$ — those are boundary artefacts and should not be
   reported as biology.
2. **Piece 1 — branch-specific regulon test** (Section B): the same
   $r(\text{TF RNA}, \text{regulon ATAC})$ correlation you ran for Sp1,
   computed for neurogenic TFs $\{$Eomes, Neurog2, Neurod1, Neurod2, Pax6$\}$
   and housekeeping controls $\{$Sp1, Nfkb1$\}$, in **both** the ExcNeuron
   branch and an unrelated branch.  Outputs a small **TF × branch** table
   and a heatmap so the lineage-specificity is visible at a glance.

**Pre-committed read for Piece 1.**

* Neurogenic TFs $r$ high in ExcNeuron and low in the comparator branch → genuine,
  lineage-specific chromatin support.  Report as validation.
* Neurogenic TFs $r$ high in both branches, or low everywhere → no
  branch-specific chromatin coupling.  Chromatin column drops to
  *suggestive*; the RNA-level Eomes/Neurog2 instability result stands on
  its own.

Generic TFs (Sp1, Nfkb1) are negative controls and are expected to
correlate everywhere — they exist to demonstrate that high $r$ alone is
uninformative.
""")


# ─────────────────────────────────────────────────────────────────────────
code(r"""
import os, sys, warnings, bisect
sys.path.insert(0, os.path.abspath('..'))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import scanpy as sc
import scjdo as sjd

mpl.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'font.family':  'DejaVu Sans'})

# ── Paths ─────────────────────────────────────────────────────────────────
DRIFT_DIR    = 'results/figure6_multiome_fa_drift_cc/'
ATAC_DIR     = 'results/figure6_multiome_fa/'
FEATURES_TSV = ('/Users/terooatt/Documents/Project_scQDiff/02_scQDiff/'
                'scIDIFF_anndata/data/multiomic/filtered_feature_bc_matrix/'
                'features.tsv.gz')

OUTDIR = 'results/figure6_branch_specificity/'
os.makedirs(OUTDIR, exist_ok=True)

# ── Inputs ────────────────────────────────────────────────────────────────
DRIFT_ADATA = DRIFT_DIR + 'adata_multiome_drift_cc.h5ad'
ATAC_ADATA  = ATAC_DIR  + 'adata_multiome_fa_atac.h5ad'

assert os.path.exists(DRIFT_ADATA), f'missing {DRIFT_ADATA} — run the CC-regressed drift notebook first'
assert os.path.exists(ATAC_ADATA),  f'missing {ATAC_ADATA}  — run the FA-integration notebook first'

ad      = sc.read_h5ad(DRIFT_ADATA)
ad_atac = sc.read_h5ad(ATAC_ADATA)
if not ad_atac.obs_names.equals(ad.obs_names):
    ad_atac = ad_atac[ad.obs_names].copy()

cc_keys = sorted(k for k in ad.uns.keys()
                 if k.startswith('scjdo_cc_')
                 and isinstance(ad.uns[k], dict)
                 and 'J_tensor' in ad.uns[k])
branches_cc = [k.replace('scjdo_cc_', '') for k in cc_keys]
print(f'RNA  : {ad.n_obs} cells × {ad.n_vars} genes')
print(f'ATAC : {ad_atac.n_obs} cells × {ad_atac.n_vars} peaks')
print(f'CC-corrected branches available: {branches_cc}')

# Pick the neurogenic branch + a non-neurogenic comparator. Defaults can be
# overridden manually below.
NEURO_BRANCH   = 'ExcNeuron' if 'ExcNeuron' in branches_cc else branches_cc[0]
def _pick_comparator():
    prefs = ['InhNeuron', 'CR', 'Astrocyte', 'OPC', 'Oligo']
    for b in prefs:
        if b in branches_cc and b != NEURO_BRANCH:
            return b
    for b in branches_cc:
        if b != NEURO_BRANCH:
            return b
    return None
COMPARATOR_BRANCH = _pick_comparator()
print(f'Neurogenic branch  : {NEURO_BRANCH}')
print(f'Comparator branch  : {COMPARATOR_BRANCH or "(none — only one branch present, specificity test cannot run)"}')
""")


# ─────────────────────────────────────────────────────────────────────────
md(r"""## Section A — Piece 2: archetype-resolved peak reconciliation

For the neurogenic branch we extract every archetype's activation curve,
locate its peak in pseudotime, look up the effective sample size $n_\mathrm{eff}$
at that peak (when the kernel scheme stored it), and rank the top driving
genes via the stored `gene_scores`.  Archetypes whose peak sits in the
boundary $\tau<0.05$ or $\tau>0.95$ and whose $n_\mathrm{eff}$ is low are
flagged as boundary artefacts — the global `peak_t` reported at the
branch level may be one of these and should be reported separately.""")
code(r"""
def archetype_profile_table(key: str, top_n_genes: int = 20):
    res        = ad.uns[key]
    activ      = np.asarray(res['activations'])             # (T, K)
    t_c        = np.asarray(res['t_centers'])               # (T,)
    K          = activ.shape[1]
    n_eff      = np.asarray(res.get('n_eff', [np.nan]*len(t_c)))
    gene_names = res.get('gene_names') or list(ad.var_names)
    gene_scores= res.get('gene_scores', {}) or {}
    rows = []
    for k in range(K):
        ak      = activ[:, k]
        pk      = int(np.argmax(ak))
        peak_t  = float(t_c[pk])
        neff_pk = float(n_eff[pk]) if pk < len(n_eff) else float('nan')
        is_boundary = peak_t < 0.05 or peak_t > 0.95
        # Top genes
        scores = gene_scores.get(str(k))
        if scores is None:
            top_genes = []
        else:
            order = np.argsort(-np.abs(scores))[:top_n_genes]
            top_genes = [gene_names[j] for j in order]
        rows.append({
            'archetype':       f'A{k+1}',
            'peak_t':          peak_t,
            'n_eff_at_peak':   neff_pk,
            'is_boundary':     is_boundary,
            'activation_amp':  float(ak.max()),
            'top_genes':       ', '.join(top_genes[:10]),
            'top_genes_full':  top_genes,
        })
    return pd.DataFrame(rows)


arche_df = archetype_profile_table(f'scjdo_cc_{NEURO_BRANCH}', top_n_genes=20)
print(f'\nArchetype dissection — scjdo_cc_{NEURO_BRANCH}')
print(arche_df.drop(columns=['top_genes_full']).to_string(index=False,
       float_format=lambda v: f'{v:.4f}'))

arche_df_save = arche_df.drop(columns=['top_genes_full'])
arche_df_save.to_csv(OUTDIR + f'archetype_table_{NEURO_BRANCH}.csv', index=False)
print(f'\nSaved: {OUTDIR}archetype_table_{NEURO_BRANCH}.csv')

# Look for the neurogenic-commitment archetype: any A_k with peak_t in
# [0.10, 0.40] whose top genes include Eomes or Neurog2.
NEURO_GENES = {'Eomes','Neurog2','Neurog1','Neurod1','Neurod2','Tbr2'}
candidate = None
for _, row in arche_df.iterrows():
    hits = NEURO_GENES.intersection(set(row['top_genes_full'][:20]))
    if hits and 0.10 <= row['peak_t'] <= 0.40:
        candidate = row['archetype']; hit_set = hits; break
if candidate:
    print(f'\nNeurogenic-commitment archetype: {candidate}  '
          f'(neurogenic genes in top-20: {sorted(hit_set)})')
else:
    print('\nNo archetype satisfies (peak in [0.10,0.40]) ∧ (Eomes/Neurog2 in top-20).')
""")


code(r"""
# Plot activations + n_eff curve so the boundary artefact (if any) is visible.
res = ad.uns[f'scjdo_cc_{NEURO_BRANCH}']
activ = np.asarray(res['activations'])
t_c   = np.asarray(res['t_centers'])
neff  = np.asarray(res.get('n_eff', np.full(len(t_c), np.nan)))
K     = activ.shape[1]
fig, axes = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True,
                         gridspec_kw=dict(height_ratios=[2.4, 1]))
cmap = plt.cm.tab10(np.linspace(0, 1, K))
for k in range(K):
    label = f'A{k+1}'
    if candidate == label:
        label += '  (neurogenic candidate)'
    axes[0].plot(t_c, activ[:, k], color=cmap[k], lw=2.0, label=label)
axes[0].set_ylabel('activation $c_k(\\tau)$')
axes[0].set_title(f'Archetype activations — scjdo_cc_{NEURO_BRANCH}')
axes[0].axvspan(0.0, 0.05, color='gray', alpha=0.10)
axes[0].axvspan(0.95, 1.0, color='gray', alpha=0.10)
axes[0].legend(fontsize=8, ncol=2)

axes[1].plot(t_c, neff, color='k', lw=1.4)
axes[1].axhline(30, color='gray', ls=':', lw=0.8, label='n_eff_min=30')
axes[1].axvspan(0.0, 0.05, color='gray', alpha=0.10)
axes[1].axvspan(0.95, 1.0, color='gray', alpha=0.10)
axes[1].set_xlabel('pseudotime $\\tau$'); axes[1].set_ylabel('$n_\\mathrm{eff}(\\tau)$')
axes[1].legend(fontsize=8, loc='upper center')
plt.tight_layout()
plt.savefig(OUTDIR + f'archetype_activations_{NEURO_BRANCH}.pdf')
plt.show()

# Report the global peak in max_real_eig + its n_eff — the artefact diagnostic.
eig = np.asarray(res['max_real_eig'])
gpk = int(np.argmax(eig))
print(f'\nGlobal max-Re(λ) peak: τ={t_c[gpk]:.3f}   λ={eig[gpk]:.4f}   '
      f'n_eff={neff[gpk]:.1f}')
if t_c[gpk] > 0.95 or t_c[gpk] < 0.05:
    print('  → in boundary; report as edge effect, not biology.')
""")


# ─────────────────────────────────────────────────────────────────────────
md(r"""## Section B — Piece 1: branch-specific regulon ATAC validation

Build target gene sets for a small panel of TFs from the chosen
regulatory network (CollecTRI by default; falls back to TRRUST or the
built-in network).  We deliberately include **neurogenic** TFs
(Eomes, Neurog2, Neurod1, Neurod2, Pax6) and **housekeeping** controls
(Sp1, Nfkb1).  The housekeeping controls are expected to correlate in
both branches — they exist to demonstrate that a high $r$ alone is
uninformative.

For each `(TF, branch)` pair we compute $r$ between

* the TF's RNA expression along that branch's pseudotime, and
* the mean log-TF-IDF accessibility of all peaks within $\pm$`WINDOW_KB`
  of the TSS of the TF's target genes, again along the same pseudotime.

Both signals are binned into the same `N_BINS` along pseudotime so they
share a temporal axis.  The decisive test is whether the neurogenic TFs
show high $r$ in the neurogenic branch only.""")
code(r"""
# Config
TFs_NEUROGENIC    = ['Eomes', 'Neurog2', 'Neurog1', 'Neurod1', 'Neurod2', 'Pax6']
TFs_HOUSEKEEPING  = ['Sp1',   'Nfkb1']                 # expected to correlate everywhere
ALL_TFs           = TFs_NEUROGENIC + TFs_HOUSEKEEPING

WINDOW_KB         = 100                                # ±100 kb of TSS for proximal peaks
N_BINS            = 30
ORGANISM          = 'mouse'
MIN_TARGETS       = 10                                 # below this we will flag regulons as too small

# Load network
net = sjd.tl.load_network(organism=ORGANISM, source='auto')
print(f'Network: {len(net)} edges, {net["source"].nunique()} unique TFs')

# Build feature -> coords table
feat = pd.read_csv(FEATURES_TSV, sep='\t', header=None,
                   names=['feature_id','feature_name','feature_type','chrom','start','end'])
gene_coords = (feat[feat.feature_type == 'Gene Expression']
               .drop_duplicates('feature_name')
               .set_index('feature_name'))
peak_coords = (feat[feat.feature_type == 'Peaks']
               .drop_duplicates('feature_id')
               .set_index('feature_id'))
peak_coords = peak_coords.loc[peak_coords.index.intersection(ad_atac.var_names)]

peaks_by_chrom = {}
for chrom, sub in peak_coords.groupby('chrom'):
    centers = ((sub['start'].values + sub['end'].values) / 2).astype(np.int64)
    order = np.argsort(centers)
    peaks_by_chrom[chrom] = {'names': sub.index.values[order],
                              'centers': centers[order]}


def proximal_peaks(gene_name: str, kb: int = WINDOW_KB) -> list:
    if gene_name not in gene_coords.index:
        return []
    row = gene_coords.loc[gene_name]
    d = peaks_by_chrom.get(row['chrom'])
    if d is None:
        return []
    tss, w = int(row['start']), kb * 1000
    lo = bisect.bisect_left( d['centers'], tss - w)
    hi = bisect.bisect_right(d['centers'], tss + w)
    return list(d['names'][lo:hi])

print(f'gene_coords: {len(gene_coords)}  ·  peaks indexed: {sum(len(d["names"]) for d in peaks_by_chrom.values())}')
""")


code(r"""
# Pre-built helpers (mirror the original Figure 6 drift validation code)
peak_name_to_col = pd.Series(np.arange(ad_atac.n_vars), index=ad_atac.var_names)
log_tfidf        = ad_atac.layers['log_tfidf']

# ATAC ↔ RNA row alignment (the multiome objects may have different cell counts)
_atac_row_for_rna = (pd.Series(np.arange(ad_atac.n_obs), index=ad_atac.obs_names)
                     .reindex(ad.obs_names).values.astype('float64'))
_present = ~np.isnan(_atac_row_for_rna)
_rows    = np.where(_present, _atac_row_for_rna, 0).astype(np.int64)
print(f'ATAC↔RNA aligned cells: {_present.sum()}/{ad.n_obs}')


def atac_score_for_peaks(peak_names: list) -> np.ndarray:
    # Per-RNA-cell mean log-TF-IDF over the requested peaks (zero where unmatched).
    if not peak_names:
        return np.zeros(ad.n_obs, dtype=np.float32)
    cols = peak_name_to_col.reindex(peak_names).dropna().astype(int).values
    if len(cols) == 0:
        return np.zeros(ad.n_obs, dtype=np.float32)
    X = log_tfidf[:, cols]
    vec = np.asarray(X.mean(axis=1)).ravel().astype(np.float32)   # (ad_atac.n_obs,)
    aligned = vec[_rows]
    aligned[~_present] = 0.0
    return aligned


def rna_vec(gene: str) -> np.ndarray:
    if gene not in ad.var_names:
        return None
    j = ad.var_names.get_loc(gene)
    if 'MAGIC_imputed_data' in ad.layers:
        X = ad.layers['MAGIC_imputed_data'][:, j]
    else:
        X = ad.X[:, j]
    X = X.toarray().ravel() if sp.issparse(X) else np.asarray(X).ravel()
    return X.astype(np.float32)


def bin_along_pt(values, pseudotime, mask, n_bins=N_BINS):
    vals = values[mask]
    pt   = pseudotime[mask]
    edges = np.linspace(0, 1, n_bins + 1)
    idx   = np.clip(np.digitize(pt, edges[1:-1]), 0, n_bins - 1)
    means = np.full(n_bins, np.nan)
    for i in range(n_bins):
        sub = vals[idx == i]
        if len(sub):
            means[i] = float(np.mean(sub))
    return 0.5 * (edges[:-1] + edges[1:]), means


def regulon_targets(tf: str) -> list:
    sub = net[net['source'] == tf]
    return sorted(set(sub['target']))


def regulon_peaks(tf: str) -> tuple:
    # Return (targets present in RNA, union of proximal peaks).
    targets = [g for g in regulon_targets(tf) if g in ad.var_names]
    peaks = []
    for g in targets:
        peaks.extend(proximal_peaks(g))
    return targets, sorted(set(peaks))
""")


code(r"""
# Per-branch pseudotime + mask retrieval
def branch_pt_and_mask(branch: str):
    # Pseudotime: use CC-pseudotime if present, else the original Palantir output.
    pt_col = 'palantir_pseudotime_cc' if 'palantir_pseudotime_cc' in ad.obs.columns else 'palantir_pseudotime'
    if pt_col not in ad.obs.columns:
        raise KeyError(f'No pseudotime column found in adata.obs (looked for {pt_col})')
    pt   = ad.obs[pt_col].values.astype(float)
    pt   = (pt - np.nanmin(pt)) / max(np.nanmax(pt) - np.nanmin(pt), 1e-9)

    # Branch mask: prefer branch_masks_cc, then branch_masks, then cell_fate.
    mask = None
    for obsm_key in ('branch_masks_cc', 'branch_masks'):
        if obsm_key in ad.obsm and branch in ad.obsm[obsm_key].columns:
            mask = ad.obsm[obsm_key][branch].values.astype(bool); break
    if mask is None:
        for fate_col in ('cell_fate_cc', 'cell_fate'):
            if fate_col in ad.obs.columns:
                mask = (ad.obs[fate_col] == branch).values; break
    if mask is None:
        # final fallback: branch_cells from scjdo result
        idx = np.array(ad.uns[f'scjdo_cc_{branch}'].get('branch_cells', []))
        mask = np.zeros(ad.n_obs, dtype=bool); mask[idx] = True
    return pt, mask


def regulon_r(tf: str, branch: str) -> dict:
    targets, peaks = regulon_peaks(tf)
    tf_expr = rna_vec(tf)
    if tf_expr is None:
        return dict(TF=tf, branch=branch, r=np.nan, n_targets=len(targets),
                    n_peaks=len(peaks), reason='TF not in var_names')
    pt, mask = branch_pt_and_mask(branch)
    if mask.sum() < 20:
        return dict(TF=tf, branch=branch, r=np.nan, n_targets=len(targets),
                    n_peaks=len(peaks), reason='branch too small')
    if len(peaks) < 5:
        return dict(TF=tf, branch=branch, r=np.nan, n_targets=len(targets),
                    n_peaks=len(peaks), reason='regulon too small')
    atac = atac_score_for_peaks(peaks)
    _, tf_b   = bin_along_pt(tf_expr, pt, mask)
    _, atac_b = bin_along_pt(atac,    pt, mask)
    valid = ~(np.isnan(tf_b) | np.isnan(atac_b))
    if valid.sum() < 5:
        return dict(TF=tf, branch=branch, r=np.nan, n_targets=len(targets),
                    n_peaks=len(peaks), reason='not enough non-empty bins')
    r = float(np.corrcoef(tf_b[valid], atac_b[valid])[0, 1])
    return dict(TF=tf, branch=branch, r=r, n_targets=len(targets),
                n_peaks=len(peaks), reason='ok')


rows = []
branches_to_test = [NEURO_BRANCH] + ([COMPARATOR_BRANCH] if COMPARATOR_BRANCH else [])
for tf in ALL_TFs:
    for br in branches_to_test:
        rows.append(regulon_r(tf, br))

res_df = pd.DataFrame(rows)
res_df['flag'] = np.where(res_df['n_targets'] < MIN_TARGETS, 'few targets', '')
print('\nBranch-specific regulon ATAC validation:')
cols = ['TF','branch','r','n_targets','n_peaks','flag','reason']
print(res_df[cols].to_string(index=False, float_format=lambda v: f'{v:+.3f}' if pd.notna(v) else '   nan'))
res_df.to_csv(OUTDIR + 'branch_specificity_regulons.csv', index=False)
print(f'\nSaved: {OUTDIR}branch_specificity_regulons.csv')
""")


code(r"""
# Build the small TF × branch heatmap that decides the chromatin claim.
if COMPARATOR_BRANCH is None:
    print('Only one branch present — specificity heatmap skipped.')
else:
    pivot = res_df.pivot(index='TF', columns='branch', values='r')
    pivot = pivot.reindex(ALL_TFs)
    fig, ax = plt.subplots(figsize=(4.5, 0.45*len(ALL_TFs)+1.3))
    vmax = float(np.nanmax(np.abs(pivot.values))) if pivot.notna().any().any() else 1.0
    im = ax.imshow(pivot.values, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels(pivot.columns, rotation=0)
    ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            txt = '' if np.isnan(v) else f'{v:+.2f}'
            ax.text(j, i, txt, ha='center', va='center',
                    color='white' if abs(v) > 0.5 else 'black', fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.85, label='Pearson r')
    ax.set_title('TF RNA  vs  regulon ATAC  (binned along pseudotime)')
    # Visual divider between neurogenic and housekeeping rows
    ax.axhline(len(TFs_NEUROGENIC) - 0.5, color='black', lw=0.8)
    plt.tight_layout(); plt.savefig(OUTDIR + 'branch_specificity_heatmap.pdf'); plt.show()

    # Pre-committed read
    print('\nPre-committed read:')
    sub = res_df.set_index(['TF','branch']).r.unstack('branch')
    for tf in TFs_NEUROGENIC:
        if tf not in sub.index: continue
        r_n = sub.loc[tf, NEURO_BRANCH]
        r_c = sub.loc[tf, COMPARATOR_BRANCH]
        if pd.isna(r_n) or pd.isna(r_c):
            print(f'  {tf:8s}  r({NEURO_BRANCH})={r_n}  r({COMPARATOR_BRANCH})={r_c}  → insufficient data')
            continue
        if r_n >= 0.3 and r_n - r_c >= 0.2:
            print(f'  {tf:8s}  r({NEURO_BRANCH})={r_n:+.2f}  r({COMPARATOR_BRANCH})={r_c:+.2f}  → branch-SPECIFIC')
        elif r_n >= 0.3 and r_c >= 0.3:
            print(f'  {tf:8s}  r({NEURO_BRANCH})={r_n:+.2f}  r({COMPARATOR_BRANCH})={r_c:+.2f}  → high but non-specific')
        else:
            print(f'  {tf:8s}  r({NEURO_BRANCH})={r_n:+.2f}  r({COMPARATOR_BRANCH})={r_c:+.2f}  → no specific coupling')
""")


# ─────────────────────────────────────────────────────────────────────────
md(r"""## How to report from this output

* **Archetype-resolved peak (Piece 2).** Lead with the interior archetype:
  e.g. *"scJDO resolves a neurogenic commitment archetype A_k peaking at
  pseudotime τ≈X, driven by {top genes}; an additional sensitivity peak
  at the trajectory terminus (τ≈Y) shows low effective sample size
  ($n_\mathrm{eff}$≈Z) and is therefore reported as a boundary effect
  rather than a commitment signal."*

* **Branch-specific chromatin (Piece 1).** Read the heatmap rules above:
  * If ≥ 1 neurogenic TF is **branch-specific** *and* the housekeeping
    controls are *not* — report as same-cell chromatin support.
  * If neurogenic TFs are **non-specific** or **absent** but housekeeping
    is high in both — the chromatin column drops to suggestive; the
    RNA-level Eomes/Neurog2 instability result stands on its own, and
    the housekeeping result is shown as the negative control that
    motivated the specificity test.

In either case, do *not* report the housekeeping TFs (Sp1, Nfkb1) as
support — they are the negative control demonstrating that high $r$
alone is uninformative.
""")


# ─────────────────────────────────────────────────────────────────────────
def main() -> None:
    out = Path(__file__).resolve().parent / "Figure6_multiome_branch_specificity_validation.ipynb"
    nb = nbf.v4.new_notebook()
    nb.cells = CELLS
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    with out.open("w") as f:
        nbf.write(nb, f)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
