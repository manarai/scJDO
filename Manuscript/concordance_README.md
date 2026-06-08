# Concordance experiment — split-notebook workflow

Run each method in its own environment (splicejac downgrades anndata which
breaks the others), then load the saved `.npz` files in the comparison
notebook to compute the T1/T2/T3 concordance tests and produce the figure.

## Workflow

```bash
# ── env A: scJDO ──
conda create -n scjdo-conc python=3.10 -y && conda activate scjdo-conc
pip install scvelo scjdo torch
jupyter nbconvert --to notebook --execute concordance_01_scjdo.ipynb

# ── env B: SpliceJAC ──
conda create -n sjac-conc python=3.10 -y && conda activate sjac-conc
pip install scvelo spliceJAC pyvis plotly
jupyter nbconvert --to notebook --execute concordance_02_splicejac.ipynb

# ── env C: Dynamo ──
conda create -n dyn-conc python=3.10 -y && conda activate dyn-conc
pip install scvelo dynamo-release
jupyter nbconvert --to notebook --execute concordance_03_dynamo.ipynb

# ── env D: compare (env A is fine; no conflict-y deps) ──
conda activate scjdo-conc
jupyter nbconvert --to notebook --execute concordance_04_compare.ipynb
```

## Outputs

`concordance_results/`:
- `shared_metadata.json`           — dataset + parameters used (written by every method nb)
- `scjdo_per_cluster.npz`          — written by 01
- `splicejac_per_cluster.npz`      — written by 02
- `dynamo_per_cluster.npz`         — written by 03

Working directory after running 04:
- `concordance_splicejac_dynamo.pdf` / `.png`  — the 3-panel concordance figure
- `concordance_metrics.json`                   — numbers for the rebuttal letter

## Save schema (per method `.npz`)

| key | shape | description |
|---|---|---|
| `method` | scalar str | `'scjdo'`, `'splicejac'`, or `'dynamo'` |
| `clusters` | (n_clu,) str | cluster names in canonical order |
| `inst_per_cluster` | (n_clu,) float32 | max Re(λ) per cluster (NaN if cluster too small) |
| `gene_names` | (n_gene,) str | gene namespace (identical across methods) |
| `gene_vec` | (n_clu, n_gene) float32 | leading unstable eigenvector projected to gene space |
| `gene_inst_rank` | (n_clu, n_gene) float32 | per-gene \|loading\| magnitudes |
| `top_genes/<cluster>` | (top_k,) str | top-K genes per cluster (one array per cluster) |

The shared preprocessing (load scvelo bonemarrow → filter+normalize → moments
→ diffmap → DPT with deterministic CD34/HSC_1 iroot → cluster labels) is
duplicated verbatim across the three method notebooks. The comparison notebook
asserts that `gene_names` and `clusters` match across the loaded files; if they
don't, it warns and the affected metrics are skipped.
