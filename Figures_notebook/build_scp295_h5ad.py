"""One-time SCP295 → AnnData converter.

Reads the gene-major TSV (~2.87 GB gzipped, 19,089 genes × 251,203 cells),
stratifies-samples N_KEEP cells by experimental day, and writes a sparse h5ad
with day, FLE coordinates, and cell-set lineage labels attached.

Output: <DATA>/scp295.h5ad   — ready for Figure4.ipynb.
"""
from __future__ import annotations
import gzip, time, sys, numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad

DATA   = '/Users/terooatt/Documents/Project_scQDiff/02_scQDiff/scIDIFF_anndata/data/SCP295'
N_KEEP = 25_000     # stratified sub-sample (per-day quota)
SEED   = 42

# ── 1. Stratified cell sample by experimental day ──────────────────────────
days = pd.read_csv(f'{DATA}/metadata/cell_days.txt', sep='\t',
                   skiprows=[1], index_col=0)['day'].astype(float)
print(f"[meta] cells with day labels: {len(days):,}")

rng = np.random.default_rng(SEED)
per_day = max(1, N_KEEP // days.nunique())
sample = (days.to_frame('day')
              .groupby('day', group_keys=False)
              .apply(lambda g: g.sample(min(per_day, len(g)), random_state=SEED)))
keep_ids = set(sample.index)
print(f"[sample] {len(keep_ids):,} cells across {days.nunique()} days "
      f"(~{per_day}/day)")

# ── 2. Stream the expression TSV, keep only sampled columns ────────────────
P = f'{DATA}/expression/Ex.Mat.dsmpl.15k.txt.gz'
t0 = time.time()
with gzip.open(P, 'rt') as f:
    header = next(f).rstrip('\n').split('\t')
    all_cells = header[1:]                                # 251,203 cell IDs
    col_idx_keep = np.array([i for i, c in enumerate(all_cells) if c in keep_ids],
                            dtype=np.int32)
    cell_ids_keep = [all_cells[i] for i in col_idx_keep]
    print(f"[expr] header parsed in {time.time()-t0:.1f}s; "
          f"keeping {len(col_idx_keep):,} of {len(all_cells):,} columns")

    genes = []
    row_idx_acc, col_idx_acc, data_acc = [], [], []
    t1 = time.time()
    for gi, line in enumerate(f):
        parts = line.rstrip('\n').split('\t')
        genes.append(parts[0])
        vals = np.asarray(parts[1:], dtype=np.float32)[col_idx_keep]
        nz = np.flatnonzero(vals)
        if nz.size:
            row_idx_acc.append(np.full(nz.size, gi, dtype=np.int32))
            col_idx_acc.append(nz.astype(np.int32))
            data_acc.append(vals[nz])
        if (gi + 1) % 2000 == 0:
            print(f"  {gi+1:>5d} genes parsed ({time.time()-t1:.0f}s elapsed)", flush=True)

print(f"[expr] full parse: {time.time()-t0:.0f}s; {len(genes)} genes")

# ── 3. Build sparse cells × genes (transpose from gene-major) ──────────────
rows = np.concatenate(row_idx_acc)      # gene index
cols = np.concatenate(col_idx_acc)      # cell index (within kept subset)
vals = np.concatenate(data_acc)
n_genes, n_cells = len(genes), len(cell_ids_keep)
X = sp.csr_matrix((vals, (cols, rows)), shape=(n_cells, n_genes))   # cells × genes
print(f"[sparse] X: {X.shape}  nnz={X.nnz:,}  "
      f"density={100*X.nnz/(n_cells*n_genes):.2f}%")

# ── 4. Annotations: day, FLE, cell-set lineage labels ──────────────────────
obs = pd.DataFrame(index=cell_ids_keep)
obs['day'] = days.reindex(obs.index)
obs['day_norm'] = obs['day'] / obs['day'].max()           # time_key in [0,1]

fle = pd.read_csv(f'{DATA}/cluster/FLE.txt', skiprows=[1], index_col=0)
obsm = {'X_fle': fle.reindex(obs.index).values.astype('float32')}

# Parse GMT: each line is "set_name\tdescription\tcell_id1\tcell_id2\t..."
cell_set = pd.Series('Other', index=obs.index, dtype=object)
sizes = {}
with open(f'{DATA}/other/cell_sets.gmt') as f:
    for line in f:
        parts = line.rstrip('\n').split('\t')
        name, members = parts[0], parts[2:]
        sizes[name] = len(members)
        m = obs.index.intersection(members)
        cell_set.loc[m] = name        # last-wins; sets do overlap slightly
obs['cell_set'] = pd.Categorical(cell_set)
print(f"[gmt] cell sets (total members in full data): {sizes}")
print("[gmt] cell_set distribution in sample:")
print(obs['cell_set'].value_counts().to_string())

# ── 5. Save ────────────────────────────────────────────────────────────────
A = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes), obsm=obsm)
out = f'{DATA}/scp295.h5ad'
A.write(out, compression='gzip')
print(f"\nWROTE  {out}   ({A.n_obs} cells × {A.n_vars} genes)")
