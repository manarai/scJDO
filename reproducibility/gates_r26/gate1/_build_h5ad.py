"""Minimal helper: mtx → myeloid-subsetted h5ad. Run as subprocess so
memory is released before the heavy scJDO/palantir/cellrank stages."""
from __future__ import annotations
import gzip, sys, gc, time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmread
import anndata as ad

LARRY_DIR = Path("/tmp/larry_data")
OUT_H5AD = Path("/Users/terooatt/Downloads/scJDO/reproducibility/gates_r26/gate1/data/larry_myeloid.h5ad")
OUT_H5AD.parent.mkdir(parents=True, exist_ok=True)
MYELOID = {"Undifferentiated", "Neutrophil", "Monocyte"}


def read_mtx(path):
    t0 = time.time()
    # prefer plain mtx over gz to save gzip decompression buffer
    plain = Path(str(path).rstrip(".gz")) if path.suffix == ".gz" else path
    if plain.exists():
        print(f"[helper] reading plain mtx {plain}...")
        m = mmread(plain)
    else:
        print(f"[helper] reading gz mtx {path}...")
        with gzip.open(path, "rt") as f:
            m = mmread(f)
    print(f"[helper] read {m.shape} nnz={m.nnz} in {time.time()-t0:.1f}s")
    return m


def main():
    t0 = time.time()
    print(f"[helper] loading counts...")
    X = read_mtx(LARRY_DIR / "stateFate_inVitro_normed_counts.mtx.gz").tocsr().astype(np.float32)
    gc.collect()

    genes = pd.read_csv(LARRY_DIR / "stateFate_inVitro_gene_names.txt.gz",
                         sep="\t", header=None)[0].values
    meta = pd.read_csv(LARRY_DIR / "stateFate_inVitro_metadata.txt.gz", sep="\t")
    C = read_mtx(LARRY_DIR / "stateFate_inVitro_clone_matrix.mtx.gz").tocsr()
    print(f"[helper] meta {meta.shape} genes {len(genes)} clone {C.shape}")

    ct = meta["Cell type annotation"].astype(str).values
    day = meta["Time point"].astype(int).values
    mye_mask = np.isin(ct, list(MYELOID))
    print(f"[helper] myeloid mask: {int(mye_mask.sum())}")

    X_mye = X[mye_mask]
    C_mye = C[mye_mask]
    del X, C; gc.collect()
    print(f"[helper] subset X={X_mye.shape} C={C_mye.shape}")

    obs = meta[mye_mask].reset_index(drop=True).copy()
    obs.index = pd.Index([f"cell_{i}" for i in range(len(obs))])
    obs["cell_type"] = obs["Cell type annotation"].astype(str)
    obs["day"] = obs["Time point"].astype(int)

    var = pd.DataFrame(index=pd.Index(genes, name="gene"))
    a = ad.AnnData(X=X_mye, obs=obs, var=var)
    a.obsm["X_clone"] = C_mye
    print(f"[helper] writing h5ad ...")
    a.write_h5ad(OUT_H5AD, compression="gzip")
    print(f"[helper] wrote {OUT_H5AD}  {a.shape}  total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
