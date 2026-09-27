"""Run fit_drift_branches for ONE seed on the preprocessed LARRY h5ad
and save features to gate1_scjdo_seed<seed>.pkl. Invoked as subprocess
by run_gate1_build.py so a stuck seed can be timed out cleanly."""
from __future__ import annotations
import sys, pickle, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))

import scanpy as sc, numpy as np, pandas as pd
from run_gate1_build import fit_scjdo_features, H5AD_PATH_PP, OUT

def main():
    seed = int(sys.argv[1])
    t0 = time.time()
    print(f"[seed {seed}] loading {H5AD_PATH_PP.name}", flush=True)
    mye = sc.read_h5ad(H5AD_PATH_PP)
    print(f"[seed {seed}] fit_scjdo_features on {mye.shape}", flush=True)
    feats = fit_scjdo_features(mye, seed)
    ckpt = OUT / f"gate1_scjdo_seed{seed}.pkl"
    ckpt.write_bytes(pickle.dumps(feats))
    print(f"[seed {seed}] wrote {ckpt} in {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
