"""One-seed scJDO fit on the v2 preproc AnnData (uses X_FA rep)."""
from __future__ import annotations
import sys, pickle, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))

import scanpy as sc
from run_gate1_v2_build import H5AD_PP_V2, OUT
from run_gate1_build import fit_scjdo_features


def main():
    seed = int(sys.argv[1])
    t0 = time.time()
    print(f"[seed {seed} v2] loading {H5AD_PP_V2.name}", flush=True)
    mye = sc.read_h5ad(H5AD_PP_V2)
    print(f"[seed {seed} v2] fit_scjdo_features on {mye.shape}", flush=True)
    feats = fit_scjdo_features(mye, seed)
    ckpt = OUT / f"gate1_v2_scjdo_seed{seed}.pkl"
    ckpt.write_bytes(pickle.dumps(feats))
    print(f"[seed {seed} v2] wrote {ckpt.name} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
