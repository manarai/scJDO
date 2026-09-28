"""One-seed scJDO fit_drift on the Task A day-2 substrate — SINGLE-BRANCH,
NO TERMINAL GUIDANCE, NO BRANCH MASKING. Uses fit_drift (not
fit_drift_branches) to avoid the "Progenitor" override collision on
day-2-only substrates."""
from __future__ import annotations
import sys, pickle, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import scanpy as sc

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))

from run_gate1_v2_fullcohort import H5AD_PP_TA, OUT
from scjdo.tl import fit_drift

H5AD_FIT = H5AD_PP_TA.parent / "larry_day2_preproc_taskA_fit.h5ad"


def fit_one_seed(a_fit, seed):
    """Fit_drift on the ~14K substrate; return J_tensor + t_centers + eigvec-list
    only (post-processing maps features back to the full 28K cohort)."""
    torch.manual_seed(seed); np.random.seed(seed)
    b = a_fit.copy()
    fit_drift(
        b, rep="X_fa", time_key="pseudotime",
        vel_scale=0.0, hidden=256, depth=4, sigma=0.10,
        windowing="kernel", bandwidth="auto", grid_size=100,
        n_archetypes=5, n_epochs=5000,
        seed=seed, verbose=False, key_added="scjdo_bulk",
    )
    res = b.uns["scjdo_bulk"]
    J_tensor = np.asarray(res["J_tensor"]).astype(np.float32)     # (T, D, D)
    t_centers = np.asarray(res["t_centers"]).astype(np.float32)   # (T,)
    max_re = np.asarray(res["max_real_eig"]).astype(np.float32)   # (T,)
    patterns = np.asarray(res["patterns"]).astype(np.float32)     # (K, D, D)
    activations = np.asarray(res["activations"]).astype(np.float32)
    bw = float(res.get("bandwidth", 0.05))
    # precompute leading eigvec per grid bin (D-vec each)
    lead_vecs = np.zeros((J_tensor.shape[0], J_tensor.shape[1]), dtype=np.float32)
    for i in range(J_tensor.shape[0]):
        ev, evec = np.linalg.eig(J_tensor[i].astype(np.float64))
        k_top = int(np.argmax(np.real(ev)))
        lead_vecs[i] = np.real(evec[:, k_top]).astype(np.float32)
    return {"Ery": {
        "J_tensor": J_tensor,
        "t_centers": t_centers,
        "max_real_eig": max_re,
        "patterns": patterns,
        "activations": activations,
        "lead_vecs": lead_vecs,
        "bandwidth": bw,
    }}


def project_to_full_cohort(feats, a_full):
    """Map J-tensor-derived features to the full 28K day-2 cohort via nearest
    bin in day-2 pseudotime. Requires the full-cohort AnnData to have a
    pseudotime — but the amendment only computed pseudotime on the fit
    substrate. Compute a lightweight pseudotime for the full cohort here
    (Palantir on the full-cohort X_FA) so nearest-bin lookup is defined.
    Return per-cell Re(λ_max), leading-J projection for full-cohort cells."""
    import palantir
    if "pseudotime" not in a_full.obs or np.isnan(a_full.obs["pseudotime"].values).any():
        print("  [project] computing full-cohort palantir pseudotime (for nearest-bin)")
        dm = palantir.utils.run_diffusion_maps(
            pd.DataFrame(a_full.obsm["X_FA"], index=a_full.obs_names), n_components=15)
        ms = palantir.utils.determine_multiscale_space(dm)
        spring_x = a_full.obs["SPRING-x"].astype(float).values \
            if "SPRING-x" in a_full.obs else np.zeros(a_full.n_obs)
        start_cell = a_full.obs_names[int(np.argmin(spring_x))]
        pr = palantir.core.run_palantir(
            ms, start_cell, num_waypoints=1500, seed=0,
            use_early_cell_as_start=False,
        )
        a_full.obs["pseudotime"] = pr.pseudotime.reindex(a_full.obs_names).values.astype(np.float32)
    tau_full = a_full.obs["pseudotime"].values.astype(np.float64)
    tc = feats["Ery"]["t_centers"]
    idx = np.searchsorted(tc, tau_full).clip(0, len(tc) - 1)
    cell_re = feats["Ery"]["max_real_eig"][idx].astype(np.float32)
    X_fa_full = a_full.obsm["X_FA"].astype(np.float64)
    lead_vecs = feats["Ery"]["lead_vecs"].astype(np.float64)
    cell_proj = np.zeros(a_full.n_obs, dtype=np.float32)
    for b in range(len(tc)):
        in_bin = (idx == b)
        if in_bin.any():
            cell_proj[in_bin] = (X_fa_full[in_bin] @ lead_vecs[b]).astype(np.float32)
    feats["Ery"]["re_lambda_max"] = cell_re
    feats["Ery"]["lead_proj"] = cell_proj
    # write the projected pseudotime back to disk so subsequent seeds cache-hit
    a_full.write_h5ad(H5AD_PP_TA)
    return feats


def main():
    seed = int(sys.argv[1])
    t0 = time.time()
    # If fit-only cache exists, skip fit
    fit_only_ck = OUT / f"gate1_v2_taskA_scjdo_seed{seed}.fitonly.pkl"
    if fit_only_ck.exists():
        print(f"[seed {seed} TaskA] fit-only cache hit — loading + projecting", flush=True)
        feats = pickle.loads(fit_only_ck.read_bytes())
    else:
        print(f"[seed {seed} TaskA] loading {H5AD_FIT.name} (~14K)", flush=True)
        a_fit = sc.read_h5ad(H5AD_FIT)
        print(f"[seed {seed} TaskA] fit_drift on {a_fit.shape} (grid=100, epochs=5000)", flush=True)
        feats = fit_one_seed(a_fit, seed)
        # Save fit-only cache BEFORE projection so a projection crash doesn't cost the fit
        fit_only_ck.write_bytes(pickle.dumps(feats))
        print(f"[seed {seed} TaskA] wrote fit-only cache in {time.time()-t0:.1f}s", flush=True)
    # Project J-tensor features to full 28K cohort
    print(f"[seed {seed} TaskA] projecting features to full 28K cohort", flush=True)
    a_full = sc.read_h5ad(H5AD_PP_TA)
    feats = project_to_full_cohort(feats, a_full)
    ck = OUT / f"gate1_v2_taskA_scjdo_seed{seed}.pkl"
    ck.write_bytes(pickle.dumps(feats))
    print(f"[seed {seed} TaskA] wrote {ck.name} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
