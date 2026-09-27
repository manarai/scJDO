"""Gate 0a — bias-plumbing diagnostic. Per PREREG_Gate0a.md, distinguish
Cases 1/2/3: is bias reaching the model as different v_hat, and does it
produce different trained weights?

Procedure (frozen in the prereg, executed verbatim here):
1. Substrate: marrow Ery branch (r24 pinned pipeline, 1151 cells).
2. Configurations: A1 (vel_scale=2, bias=1.5) and B1 (vel_scale=2, bias=0).
3. Seeds: {0, 1, 2}.
4. For each fit: extract V_ref stored in the VelocityPrior module (this IS
   the v_hat source; v_hat at any cell x is a linear interpolation from
   V_ref via the k-NN weights). Also save the full state_dict().
5. Diff V_ref and state_dict() between A1[s] and B1[s] pairwise.
6. Classify as Case 1/2/3 per the prereg table.

Total compute: 6 fits × ~5 min each = ~30 min.
"""
from __future__ import annotations
import json, sys, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "reproducibility" / "operator_claims_benchmark" / "scripts"))

from run_operator_claims import prepare_marrow_ery
from scjdo.tl import fit_drift_branches

OUT = REPO / "reproducibility" / "gates_r26" / "gate0a"
OUT.mkdir(parents=True, exist_ok=True)


def fit_ery_config(ery, vel_scale, bias_strength, seed):
    """Fit Ery-only fit_drift_branches with the given knobs. Return the
    trained model's V_ref (fixed input velocity), state_dict, and top-15
    leading-direction genes for cross-checking against r24."""
    import torch
    torch.manual_seed(seed); np.random.seed(seed)

    b = ery.copy()
    b.obsm["branch_masks"] = pd.DataFrame({"Ery": np.ones(b.n_obs, dtype=bool)},
                                             index=b.obs_names)
    pt = b.obs["pseudotime"].values
    b.obs["cell_fate"] = "Other"
    b.obs.loc[b.obs["pseudotime"] <= np.quantile(pt, 0.3), "cell_fate"] = "Progenitor"
    b.obs.loc[b.obs["pseudotime"] >= np.quantile(pt, 0.7), "cell_fate"] = "Ery"

    # Hook: monkey-patch fit_drift to intercept the DriftField model after
    # training. Simpler than instrumenting forward — we just need V_ref +
    # state_dict, both available on the trained model post-fit.
    #
    # But fit_drift_branches doesn't return the model — it stores results
    # in adata.uns and drops the model. So we intercept via a wrapper of
    # DriftField.__init__ that saves 'self' to a module-level list.
    from scjdo.models import drift as _drift
    captured = {"model": None, "V_ref": None}
    orig_init = _drift.DriftField.__init__

    def _spy_init(self, cfg, X_ref=None, V_ref=None):
        orig_init(self, cfg, X_ref=X_ref, V_ref=V_ref)
        captured["model"] = self
        # V_ref is stored as a buffer on self.vel (a VelocityPrior module)
        if getattr(self, "vel", None) is not None and hasattr(self.vel, "V_ref"):
            captured["V_ref"] = self.vel.V_ref.detach().cpu().numpy().copy()

    _drift.DriftField.__init__ = _spy_init
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_drift_branches(
                b, rep="X_fa", branch_key="branch_masks", branch_names=["Ery"],
                time_key="pseudotime", groupby="cell_fate",
                progenitor_cluster="Progenitor",
                terminal_clusters={"Ery": "Ery"},
                bias_strength=bias_strength,
                n_archetypes=5, n_epochs=5000,
                vel_scale=vel_scale, hidden=256, depth=4, sigma=0.10,
                windowing="kernel", bandwidth="auto", grid_size=200,
                seed=seed, verbose=False,
            )
    finally:
        _drift.DriftField.__init__ = orig_init

    if captured["model"] is None:
        raise RuntimeError("failed to capture DriftField model")

    state = {k: v.detach().cpu().numpy().copy()
              for k, v in captured["model"].state_dict().items()}

    # Extract top-15 for r24 cross-check
    res = b.uns["scjdo_Ery"]
    J = np.asarray(res["J_tensor"])
    max_eig = np.asarray(res["max_real_eig"])
    t_peak = int(np.argmax(max_eig))
    eigvals, eigvecs = np.linalg.eig(J[t_peak])
    k_top = int(np.argmax(np.real(eigvals)))
    v_peak = np.real(eigvecs[:, k_top])
    L = ery.varm["fa_loadings"]
    gene_load = np.abs(L @ v_peak)
    top15 = [ery.var_names[i] for i in np.argsort(-gene_load)[:15]]

    return {"V_ref": captured["V_ref"], "state_dict": state,
             "peak_max_eig": float(max_eig[t_peak]),
             "peak_tau": float(res["t_centers"][t_peak]),
             "top15": top15}


def diff_arrays(A, B):
    A = np.asarray(A); B = np.asarray(B)
    if A.shape != B.shape:
        return {"same_shape": False, "shape_A": A.shape, "shape_B": B.shape}
    diff = A - B
    return {
        "same_shape": True,
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "identical_to_float32": bool(np.allclose(A, B, atol=1e-6, rtol=1e-6)),
        "bitwise_identical": bool(np.array_equal(A, B)),
    }


def main():
    ery = prepare_marrow_ery()
    print(f"Ery substrate: {ery.n_obs} × {ery.n_vars}")

    fits = {}
    for cfg_name, vel, bias in (("A1", 2.0, 1.5), ("B1", 2.0, 0.0)):
        for seed in (0, 1, 2):
            print(f"\n[fit] {cfg_name} seed={seed} (vel={vel}, bias={bias}) ...")
            t0 = time.time()
            r = fit_ery_config(ery, vel_scale=vel, bias_strength=bias, seed=seed)
            print(f"  fit {time.time()-t0:.1f}s | peak τ={r['peak_tau']:.3f}  "
                  f"max_eig={r['peak_max_eig']:+.3f}  top-5={r['top15'][:5]}")
            fits[(cfg_name, seed)] = r

    # Diff per seed
    print("\n" + "="*80)
    print("Per-seed diff: A1 vs B1 (same seed)")
    print("="*80)
    diffs = {}
    for seed in (0, 1, 2):
        A = fits[("A1", seed)]
        B = fits[("B1", seed)]
        v_diff = diff_arrays(A["V_ref"], B["V_ref"])
        # State dict diff
        param_diffs = {}
        for k in A["state_dict"]:
            if k in B["state_dict"]:
                param_diffs[k] = diff_arrays(A["state_dict"][k], B["state_dict"][k])
        max_abs_param = max(d.get("max_abs", 0.0) for d in param_diffs.values())
        n_identical = sum(1 for d in param_diffs.values() if d.get("bitwise_identical"))
        print(f"\nseed {seed}:")
        print(f"  V_ref     max |A-B|: {v_diff.get('max_abs', 'shape mismatch'):.6f}"
              f"   bitwise_identical: {v_diff.get('bitwise_identical')}")
        print(f"  state_dict: {n_identical}/{len(param_diffs)} params bitwise-identical, "
              f"max |Δparam| across all params: {max_abs_param:.6f}")
        # Top-15 comparison
        overlap = set(A["top15"]) & set(B["top15"])
        print(f"  top-15 overlap: {len(overlap)}/15   A: {A['top15'][:5]}   "
              f"B: {B['top15'][:5]}")
        diffs[seed] = {
            "v_ref_max_abs": v_diff.get("max_abs"),
            "v_ref_bitwise_identical": v_diff.get("bitwise_identical"),
            "state_dict_max_abs_param": max_abs_param,
            "state_dict_n_identical": n_identical,
            "state_dict_n_total": len(param_diffs),
            "top15_overlap": len(overlap),
            "A_top5": A["top15"][:5],
            "B_top5": B["top15"][:5],
            "A_peak_max_eig": A["peak_max_eig"],
            "B_peak_max_eig": B["peak_max_eig"],
        }

    # Case classification (per PREREG_Gate0a.md)
    v_diff_all = [d["v_ref_max_abs"] for d in diffs.values() if d["v_ref_max_abs"] is not None]
    weights_diff_all = [d["state_dict_max_abs_param"] for d in diffs.values()]
    v_identical = all(x is not None and x < 1e-6 for x in v_diff_all)
    weights_identical = all(x < 1e-6 for x in weights_diff_all)

    if v_identical and weights_identical:
        verdict = "Case 1 (BUG): bias never reaches the model; v_hat identical AND weights identical"
    elif not v_identical and weights_identical:
        verdict = ("Case 2 (unusual): v_hat differs across configs but training "
                    "converges to bit-identical weights")
    elif not v_identical and not weights_identical:
        verdict = ("Case 3: bias reaches model and produces different weights; "
                    "any top-K basin equivalence is dominated by seed, not bias")
    else:  # v_identical but weights differ — shouldn't happen but log it
        verdict = ("Anomaly: v_hat identical but weights differ across configs. "
                    "Investigate: implies training injects randomness beyond seed.")

    print("\n" + "="*80)
    print(f"VERDICT: {verdict}")
    print("="*80)

    (OUT / "gate0a_summary.json").write_text(json.dumps({
        "per_seed_diffs": diffs,
        "v_identical_across_all_seeds": v_identical,
        "weights_identical_across_all_seeds": weights_identical,
        "verdict": verdict,
    }, indent=2))
    print(f"\nWritten: {OUT / 'gate0a_summary.json'}")


if __name__ == "__main__":
    main()
