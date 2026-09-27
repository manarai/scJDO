"""
r22 — Benchmark suite for scJDO's three differentiator claims that
Rounds 1-7 did NOT test:

  A. Temporal Jacobian tensor as primary object
     — does the temporal decomposition reveal features that per-state
       Jacobian analysis (Dynamo/SpliceJAC-style) structurally cannot?
     Metric: does scJDO's max_real_eig peak at a specific pseudotime
     window (reproducible across seeds)? Does dynamo's per-state aggregate
     produce a comparable temporal signature or a flat one?

  B. Archetype decomposition — real-data reproducibility across seeds
     — Fig 2 reports matched Kendall τ = 0.85-0.97 on synthetic
       sequential handoffs. Does the same hold on real hematopoiesis?
     Metric: pairwise archetype matching (Hungarian on pattern cosine),
     matched activation-profile Kendall τ, top-K gene-loading Jaccard.
     Target: Kendall τ ≥ 0.7 on real data.

  C. Constructed Obstruction — empirical validation
     — Supp Note S1's analytic prediction is that the leading-direction
       summary is preserved on the ψ ∝ ρ subfamily and destroyed under
       sinusoidal ψ. Real-world "priors" that vary are network
       architecture (hidden, depth), velocity prior weight (vel_scale),
       and denoising sigma. Do the empirically-stable-vs-prior outputs
       match the Constructed Obstruction classification?
     Metric: leading eigenvector cosine similarity across prior configs;
     top-K gene Jaccard; max_real_eig curve correlation.

All three rounds use the same substrate: marrow Ery branch cells
(subset of examples/marrow_sample_scseq_counts.h5ad + palantir), so
compute is bounded.

Compute budget: ~60-90 min total on CPU (mostly fit_drift refits).
"""
from __future__ import annotations
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

OUT = REPO / "New_analysis" / "operator_claims_benchmark" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
N_HVG = 1000
N_LATENT = 30


# ---------------------------------------------------------------------------
# Shared setup: load marrow → palantir pseudotime → Ery branch cells
# ---------------------------------------------------------------------------

def prepare_marrow_ery():
    """Load marrow, compute palantir pseudotime + Ery branch cells, return
    an AnnData ready for fit_drift."""
    import scanpy as sc
    import palantir
    import anndata as ad
    from sklearn.decomposition import FactorAnalysis

    print("Loading marrow, computing palantir pseudotime + Ery branch cells...")
    src = REPO / "examples" / "marrow_sample_scseq_counts.h5ad"
    a = ad.read_h5ad(src)
    sc.pp.normalize_per_cell(a)
    palantir.preprocess.log_transform(a)
    sc.pp.highly_variable_genes(a, n_top_genes=N_HVG, flavor="cell_ranger")

    hvg_mask = a.var["highly_variable"].values
    Xhvg = a[:, hvg_mask].X
    if sp.issparse(Xhvg):
        Xhvg = Xhvg.toarray()
    fa = FactorAnalysis(n_components=N_LATENT, random_state=SEED)
    a.obsm["X_fa"] = fa.fit_transform(Xhvg).astype(np.float32)
    a.varm["fa_loadings"] = np.zeros((a.n_vars, N_LATENT), dtype=np.float32)
    a.varm["fa_loadings"][hvg_mask] = fa.components_.T.astype(np.float32)

    palantir.utils.run_diffusion_maps(a, n_components=20, pca_key="X_fa")
    palantir.utils.determine_multiscale_space(a)
    sc.pp.neighbors(a, use_rep="X_fa")

    # Manuscript's terminal cells (fig3 recipe)
    START = "Run5_164698952452459"
    TERMINALS = pd.Series(
        ["DC", "Mono", "Ery"],
        index=["Run5_131097901611291", "Run5_134936662236454",
                "Run4_200562869397916"])
    ts_present = TERMINALS[TERMINALS.index.isin(a.obs_names)]
    palantir.core.run_palantir(a, START, num_waypoints=500,
                                 terminal_states=ts_present)
    palantir.presults.select_branch_cells(a, q=0.01, eps=0.01,
                                            masks_key="branch_masks",
                                            save_as_df=True)
    # Ery branch cells
    ery_mask = a.obsm["branch_masks"]["Ery"].values.astype(bool)
    ery = a[ery_mask].copy()
    ery.obs["pseudotime"] = ery.obs["palantir_pseudotime"].astype(np.float32)
    print(f"  Ery branch: {ery.n_obs} cells × {ery.n_vars} genes")
    return ery


def fit_drift_once(adata, seed, hidden=256, vel_scale=2.0, sigma=0.1,
                    n_epochs=3000, n_archetypes=5, verbose=False):
    """Wrapper around sjd.tl.fit_drift with knobs the Constructed
    Obstruction audit varies."""
    from scjdo.tl import fit_drift
    import torch
    torch.manual_seed(seed); np.random.seed(seed)
    b = adata.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_drift(
            b, time_key="pseudotime", rep="X_fa",
            n_epochs=n_epochs, hidden=hidden, depth=3,
            vel_scale=vel_scale, sigma=sigma,
            n_archetypes=n_archetypes,
            windowing="kernel", bandwidth="auto", grid_size=100,
            seed=seed, verbose=verbose, key_added="scjdo",
        )
    return b


# ---------------------------------------------------------------------------
# Round A — Temporal Jacobian tensor validation
# ---------------------------------------------------------------------------

def round_A_temporal_tensor(ery, n_seeds=3):
    """Fit scJDO on Ery branch × N seeds. Extract per-seed max_real_eig
    curve and archetype activation curves. Score reproducibility of
    temporal features that per-state analysis structurally cannot give."""
    print("\n" + "="*66)
    print("Round A — Temporal Jacobian tensor validation")
    print("="*66)
    curves = []
    peaks = []
    for seed in range(n_seeds):
        t0 = time.time()
        b = fit_drift_once(ery, seed=seed, n_epochs=3000)
        elapsed = time.time() - t0
        res = b.uns["scjdo"]
        max_eig = res["max_real_eig"]         # (T,)
        t_grid = res["t_centers"]              # (T,)
        activations = res["activations"]       # (T, K)
        peak_t = float(t_grid[int(np.argmax(max_eig))])
        curves.append({"seed": seed, "max_eig_curve": max_eig,
                        "t_centers": t_grid, "activations": activations,
                        "elapsed_s": elapsed})
        peaks.append(peak_t)
        print(f"  seed={seed}: fit {elapsed:.1f}s | max_eig peak at τ={peak_t:.3f} "
              f"| max_eig range [{max_eig.min():+.3f}, {max_eig.max():+.3f}]")

    # Reproducibility: peak-τ std across seeds
    peaks_arr = np.array(peaks)
    peak_mean = float(peaks_arr.mean()); peak_std = float(peaks_arr.std())
    print(f"\n  peak τ across {n_seeds} seeds: {peak_mean:.3f} ± {peak_std:.3f}")

    # Curve pairwise Pearson correlation
    curve_arrs = [c["max_eig_curve"] for c in curves]
    ncrs = len(curve_arrs)
    corrs = []
    for i in range(ncrs):
        for j in range(i+1, ncrs):
            c = np.corrcoef(curve_arrs[i], curve_arrs[j])[0, 1]
            corrs.append(float(c))
    mean_corr = float(np.mean(corrs)) if corrs else float("nan")
    print(f"  pairwise Pearson r of max_eig curve: {mean_corr:.3f}  "
          f"(over {len(corrs)} pairs)")

    # Save
    (OUT / "roundA_temporal_summary.json").write_text(json.dumps({
        "peak_tau_mean": peak_mean, "peak_tau_std": peak_std,
        "pairwise_curve_pearson_mean": mean_corr,
        "n_seeds": n_seeds,
        "per_seed_peaks": peaks,
    }, indent=2))
    np.savez_compressed(OUT / "roundA_temporal_curves.npz",
                        curves=[c["max_eig_curve"] for c in curves],
                        t_centers=curves[0]["t_centers"],
                        activations=[c["activations"] for c in curves],
                        peaks=peaks)
    return curves


# ---------------------------------------------------------------------------
# Round B — Archetype decomposition reproducibility on real data
# ---------------------------------------------------------------------------

def round_B_archetype_reproducibility(ery, n_seeds=3):
    """Fit scJDO × N seeds. For each pair of seeds, match archetypes
    via Hungarian on pattern cosine sim, then score matched activation
    Kendall τ and matched pattern cosine."""
    print("\n" + "="*66)
    print("Round B — Archetype reproducibility on real data (Ery branch)")
    print("="*66)
    from scipy.optimize import linear_sum_assignment
    from scipy.stats import kendalltau

    fits = []
    for seed in range(n_seeds):
        t0 = time.time()
        b = fit_drift_once(ery, seed=seed, n_epochs=3000, n_archetypes=5)
        elapsed = time.time() - t0
        res = b.uns["scjdo"]
        patterns = res["patterns"]           # (K, D, D)
        activations = res["activations"]     # (T, K)
        fits.append({"seed": seed, "patterns": patterns,
                      "activations": activations, "elapsed_s": elapsed})
        print(f"  seed={seed}: fit {elapsed:.1f}s | K={patterns.shape[0]}")

    # Pairwise: match archetypes via Hungarian on |pattern cosine|
    def _pattern_cos(A, B):
        a = A.reshape(-1); b = B.reshape(-1)
        n = np.linalg.norm(a) * np.linalg.norm(b) + 1e-9
        return float(np.dot(a, b) / n)

    K = fits[0]["patterns"].shape[0]
    pairs_stats = []
    for i in range(len(fits)):
        for j in range(i+1, len(fits)):
            Pi = fits[i]["patterns"]; Pj = fits[j]["patterns"]
            Ai = fits[i]["activations"]; Aj = fits[j]["activations"]
            # Cost = -|cos| (Hungarian minimizes)
            cost = np.zeros((K, K))
            for a in range(K):
                for c in range(K):
                    cost[a, c] = -abs(_pattern_cos(Pi[a], Pj[c]))
            r, c = linear_sum_assignment(cost)
            matched_cos = float(np.mean(-cost[r, c]))
            # Match activation Kendall τ per matched pair, then average
            taus = []
            for a, cx in zip(r, c):
                tau, _ = kendalltau(Ai[:, a], Aj[:, cx])
                if np.isfinite(tau):
                    taus.append(float(tau))
            mean_tau = float(np.mean(taus)) if taus else float("nan")
            pairs_stats.append({"seed_i": fits[i]["seed"],
                                 "seed_j": fits[j]["seed"],
                                 "matched_pattern_cos_mean": matched_cos,
                                 "matched_activation_kendall_tau_mean": mean_tau})
            print(f"  pair ({fits[i]['seed']},{fits[j]['seed']}): "
                  f"matched-pattern-cos={matched_cos:.3f}  "
                  f"activation Kendall τ={mean_tau:.3f}")

    mean_cos = float(np.mean([s["matched_pattern_cos_mean"] for s in pairs_stats]))
    mean_tau = float(np.mean([s["matched_activation_kendall_tau_mean"] for s in pairs_stats]))
    print(f"\n  mean matched-pattern cos over pairs: {mean_cos:.3f}")
    print(f"  mean matched activation Kendall τ:   {mean_tau:.3f}")
    print(f"  Fig 2 synthetic Kendall τ target: 0.85 (sharp) / 0.97 (gradual)")

    (OUT / "roundB_archetype_reproducibility.json").write_text(json.dumps({
        "n_seeds": n_seeds, "K": K,
        "mean_matched_pattern_cos": mean_cos,
        "mean_matched_activation_kendall_tau": mean_tau,
        "pairs": pairs_stats,
    }, indent=2))
    return {"mean_cos": mean_cos, "mean_tau": mean_tau}


# ---------------------------------------------------------------------------
# Round C — Constructed Obstruction empirical audit
# ---------------------------------------------------------------------------

def round_C_constructed_obstruction_audit(ery):
    """Fit scJDO on Ery under several prior configurations. Measure
    stability of leading eigenvector direction and top-K gene list
    across priors. Constructed Obstruction (Supp Note S1) predicts
    that leading-direction summaries are preserved on the ψ ∝ ρ
    subfamily and destroyed under sinusoidal ψ. Here the empirical
    "priors that vary" are network architecture and vel_scale."""
    print("\n" + "="*66)
    print("Round C — Constructed Obstruction empirical audit")
    print("="*66)
    configs = [
        {"name": "default",       "hidden": 256, "vel_scale": 2.0, "sigma": 0.10},
        {"name": "no_vel",        "hidden": 256, "vel_scale": 0.0, "sigma": 0.10},
        {"name": "smaller_net",   "hidden": 128, "vel_scale": 2.0, "sigma": 0.10},
        {"name": "hi_sigma",      "hidden": 256, "vel_scale": 2.0, "sigma": 0.20},
    ]
    fits = []
    for cfg in configs:
        t0 = time.time()
        b = fit_drift_once(ery, seed=SEED, hidden=cfg["hidden"],
                            vel_scale=cfg["vel_scale"], sigma=cfg["sigma"],
                            n_epochs=3000)
        elapsed = time.time() - t0
        res = b.uns["scjdo"]
        # Top eigenvector at the τ where max_real_eig is maximum
        J = res["J_tensor"]                     # (T, D, D)
        max_eig = res["max_real_eig"]           # (T,)
        t_peak = int(np.argmax(max_eig))
        eigvals, eigvecs = np.linalg.eig(J[t_peak])
        k_top = int(np.argmax(np.real(eigvals)))
        v_peak = np.real(eigvecs[:, k_top])
        # Project to gene space using fa_loadings
        L = ery.varm["fa_loadings"]
        gene_load = np.abs(L @ v_peak)         # (n_genes,)
        fits.append({"cfg": cfg, "v_peak": v_peak, "max_eig_curve": max_eig,
                      "gene_load": gene_load, "elapsed_s": elapsed})
        top_5 = np.argsort(-gene_load)[:5]
        top_5_names = [ery.var_names[i] for i in top_5]
        print(f"  cfg={cfg['name']:<13}: fit {elapsed:.1f}s | top-5 genes "
              f"= {top_5_names}")

    # Stability: cosine between leading eigenvectors of default vs alternatives
    ref = fits[0]["v_peak"]
    ref_curve = fits[0]["max_eig_curve"]
    ref_top15 = set(np.argsort(-fits[0]["gene_load"])[:15])
    stability = []
    for f in fits[1:]:
        v = f["v_peak"]
        # sign-invariant cosine (eigenvectors are only defined up to ±1)
        cos = float(abs(np.dot(ref, v) / (np.linalg.norm(ref) * np.linalg.norm(v) + 1e-9)))
        curve_r = float(np.corrcoef(ref_curve, f["max_eig_curve"])[0, 1])
        top15 = set(np.argsort(-f["gene_load"])[:15])
        jacc = len(ref_top15 & top15) / len(ref_top15 | top15)
        stability.append({
            "cfg": f["cfg"]["name"],
            "leading_eigvec_cos_vs_default": cos,
            "max_eig_curve_pearson_vs_default": curve_r,
            "top15_gene_jaccard_vs_default": float(jacc),
        })
        print(f"  {f['cfg']['name']:<13}: eigvec cos={cos:.3f}  "
              f"curve r={curve_r:.3f}  top-15 Jaccard={jacc:.3f}")

    (OUT / "roundC_constructed_obstruction_audit.json").write_text(json.dumps({
        "configs_tested": [c["cfg"] for c in fits[1:]],
        "stability_vs_default": stability,
        "prediction": ("Constructed Obstruction (Supp Note S1) predicts leading-"
                        "direction eigenvector cos should stay high (~0.9+) under "
                        "conservative priors (vel_scale change, hidden change), "
                        "and degrade under aggressive priors (much higher sigma)."),
    }, indent=2))
    return stability


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ery = prepare_marrow_ery()
    round_A_temporal_tensor(ery, n_seeds=3)
    round_B_archetype_reproducibility(ery, n_seeds=3)
    round_C_constructed_obstruction_audit(ery)
    print(f"\nAll outputs in {OUT}")


if __name__ == "__main__":
    main()
