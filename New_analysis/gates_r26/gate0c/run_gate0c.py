"""Gate 0c — consensus archetypes (cNMF-style) per PREREG_Gate0c.md.
Runs R=10 seeds × 2 arms (vel_scale ∈ {0.0, 2.0}) of fit_drift_branches
on marrow Ery, extracts 5 patterns per replicate, and consensus-clusters
the 50 patterns per arm into K=5 groups. Cluster passes if it draws from
≥ 80% of replicates. Default arm = higher n_consensus_clusters (tiebreak:
higher mean intra-cluster |cos|)."""
from __future__ import annotations
import json, sys, warnings, time
from pathlib import Path
import numpy as np, pandas as pd
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "New_analysis" / "operator_claims_benchmark" / "scripts"))

from run_operator_claims import prepare_marrow_ery
from scjdo.tl import fit_drift_branches

OUT = REPO / "New_analysis" / "gates_r26" / "gate0c"
OUT.mkdir(parents=True, exist_ok=True)

R = 10
K = 5
CONSENSUS_THRESHOLD = 8  # >= 80% of R = 10
ARMS = [("V0", 0.0), ("V2", 2.0)]


def fit_one(ery, vel_scale, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    b = ery.copy()
    b.obsm["branch_masks"] = pd.DataFrame(
        {"Ery": np.ones(b.n_obs, dtype=bool)}, index=b.obs_names,
    )
    pt = b.obs["pseudotime"].values
    b.obs["cell_fate"] = "Other"
    b.obs.loc[b.obs["pseudotime"] <= np.quantile(pt, 0.3), "cell_fate"] = "Progenitor"
    b.obs.loc[b.obs["pseudotime"] >= np.quantile(pt, 0.7), "cell_fate"] = "Ery"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_drift_branches(
            b, rep="X_fa", branch_key="branch_masks", branch_names=["Ery"],
            time_key="pseudotime", groupby="cell_fate",
            progenitor_cluster="Progenitor",
            terminal_clusters={"Ery": "Ery"},
            bias_strength=1.5, n_archetypes=K, n_epochs=5000,
            vel_scale=vel_scale, hidden=256, depth=4, sigma=0.10,
            windowing="kernel", bandwidth="auto", grid_size=200,
            seed=seed, verbose=False,
        )
    P = np.asarray(b.uns["scjdo_Ery"]["patterns"])  # (K, D, D)
    return P


def sign_invariant_flatten(P):
    """(K, D, D) → (K, D*D) unit-L2 with leading-element sign convention."""
    K_, D, _ = P.shape
    F = P.reshape(K_, -1).astype(np.float64)
    norms = np.linalg.norm(F, axis=1, keepdims=True) + 1e-12
    F = F / norms
    # sign-invariance: flip so first non-negligible component is positive
    lead = F[:, 0]
    flip = np.where(lead < 0, -1.0, 1.0).reshape(-1, 1)
    return F * flip


def consensus_cluster(F, rep_ids, K_cluster=K):
    """F: (N, D²) sign-normalised. rep_ids: (N,) replicate index per pattern.
    Returns list of clusters, each a dict with:
      - members (idx into F)
      - reps (set of replicate ids)
      - R_cluster (len of reps)
      - intra_cos (mean pairwise |cos| within cluster)
      - centroid (D²,) normalised mean
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    # |cos| distance
    S = np.abs(F @ F.T)
    np.clip(S, 0.0, 1.0, out=S)
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    # symmetrise for numeric safety
    D = 0.5 * (D + D.T)
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method="single")
    labels = fcluster(Z, t=K_cluster, criterion="maxclust")

    clusters = []
    for lbl in np.unique(labels):
        members = np.where(labels == lbl)[0]
        reps = set(int(rep_ids[m]) for m in members)
        # intra-cluster mean |cos| (upper triangle only)
        if len(members) > 1:
            sub = S[np.ix_(members, members)]
            iu = np.triu_indices(len(members), k=1)
            intra = float(sub[iu].mean())
        else:
            intra = 1.0
        centroid = F[members].mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        clusters.append({
            "label": int(lbl),
            "members": members.tolist(),
            "reps": sorted(reps),
            "R_cluster": len(reps),
            "intra_cos": intra,
            "centroid": centroid,
        })
    return clusters, labels, S


def main():
    ery = prepare_marrow_ery()
    print(f"Substrate: Ery {ery.n_obs} × {ery.n_vars}")
    D = ery.obsm["X_fa"].shape[1]

    arm_results = {}
    for arm_name, vel in ARMS:
        print(f"\n[Arm {arm_name}: vel_scale={vel}, R={R} seeds]")
        patterns_stack = []
        rep_ids = []
        t_start = time.time()
        for seed in range(R):
            t0 = time.time()
            P = fit_one(ery, vel, seed)
            patterns_stack.append(P)
            rep_ids.extend([seed] * K)
            print(f"  seed {seed}: {time.time()-t0:.1f}s  patterns {P.shape}")
        patterns_stack = np.concatenate(patterns_stack, axis=0)  # (R*K, D, D)
        rep_ids = np.array(rep_ids)
        print(f"  total {time.time()-t_start:.1f}s")

        F = sign_invariant_flatten(patterns_stack)
        clusters, labels, S = consensus_cluster(F, rep_ids, K_cluster=K)

        passing = [c for c in clusters if c["R_cluster"] >= CONSENSUS_THRESHOLD]
        n_pass = len(passing)
        mean_intra = float(np.mean([c["intra_cos"] for c in passing])) if passing else 0.0

        _cluster_report = sorted(
            [(c["label"], c["R_cluster"], round(c["intra_cos"], 3)) for c in clusters],
            key=lambda x: -x[1],
        )
        print(f"  cluster label recurrences: {_cluster_report}")
        print(f"  passing (R>=8): {n_pass}, mean intra |cos| = {mean_intra:.3f}")

        # sanity: duplicate-within-replicate — do multiple archetypes of the
        # same replicate end up in the SAME passing cluster?
        dup_counts = []
        for c in passing:
            reps_list = [int(rep_ids[m]) for m in c["members"]]
            # count replicates that contributed > 1 pattern
            from collections import Counter
            ctr = Counter(reps_list)
            dup_counts.append(sum(1 for v in ctr.values() if v > 1))
        print(f"  duplicate-within-replicate per passing cluster: {dup_counts}")

        arm_results[arm_name] = {
            "vel_scale": vel,
            "clusters": [
                {
                    "label": c["label"],
                    "R_cluster": c["R_cluster"],
                    "intra_cos": c["intra_cos"],
                    "n_members": len(c["members"]),
                    "reps": c["reps"],
                }
                for c in clusters
            ],
            "n_consensus_clusters": n_pass,
            "mean_intra_cos_passing": mean_intra,
            "dup_within_rep_per_passing": dup_counts,
        }
        # save centroids for the chosen-arm downstream use
        np.savez(
            OUT / f"consensus_centroids_{arm_name}.npz",
            centroids=np.stack([c["centroid"] for c in clusters]),
            labels=np.array([c["label"] for c in clusters]),
            R_clusters=np.array([c["R_cluster"] for c in clusters]),
            intra_cos=np.array([c["intra_cos"] for c in clusters]),
            passing_mask=np.array([c["R_cluster"] >= CONSENSUS_THRESHOLD for c in clusters]),
            D=D,
        )

    # deterministic default choice
    n_v0 = arm_results["V0"]["n_consensus_clusters"]
    n_v2 = arm_results["V2"]["n_consensus_clusters"]
    if n_v0 > n_v2:
        chosen = "V0"
    elif n_v2 > n_v0:
        chosen = "V2"
    else:
        # tiebreak: higher mean intra_cos
        if arm_results["V0"]["mean_intra_cos_passing"] >= arm_results["V2"]["mean_intra_cos_passing"]:
            chosen = "V0"
        else:
            chosen = "V2"

    print("\n" + "="*80)
    print(f"n_consensus_clusters: V0={n_v0}, V2={n_v2}")
    print(f"mean intra |cos|:     V0={arm_results['V0']['mean_intra_cos_passing']:.3f}, "
          f"V2={arm_results['V2']['mean_intra_cos_passing']:.3f}")
    print(f"CHOSEN DEFAULT ARM: {chosen}  (vel_scale={arm_results[chosen]['vel_scale']})")
    if n_v0 == 0 and n_v2 == 0:
        print("STOP-RULE FLAG: both arms have zero passing clusters (per PREREG).")
    print("="*80)

    (OUT / "gate0c_summary.json").write_text(json.dumps({
        "R": R, "K": K,
        "consensus_threshold_R": CONSENSUS_THRESHOLD,
        "arm_V0": arm_results["V0"],
        "arm_V2": arm_results["V2"],
        "chosen_default_arm": chosen,
        "chosen_vel_scale": arm_results[chosen]["vel_scale"],
        "n_consensus_clusters_V0": n_v0,
        "n_consensus_clusters_V2": n_v2,
        "mean_intra_cos_V0": arm_results["V0"]["mean_intra_cos_passing"],
        "mean_intra_cos_V2": arm_results["V2"]["mean_intra_cos_passing"],
    }, indent=2))
    print(f"\nWritten: {OUT / 'gate0c_summary.json'}")
    print(f"Consensus centroids saved as: consensus_centroids_V0.npz, consensus_centroids_V2.npz")


if __name__ == "__main__":
    main()
