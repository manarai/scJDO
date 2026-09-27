"""Gate 1 v2 eval — 5 rows: E_PCA, E_FA, E_scVI, S_FA, E_FA + S_FA."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).parent
FEAT = OUT / "gate1_v2_features.npz"

CV_SEEDS = [42, 43, 44]
N_SPLITS = 5
N_BOOT = 2000


def group_kfold_auroc(X, y, groups, cv_seed, n_splits=N_SPLITS):
    uniq = np.unique(groups)
    perm = shuffle(np.arange(len(uniq)), random_state=cv_seed)
    reordered = uniq[perm]
    mp = {g: i for i, g in enumerate(reordered)}
    g2 = np.array([mp[g] for g in groups])
    gkf = GroupKFold(n_splits=n_splits)
    aucs = []
    for tr, te in gkf.split(X, y, g2):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(penalty="l2", C=1.0, class_weight="balanced",
                                  solver="liblinear", max_iter=1000)
        clf.fit(sc.transform(X[tr]), y[tr])
        if len(np.unique(y[te])) < 2:
            aucs.append(np.nan); continue
        proba = clf.predict_proba(sc.transform(X[te]))[:, 1]
        aucs.append(float(roc_auc_score(y[te], proba)))
    return aucs


def eval_set(X, y, groups):
    aucs = []
    for cv_seed in CV_SEEDS:
        aucs.extend(group_kfold_auroc(X, y, groups, cv_seed))
    return np.array(aucs)


def paired_bootstrap_ci(a, b, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    diffs = a - b
    bmeans = []
    for _ in range(n_boot):
        ii = rng.integers(0, len(diffs), size=len(diffs))
        bmeans.append(diffs[ii].mean())
    bmeans = np.array(bmeans)
    return float(diffs.mean()), (float(np.percentile(bmeans, 2.5)),
                                  float(np.percentile(bmeans, 97.5)))


def main():
    print("=" * 80); print("Gate 1 v2 evaluation"); print("=" * 80)
    z = np.load(FEAT, allow_pickle=True)
    mask = z["cohort_mask"].astype(bool)
    y = z["cohort_label"][mask].astype(int)
    groups = z["cohort_group"][mask].astype(np.int64)
    print(f"cohort: n={mask.sum()}, neut={(y==1).sum()}, mono={(y==0).sum()}, "
          f"clones={len(np.unique(groups))}")

    feats = {
        "E_PCA":   z["X_PCA"][mask].astype(np.float32),
        "E_FA":    z["X_FA"][mask].astype(np.float32),
        "E_scVI":  z["X_scVI"][mask].astype(np.float32),
    }
    available_seeds = z["available_seeds"].tolist()
    S_seeds = {s: z[f"S_FA_seed{s}"][mask].astype(np.float32) for s in available_seeds}

    results = {}
    for name, X in feats.items():
        aucs = eval_set(X, y, groups)
        results[name] = aucs
        print(f"[{name}]  X={X.shape}  mean AUROC = {np.nanmean(aucs):.4f}  "
              f"across-fold sd={np.nanstd(aucs):.4f}")

    # S_FA per seed
    S_seed_aucs = {}
    for s in sorted(S_seeds):
        aucs = eval_set(S_seeds[s], y, groups)
        S_seed_aucs[s] = aucs
        print(f"[S_FA_seed{s}]  X={S_seeds[s].shape}  mean AUROC = {np.nanmean(aucs):.4f}")
    # per-fold mean of S_FA across scJDO seeds (15 values)
    S_FA_per_fold = np.array([
        np.mean([S_seed_aucs[s][f] for s in sorted(S_seeds)])
        for f in range(len(results["E_PCA"]))
    ])
    S_FA_all = np.concatenate([S_seed_aucs[s] for s in sorted(S_seeds)])
    results["S_FA"] = S_FA_all
    print(f"[S_FA]  mean AUROC (per-fold avg over {len(S_seeds)} scJDO seeds) = "
          f"{float(np.nanmean(S_FA_per_fold)):.4f}")

    # E_FA + S_FA
    combo_arr_all = []
    for s in sorted(S_seeds):
        X_combo = np.concatenate([feats["E_FA"], S_seeds[s]], axis=1)
        aucs = eval_set(X_combo, y, groups)
        combo_arr_all.append(aucs)
    combo_arr = np.stack(combo_arr_all, axis=0)  # (3, 15)
    combo_per_fold = combo_arr.mean(axis=0)
    results["E_FA+S_FA"] = combo_arr.flatten()
    print(f"[E_FA + S_FA]  mean AUROC = {float(np.nanmean(combo_arr)):.4f}")

    # Best baseline of {E_PCA, E_FA, E_scVI}
    baseline_means = {k: float(np.nanmean(results[k])) for k in ["E_PCA", "E_FA", "E_scVI"]}
    best = max(baseline_means, key=baseline_means.get)
    S_mean = float(np.nanmean(S_FA_per_fold))
    delta = S_mean - baseline_means[best]
    print("\n=== Best baseline: ", best, f"= {baseline_means[best]:.4f}")
    print(f"S_FA mean AUROC (per-fold-averaged): {S_mean:.4f}")
    print(f"S_FA - {best}: {delta:+.4f}")

    # Paired bootstrap CIs
    print("\nPaired bootstrap 95% CIs")
    pair = {}
    for baseline in ["E_PCA", "E_FA", "E_scVI"]:
        diff, ci = paired_bootstrap_ci(S_FA_per_fold, results[baseline])
        pair[f"S_FA_vs_{baseline}"] = {"diff": diff, "ci": ci,
                                        "excl_zero": bool(ci[0] > 0 or ci[1] < 0)}
        print(f"  S_FA - {baseline}: {diff:+.4f}  95%CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  "
              f"{'*' if pair[f'S_FA_vs_{baseline}']['excl_zero'] else ''}")
    diff, ci = paired_bootstrap_ci(combo_per_fold, results["E_FA"])
    pair["E_FA+S_FA_vs_E_FA"] = {"diff": diff, "ci": ci,
                                   "excl_zero": bool(ci[0] > 0 or ci[1] < 0)}
    print(f"  E_FA+S_FA - E_FA: {diff:+.4f}  95%CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  "
          f"{'*' if pair['E_FA+S_FA_vs_E_FA']['excl_zero'] else ''}")

    # Primary criterion
    primary_pass = (delta >= 0.03) and pair[f"S_FA_vs_{best}"]["excl_zero"] \
                    and pair[f"S_FA_vs_{best}"]["ci"][0] > 0
    verdict = "PASS Gate 1 v2" if primary_pass else "FAIL Gate 1 v2"
    print(f"\nVerdict: {verdict}")

    # Save summary
    def _tn(v):
        if isinstance(v, (np.floating,)): return float(v)
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, (list, tuple)): return [_tn(x) for x in v]
        if isinstance(v, dict): return {k: _tn(x) for k, x in v.items()}
        return v
    (OUT / "gate1_v2_eval_summary.json").write_text(json.dumps(_tn({
        "cohort_n": int(mask.sum()),
        "n_clones": int(len(np.unique(groups))),
        "row_means": {
            "E_PCA": float(np.nanmean(results["E_PCA"])),
            "E_FA": float(np.nanmean(results["E_FA"])),
            "E_scVI": float(np.nanmean(results["E_scVI"])),
            "S_FA": S_mean,
            "E_FA+S_FA": float(np.nanmean(results["E_FA+S_FA"])),
        },
        "row_sds": {
            "E_PCA": float(np.nanstd(results["E_PCA"])),
            "E_FA": float(np.nanstd(results["E_FA"])),
            "E_scVI": float(np.nanstd(results["E_scVI"])),
            "S_FA_per_scjdo_seed": [float(np.nanmean(S_seed_aucs[s])) for s in sorted(S_seeds)],
            "E_FA+S_FA_per_scjdo_seed": [float(np.nanmean(combo_arr[i])) for i in range(len(sorted(S_seeds)))],
        },
        "pair_ci": pair,
        "best_baseline": best,
        "delta_vs_best_baseline": delta,
        "verdict": verdict,
    }), indent=2))
    np.savez(OUT / "gate1_v2_eval_aucs.npz", **{k: v for k, v in results.items()})
    print(f"\n[written] gate1_v2_eval_summary.json + gate1_v2_eval_aucs.npz")


if __name__ == "__main__":
    main()
