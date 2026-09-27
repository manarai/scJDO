"""Gate 1 — classifier evaluation (fast, iterable).
Loads gate1_features.npz, runs L2 logistic regression with GroupKFold(5)
by clone across CV seeds {42, 43, 44} × 8 feature sets, reports AUROC
per fold + paired bootstrap CIs vs scJDO S bundle."""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).parent
FEAT = OUT / "gate1_features.npz"

CV_SEEDS = [42, 43, 44]
N_SPLITS = 5
N_BOOT = 2000


def load_features():
    z = np.load(FEAT, allow_pickle=True)
    mask = z["cohort_mask"].astype(bool)
    y = z["cohort_label"][mask].astype(int)
    groups = z["cohort_group"][mask].astype(np.int64)
    feats = {
        "E": z["X_pca"][mask].astype(np.float32),
        "F": z["X_fa"][mask].astype(np.float32),
        "C": z["cov_features"][mask].astype(np.float32),
        "K": z["cellrank_features"][mask].astype(np.float32),
    }
    S_seeds = {}
    for s in (0, 1, 2):
        key = f"S_seed{s}"
        if key in z.files:
            S_seeds[s] = z[key][mask].astype(np.float32)
    print(f"cohort: n={mask.sum()}, neut={(y==1).sum()}, mono={(y==0).sum()}")
    print(f"n unique clones: {len(np.unique(groups))}")
    print(f"E={feats['E'].shape}, F={feats['F'].shape}, "
          f"C={feats['C'].shape}, K={feats['K'].shape}")
    print(f"S seeds: {[(s, S_seeds[s].shape) for s in S_seeds]}")
    return feats, S_seeds, y, groups


def group_kfold_auroc(X, y, groups, cv_seed, n_splits=N_SPLITS):
    """Return list of n_splits AUROC values under one CV seed."""
    uniq = np.unique(groups)
    perm = shuffle(np.arange(len(uniq)), random_state=cv_seed)
    reordered = uniq[perm]
    # remap groups so GroupKFold splits are shuffled
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


def eval_set(name, X, y, groups):
    """15 folds (5 × 3 CV seeds)."""
    aucs = []
    for cv_seed in CV_SEEDS:
        aucs.extend(group_kfold_auroc(X, y, groups, cv_seed))
    return np.array(aucs)


def paired_bootstrap_ci(a, b, n_boot=N_BOOT, seed=0):
    """Bootstrap 95% CI of mean(a) - mean(b), paired over folds."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(a))
    diffs = a - b
    bmeans = []
    for _ in range(n_boot):
        ii = rng.integers(0, len(idx), size=len(idx))
        bmeans.append(diffs[ii].mean())
    bmeans = np.array(bmeans)
    return float(diffs.mean()), (float(np.percentile(bmeans, 2.5)),
                                  float(np.percentile(bmeans, 97.5)))


def main():
    print("=" * 80); print("Gate 1: evaluation"); print("=" * 80)
    feats, S_seeds, y, groups = load_features()

    results = {}
    for name, X in feats.items():
        print(f"\n[{name}] X={X.shape}")
        aucs = eval_set(name, X, y, groups)
        results[name] = aucs
        print(f"  mean AUROC = {np.nanmean(aucs):.4f} (n={len(aucs)}, "
              f"across-fold sd={np.nanstd(aucs):.4f})")

    # S per seed
    S_all = []
    for s in sorted(S_seeds):
        print(f"\n[S_seed{s}] X={S_seeds[s].shape}")
        aucs = eval_set(f"S_seed{s}", S_seeds[s], y, groups)
        results[f"S_seed{s}"] = aucs
        S_all.extend(aucs)
        print(f"  mean AUROC = {np.nanmean(aucs):.4f} (n={len(aucs)}, "
              f"across-fold sd={np.nanstd(aucs):.4f})")
    # S aggregate: 45 fold-values
    results["S"] = np.array(S_all)
    # S per-fold mean across seeds (15 values), used for paired-bootstrap vs baselines
    S_per_fold_mean = np.array(
        [np.mean([results[f"S_seed{s}"][f] for s in sorted(S_seeds)])
         for f in range(len(results["E"]))]
    )
    results["S_per_fold_mean"] = S_per_fold_mean

    # E+S, F+S, C+S — concat features (use average across S seeds for combined baseline test)
    # Since combined set must have same seed as X, we take seed-0 S for the combined arms.
    for combo_name, base_name, X_base in [("E+S", "E", feats["E"]),
                                            ("F+S", "F", feats["F"]),
                                            ("C+S", "C", feats["C"])]:
        combo_aucs_per_seed = []
        for s in sorted(S_seeds):
            X_combo = np.concatenate([X_base, S_seeds[s]], axis=1)
            aucs = eval_set(combo_name, X_combo, y, groups)
            combo_aucs_per_seed.append(aucs)
        combo_arr = np.array(combo_aucs_per_seed)  # (3, 15)
        combo_mean_per_fold = combo_arr.mean(axis=0)  # (15,)
        results[combo_name] = combo_arr.flatten()
        results[f"{combo_name}_per_fold_mean"] = combo_mean_per_fold
        print(f"\n[{combo_name}] mean AUROC = {np.nanmean(combo_arr):.4f} "
              f"(n={combo_arr.size}, across-fold sd={np.nanstd(combo_arr):.4f})")

    # Paired bootstrap CIs
    print("\n" + "=" * 80); print("Paired bootstrap 95% CIs"); print("=" * 80)
    pair_results = {}
    for baseline in ["E", "F", "C", "K"]:
        diff, ci = paired_bootstrap_ci(S_per_fold_mean, results[baseline])
        pair_results[f"S_vs_{baseline}"] = {"diff": diff, "ci_lo": ci[0], "ci_hi": ci[1],
                                              "excl_zero": bool(ci[0] > 0 or ci[1] < 0)}
        print(f"  S - {baseline}: {diff:+.4f}  95%CI [{ci[0]:+.4f}, {ci[1]:+.4f}] "
              f"{'*' if ci[0] > 0 or ci[1] < 0 else ''}")
    for combo, base in [("E+S", "E"), ("F+S", "F"), ("C+S", "C")]:
        diff, ci = paired_bootstrap_ci(results[f"{combo}_per_fold_mean"], results[base])
        pair_results[f"{combo}_vs_{base}"] = {"diff": diff, "ci_lo": ci[0], "ci_hi": ci[1],
                                                "excl_zero": bool(ci[0] > 0 or ci[1] < 0)}
        print(f"  {combo} - {base}: {diff:+.4f}  95%CI [{ci[0]:+.4f}, {ci[1]:+.4f}] "
              f"{'*' if ci[0] > 0 or ci[1] < 0 else ''}")

    # Best baseline
    baseline_means = {k: float(np.nanmean(results[k])) for k in ["E", "F", "C", "K"]}
    best_baseline = max(baseline_means, key=baseline_means.get)
    print(f"\nBest baseline: {best_baseline} (mean AUROC {baseline_means[best_baseline]:.4f})")
    S_mean = float(np.nanmean(S_per_fold_mean))
    print(f"S mean AUROC (per-fold-averaged over 3 scJDO seeds): {S_mean:.4f}")
    delta = S_mean - baseline_means[best_baseline]
    ci_best = pair_results[f"S_vs_{best_baseline}"]
    print(f"S − best baseline: {delta:+.4f}  95%CI [{ci_best['ci_lo']:+.4f}, {ci_best['ci_hi']:+.4f}]")

    # Verdict per prereg
    primary_pass = (delta >= 0.03) and ci_best["excl_zero"] and (ci_best["ci_lo"] > 0)
    sec1 = S_mean > baseline_means["C"]  # Jacobian beats local covariance
    sec2 = S_mean > baseline_means["F"]  # scJDO beats own input rep
    if primary_pass and sec1 and sec2:
        verdict = "PASS Gate 1"
    elif primary_pass and (sec1 or sec2):
        verdict = "PARTIAL Gate 1 pass (primary met; not all secondaries)"
    elif primary_pass and not (sec1 or sec2):
        verdict = "AMBIGUOUS: primary met but S beats neither C nor F individually"
    else:
        verdict = "FAIL Gate 1"
    print(f"\nVerdict: {verdict}")

    # Save
    (OUT / "gate1_eval_summary.json").write_text(json.dumps({
        "baseline_means": baseline_means,
        "S_mean_per_fold_average": S_mean,
        "S_per_seed_means": {f"S_seed{s}": float(np.nanmean(results[f'S_seed{s}']))
                              for s in sorted(S_seeds)},
        "delta_vs_best_baseline": delta,
        "best_baseline": best_baseline,
        "pair_ci": pair_results,
        "verdict": verdict,
        "n_folds": len(results["E"]),
        "n_scjdo_seeds": len(S_seeds),
        "cohort_size": int((y >= 0).sum()),
        "n_clones": int(len(np.unique(groups))),
    }, indent=2))
    # per-set AUROC arrays
    np.savez(OUT / "gate1_eval_aucs.npz",
             **{k: v for k, v in results.items()})
    print(f"\n[written] gate1_eval_summary.json, gate1_eval_aucs.npz")


if __name__ == "__main__":
    main()
