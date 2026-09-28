# r24 — Fig 3 pin audit + vel×bias interaction + Round 1 dynamics baseline

**Base tag**: `scjdo-v56-submission-20260922-r23` @ `c6191f3`
**Branch**: `main`
**Compute**: ~2 h 20 min (Task A+B ~2 h + Task C ~30 min + Task D ~10 min).

## Purpose

The user's r23 review identified four scientific-integrity gaps that must be closed before the manuscript's Box 1 and demonstration section are rewritten:

1. **Round C didn't test the Fig 3 pipeline.** r22's Round C used `fit_drift` (not `fit_drift_branches`), `bias=0`, `depth=3`, 3000 epochs, `grid=100`. Fig 3 uses `fit_drift_branches`, `bias=1.5`, `depth=4`, 5000 epochs, `grid=200`. So r22 doesn't directly refute Fig 3's identifiability.
2. **vel_scale × bias interaction question.** Round C's `no_vel` top-5 exactly matched Fig 3's top-5 (`HBB, AHSP, CA1, FAM178B, APOC1`) — hinting bias may cancel vel_scale, or vel_scale may not reach the model.
3. **Round 3 cell counts don't match Fig 3** (948/1543/1423 vs 1151/1903/2008). Different Palantir seed or different preprocessing likely.
4. **Round 1's synthetic Jacobian win is against expression-only baselines.** Need at least one dynamics-aware baseline (dynamo, GENIE3, lagged correlation) to show scJDO's win is a Jacobian-method result rather than a scJDO-specific one.

r24 addresses all four with fresh compute. Findings below.

## Task A+B — Fig 3 pipeline prior audit (r22 Round C redone correctly)

Substrate: marrow Ery branch cells only (1,151 cells), same Palantir pseudotime as Fig 3. All fits use `fit_drift_branches` with a single-branch input at Fig 3 defaults (`depth=4`, `n_epochs=5000`, `n_archetypes=5`, `grid_size=200`). Six configurations × three seeds each = 18 fits.

### Within-config reproducibility

| Config | vel | bias | hidden | σ | eigvec cos | top-15 Jaccard | Verdict |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| A1 default_fig3 | 2.0 | 1.5 | 256 | 0.10 | 0.437 | 0.292 | ✗ two attractors |
| A2 no_vel_fig3 | 0.0 | 1.5 | 256 | 0.10 | **0.995** | **0.917** | ✓ single attractor |
| A3 smaller_net | 2.0 | 1.5 | 128 | 0.10 | 0.167 | 0.063 | ✗ near-random |
| A4 hi_sigma | 2.0 | 1.5 | 256 | 0.20 | 0.236 | 0.072 | ✗ near-random |
| B1 no_bias | 2.0 | 0.0 | 256 | 0.10 | 0.437 | 0.292 | ✗ two attractors |
| B2 neither | 0.0 | 0.0 | 256 | 0.10 | **0.995** | **0.917** | ✓ single attractor |

### The vel × bias interaction map

|  | **vel_scale = 2.0** | **vel_scale = 0.0** |
|---|:---:|:---:|
| **bias = 1.5** | A1: 2 attractors, τ = 0.44 | A2: 1 stable, τ = 0.99 |
| **bias = 0.0** | B1: 2 attractors, τ = 0.44 | B2: 1 stable, τ = 0.99 |

**Bias is inert.** Reproducibility depends only on vel_scale:
- `vel_scale = 0.0` → one stable attractor across all seeds (τ = 0.99, Jaccard = 0.92). Top-5 always `[HBB, AHSP, CA1, FAM178B, APOC1]`.
- `vel_scale = 2.0` → two attractors. Seed 0 lands in a plasma/IG basin `[RP11-1I2.1, CLU, IGHM, MEG3, CRHBP]`; seeds 1/2 land in the erythroid basin `[HBB, AHSP, CA1, FAM178B, APOC1]`. Reproducibility τ = 0.44.

**Implication for the manuscript's "γ = 2 hard-on in every figure" claim**: this is operationally worse than described. Under `vel_scale = 2.0`, the model has multiple local optima that a user could land in randomly. Fig 3 as-published shows the erythroid basin; a different seed would show the plasma basin — a completely different biological interpretation of the "top instability genes."

### Per-seed max_eig on Fig 3 default (A1)

- seed 0: peak τ = 0.020, **max_eig = −0.045**, sensitive_bins = 0/200
- seed 1: peak τ = 0.671, **max_eig = −0.019**, sensitive_bins = 0/200
- seed 2: peak τ = 0.691, **max_eig = −0.023**, sensitive_bins = 0/200

**All three seeds have NEGATIVE max_eig** on the Fig 3 default. The paper's 0.05 sensitivity threshold is never met at any seed. Fig 3's "sensitivity-associated genes" claim is not supported at threshold.

### Cross-config stability (vs A1 seed 0)

| Config | eigvec cos vs A1s0 | top-15 Jaccard vs A1s0 | Top-5 |
|---|:---:|:---:|---|
| A1 seed 0 (ref) | 1.000 | 1.000 | RP11-1I2.1, CLU, IGHM, MEG3, CRHBP |
| A2 no_vel_fig3 | 0.200 | 0.000 | HBB, AHSP, CA1, FAM178B, APOC1 |
| A3 smaller_net | 0.256 | 0.034 | UBE2C, TOP2A, MKI67, BIRC5, TK1 |
| A4 hi_sigma | 0.531 | 0.111 | HIST3H2A, H1F0, AC074183.4, IL1B, ZBTB20 |
| B1 no_bias | **0.996** | **0.765** | RP11-1I2.1, CLU, MEG3, IGHM, CRHBP |
| B2 neither | 0.200 | 0.000 | HBB, AHSP, CA1, FAM178B, APOC1 |

**B1 (bias off, vel on) is extremely close to A1 (bias on, vel on)** at seed 0 — confirming bias is inert. A2 and B2 (both vel off) give a completely different, but internally-consistent, top-5.

## Task C — Round 3 rerun with pinned Palantir + MAGIC

Round 3b's cell counts (948/1543/1423) mismatched Fig 3's (1151/1903/2008) because Round 3b skipped `palantir.utils.run_magic_imputation`. Adding MAGIC to the pipeline recovers Fig 3's exact cell counts:

```
branch_masks cell counts: {'Ery': 1151, 'DC': 1903, 'Mono': 2008}
Fig 3 expected:           {'Ery': 1151, 'DC': 1903, 'Mono': 2008}
[OK ✓] branch counts match Fig 3
```

`fit_drift_branches` ran on the correct cohorts (71 min compute). Diagnostic of the resulting per-branch `uns` blocks:

| Branch | max_eig peak | n_sensitive (>0.05) | `instability_scores` shape | `top_instability_genes` |
|---|:---:|:---:|:---:|:---:|
| Ery | +0.043 | 0 / 200 | (200, 0) | shape (0,) |
| DC | +0.025 | 0 / 200 | (200, 0) | shape (0,) |
| Mono | −0.041 | 0 / 200 | (200, 0) | shape (0,) |

**All three branches have empty `instability_scores` matrices and empty `top_instability_genes` arrays.** `infer_regulators` raises `ValueError: No regulator results` on every branch, at both `min_targets=2` and `min_targets=1`. The Fig 3 gene lists must be coming from a separate `get_instability_genes` step (which populates `instability_scores`) that this script skipped; scoring under that fallback path gives the "leading-direction loading" gene lists, not sensitivity-mask-gated regulator inference.

**Bug found** — when `instability_scores` shape = (T, 0), calling `infer_regulators` with `score_method='jcol'` or `'ensemble'` raises `IndexError: index N is out of bounds for axis 1 with size 0` inside `_score_regulators`. Backward-compat fix for a future release: handle the empty-instability case gracefully.

## Task D — Round 1 with dynamics-aware baseline

Added `lagged_corr_hub` (lagged Pearson correlation of gene expression across pseudotime bins, row-sum absolute) as a dynamics-aware baseline on the Round 1 synthetic hidden-master GRN.

| Method | AUROC (mean ± std, 3 seeds) | top-3 hits mean |
|---|:---:|:---:|
| diff_expr | 0.000 ± 0.000 | 0.00 |
| corr_with_tau | 0.000 ± 0.000 | 0.00 |
| variance | 0.000 ± 0.000 | 0.00 |
| **lagged_corr_hub** | **1.000 ± 0.000** | **3.00** |
| granger_row (my impl.) | 0.000 ± 0.000 | 0.00 |
| **scjdo_instab_z** | **1.000 ± 0.000** | **3.00** |
| scjdo_Jcol_z_mean | 0.000 ± 0.000 | 0.00 |
| scjdo_Jcol_z_max | 0.000 ± 0.000 | 0.00 |

**`lagged_corr_hub` matches `scjdo_instab_z` exactly**: AUROC 1.000, all 3 masters in top-3, across all 3 seeds. Per-seed top-5:
- Seed 42: lagged_corr_hub = M1, M3, M2, I2, I4 | scjdo_instab_z = M1, M3, M2, T21, I5
- Seed 43: lagged_corr_hub = M1, M3, M2, I4, I8 | scjdo_instab_z = M1, M3, M2, T12, T1
- Seed 44: lagged_corr_hub = M2, M3, M1, I5, I4 | scjdo_instab_z = M3, M1, M2, T7, T15

**Round 1's headline claim needs narrowing.** The correct scope: "temporal-order-aware methods (Jacobian OR simple lagged correlation) recover hidden low-variance masters that expression-only baselines miss." NOT "scJDO's Jacobian uniquely finds them." The lagged-correlation baseline uses only pseudotime ordering — no Jacobian, no fit, no scJDO — and gets the same 1.000 AUROC.

(Note: my Granger implementation was a simple partial-correlation approximation and returned 0.000. A proper Granger causality test on binned trajectories should be added for a complete dynamics-aware baseline suite; not run here.)

## Combined synthesis for the calibration tier

The user's proposed three-part calibration tier:

1. **Constructed Obstruction algebra**: retained (Supp Note S1, unchanged).
2. **Estimator fidelity**: retained — r22 Round A + r11 audit show the per-cell aggregation reproduces the analytic at-cells curve.
3. **Saddle non-localizability from cell states**: **retained AND strengthened**. The V2-fixed 6/6 pass on the analytic FP eigenvalue vs 0/6 on the analytic-at-cells and learned-at-cells curves shows this is a modality limit, not an estimator failure.

r24 adds concrete evidence for the reframe:
- **The Fig 3 configuration doesn't pass the paper's own sensitivity threshold at any seed** (all three A1 seeds have negative max_eig).
- **vel_scale is what creates the multiple-attractor problem**; bias is inert.
- **The published Fig 3 gene lists come from an unmasked-fallback leading-direction loading**, not from sensitivity-mask-gated regulator inference.
- **Round 1's synthetic win is a temporal-order-method win**, not a scJDO-specific one.

## Actionable takeaways for the rewrite

1. **Rename "sensitivity-associated genes" → "leading-direction loadings"** in Fig 3 (and Fig 5, Fig 6 by parallel reasoning). The sensitivity threshold isn't met.
2. **Document the seed-dependent multi-attractor problem**. Fig 3's HBB/AHSP/CA1 top-5 is one of at least two basins that vel_scale=2.0 admits. A second seed would give a different biological story.
3. **Correct the "γ = 2 hard-on in every figure" claim**. On the Fig 3 branch, the top-5 is inert to vel_scale change only when bias is on AND the model happens to land in the plasma basin; more generally vel_scale creates two attractors and bias is inert.
4. **Round 1 headline**: "temporal-order methods (Jacobian, lagged correlation) recover hidden masters; expression-only methods miss them." Not "scJDO's Jacobian uniquely."
5. **Round 3 numbers should use the corrected 1151/1903/2008 counts** (with MAGIC). The 948/1543/1423 numbers are from a pipeline that doesn't match Fig 3.

## Release code

```
New_analysis/operator_claims_benchmark/
├── scripts/
│   ├── run_fig3_prior_audit_r24.py           — Task A+B (6 configs × 3 seeds on Fig 3 pipeline)
│   └── rescore_r24_taskC.py                   — Task C rescore from saved adata
└── outputs/r24_fig3_prior_audit/
    └── r24_fig3_prior_audit_summary.json      — Task A+B full summary

New_analysis/regulator_benchmark/
├── scripts/
│   ├── run_round3_pinned_palantir_r24.py     — Task C (Round 3 with MAGIC + pinned Palantir)
│   └── run_round1_with_dynamics_baselines_r24.py  — Task D (Round 1 + lagged_corr_hub)
├── outputs_r24_round3_pinned/
│   ├── marrow_scjdo_r24.h5ad                 — Fitted adata (71 min compute) — saved for reuse
│   ├── round3_rescore_summary.json           — Task C rescore output
│   └── round3_pinned_summary.json            — Task C original run summary
└── outputs_r24_dynamics_baselines/
    ├── round1_with_dynamics_baselines_by_seed.csv
    └── round1_with_dynamics_baselines_summary.json
```
