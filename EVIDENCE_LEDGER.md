# Evidence ledger — scJDO calibration paper

One row per finding that may appear in the paper. Anything not in this ledger does not enter the manuscript.

Columns:
- **ID**: `EL##`.
- **Claim**: one-sentence statement of the finding.
- **Source**: gate / round / task producing the evidence.
- **Key number**: quantitative headline with seed count / n and 95 % CI or across-seed spread. All CIs are bootstrap unless noted.
- **Commit**: git SHA on `main`.
- **Figure/Table**: where in the paper it will be shown (v57 draft location, placeholder).
- **Status**: `established` (positive claim, evidence stands) / `qualified` (positive claim with caveats) / `negative` (a claim that a route does NOT work) / `retracted` (v56 or earlier claim withdrawn; points to superseding evidence) / `not reported` (finding known but not shown in main text).

---

## Positive / theoretical findings

| ID | Claim | Source | Key number | Commit | Fig/Table | Status |
|:---:|:---|:---:|:---|:---:|:---:|:---:|
| EL01 | scJDO's Jacobian archetypes are structurally distinct from local covariance archetypes on the same substrate. | Gate 0b (marrow Ery, r26) | mean matched cosine = 0.16 across 5 semi-NMF pairs × 3 seeds; top-15 gene-loading Jaccard ≤ 0.22; 0/5 pairs at cos ≥ 0.9 | `2ca0f09` | Fig 3 supp | established |
| EL02 | scJDO's Jacobian is also not local precision-matrix content on the same substrate. | Gate 0d v1 (r26) | precision baseline: 5 matched signed cosines, median 0.356 | `93d15f7` | Fig 3 supp | established |
| EL03 | Whitened symmetric part of scJDO's Jacobian shows partial agreement at the matrix level with the Lyapunov-gauge prediction `−½ Σ^{−1/2} D̂ Σ^{−1/2}`; the leading direction is not reliably recovered; the prediction is better than a local-precision baseline. | Gate 0d v2 (r26), Task B across-fit rescore | leading-eigvec |cos|: 0.33 ± 0.15 (predicted) vs 0.28 ± 0.07 (precision baseline); Frobenius corr: 0.46 ± 0.01 (predicted) vs −0.38 ± 0.00 (precision baseline); 6 REAL fits (3 folds × 2 seeds) at marrow Ery, mean ± SD across fits | `41a509d` (+ Task B) | Fig 3 | established |
| EL04 | scJDO's bias-strength pathway reaches the model (V_ref, weights differ under the two bias values); an earlier "bias is inert" observation was a basin-of-training phenomenon, not a plumbing failure. | Gate 0a (r26) | V_ref max |Δ| = 1.099 between A1 (bias=1.5) and B1 (bias=0) at seed=0; 50/53 state_dict tensors differ | `130642f` | supplement | established |
| EL05 | Reversing pseudotime reverses the sign of the antisymmetric part of scJDO's Jacobian on the cycle plane, under the additive pseudotime-gradient prior (`vel_scale = 2`); the DSM-only training (`vel_scale = 0`) does not carry this signature. | Task 2 (r26 chain, synthetic 30-D circular substrate) | vel2 median A_12: forward +0.149 (±0.027), reverse −0.204 (±0.044); vel0 same axis −0.015 (±0.012) and −0.050 (±0.046) — magnitudes ~10× smaller; 3 seeds per arm, N=1500 cells | `227df70` | Fig 5 | established |
| EL06 | Velocity supervision via a matching loss produces a time-varying Jacobian whose leading direction agrees with the labelling reference per-window; it also halves the across-seed variance vs geometry-only. | Task 1 (Gate 2 redo temporal-contrast) | median |cos(v_L, v_R)| = 0.974 across τ; median contrast(J_L) = 0.532 vs contrast(J_R) = 0.234; across-seed SD of median contrast 0.011 (L) vs 0.021 (G); 3 seeds each; N=3060 cells | `2c1f63f` | Fig 5 | qualified |
| EL07 | Per-cell aggregation of the learned Jacobian tracks the analytic at-cells Re λ_max curve. | r11 interior benchmark (FOLLOWUP6_aggregation on the lifted-truth synthetic) + Round A temporal-Jacobian reproducibility on marrow Ery | argmax within 0.01–0.09 of oracle argmax across 3 seeds; matrix- and per-cell-aggregation agree to within one grid step on lifted truth (h ∈ {0.005..0.04}); Round A: peak-τ 0.020 ± 0.000, pairwise curve Pearson 0.806 across 3 seeds | `ed1499c` (r11 data); Task C (Round A restore) | Fig 2C | established |

---

## Negative findings

| ID | Claim | Source | Key number | Commit | Fig/Table | Status |
|:---:|:---|:---:|:---|:---:|:---:|:---:|
| EL08 | V2-fixed 2-branch pitchfork crossing scorecard: neither oracle-at-cells nor learned-at-cells crosses zero on the interior grid at the bifurcation, while the analytic Jacobian evaluated at the symmetric fixed point crosses at τ_crit for every condition. Modality limit: no cell-evaluated Jacobian localises the bifurcation. | V2-fixed benchmark (restored) | oracle-FP crossings 6/6 within ±0.05 of τ_crit; oracle-at-cells 0/6; learned-at-cells 0/6; learned-at-estimated-FP 0/6; τ_crit ∈ {0.1, 0.3, 0.5} × 2 seeds | Task C (V2-fixed restore) | Fig 2A/B | negative |
| EL10 | scJDO per-cell operator features do NOT predict Neu-vs-Mono clone commitment on LARRY day-2 cells beyond expression baselines. | Gate 1 v2 | E_PCA 0.8254 / E_FA 0.8299 / E_scVI 0.8281 / S_FA 0.5076 / E_FA+S_FA 0.8293; ΔAUROC S_FA − E_FA = −0.32 95 % CI [−0.36, −0.28]; N=154 cells / 106 clones (5-fold GroupKFold × 3 CV seeds); S_FA 3 scJDO seeds spread 0.096 | `2d5f60e` | Fig 4 | superseded → EL10b |
| EL10b | scJDO per-cell operator features do NOT predict Neu-vs-Mono clone commitment on the LARRY day-2 full eligible cohort beyond expression baselines; the negative holds when the cohort grows from 154 cells to the full eligible set. | Task A (Gate 1 v2 full-cohort rerun; field trained on ~14K-cell day-2 subsample = 4,638 barcoded + 9,362 unbarcoded) | E_PCA 0.867 / E_FA 0.864 / E_scVI 0.879 / S_FA 0.554 / E_FA+S_FA 0.863; ΔAUROC S_FA − E_scVI = −0.325 95 % CI [−0.340, −0.309]; ΔAUROC S_FA − E_FA = −0.310 95 % CI [−0.324, −0.297]; ΔAUROC E_FA+S_FA − E_FA = −0.001 95 % CI [−0.002, +0.000]; N=970 cells / 632 clones (5-fold GroupKFold × 5 CV seeds); S_FA 3 scJDO seeds spread 0.033 | Task A commit | Fig 4 | negative |
| EL11 | Local-covariance softness ratio and Mojtahedi Ic add nothing to a Factor-Analysis expression baseline for predicting whether a LARRY day-2 clone is still mixed. | Gate 3 strict | Arm 1 (E)=0.6464; Arm 2 (E+C) ΔAUROC +0.0001 CI [−0.012, +0.012]; Arm 3 (E+Ic) +0.0004 CI [−0.002, +0.003]; Δlog-loss ≈ 0; 520 clones, 5×5 fold-seed × clone-grouped CV | `6edbb1e` | Fig 4 | negative |
| EL12 | Additive `V_ref` prior (`vel_scale = 2` slot) on the DriftField destabilises fits on scNT-seq metabolic labelling and does NOT recover the labelling reference's leading direction. | Gate 2 v1 | median cos(leading eigvec, R): G=0.34, L=0.03; Spearman(Re λ_max_L, Re λ_max_R) = −0.578; var_L/var_G = 16.6 CI [16.4, 16.8]; N=3060, 3 seeds | `504cb7b` | Fig 5 supp | negative |
| EL13 | Even with a proper velocity-matching loss, scJDO's Jacobian features do not beat a plain `RidgeCV(X_pca)` at predicting the labelling velocity reference held-out. | Task 1 held-out test | R² from J_L features = 0.4344 CI [0.426, 0.442] (n=15 fold-seed values); R² from RidgeCV(X_pca) = 0.9197 CI [0.919, 0.921] (n=5) | `2c1f63f` | Fig 5 supp | negative |
| EL14 | Vanilla scJDO with `vel_scale = 0` does NOT encode pseudotime direction on cyclic data — neither M1 (mean tangent-projected drift) nor A_12 (antisymmetric on cycle plane) discriminate forward vs reverse pseudotime at that setting. | Cell-cycle reversal (v1 + Task 2) | vel0 M1 fwd = rev = −0.00017 (no flip; symmetry-cancelled); vel0 A_12 fwd −0.015, rev −0.050 (~10× smaller than vel2 magnitudes) | `227df70` | Fig 5 supp | negative |
| EL15 | Held-out reconstruction gain against a block-permutation null on marrow Ery gives K_eff = 1: only one archetype's gain exceeds the block-null 95th percentile. | Gate 0d v2 | gains {K=1: +0.938 vs null 0.937; K=2: +0.021 vs 0.024; K=3-5: below null}; K_eff = 1; discovery + 2 validation folds, 10 block-null draws per fold, 1151 cells | `41a509d` | Fig 3 | negative |
| EL16 | The Klein LARRY in vitro release does not carry spliced/unspliced or metabolic-labelling layers, so Dynamo velocity cannot be run on it as a comparator to scJDO's Gate 1 features. | Task 4 | permanent data-availability constraint; row D omitted from Gate 1 v2 table | `b72bb1a` | not reported | not reported |

---

## Retracted / superseded v56-era claims

| ID | Retracted claim | Evidence that retracts it | Superseding ledger row |
|:---:|:---|:---:|:---:|
| EL20 | "Two attractors" describing vel_scale × bias outcomes on Fig 3 | Gate 0a shows these are training basins (basin dominated by seed), not dynamical-system attractors; terminology change required. | EL04 |
| EL21 | "Bias is inert" as a bug hypothesis on Fig 3 | Gate 0a empirical trace: V_ref and 50/53 state_dict tensors differ between the two bias settings — plumbing works, but seed dominates basin selection at these fit parameters. | EL04 |
| EL22 | Round A framed as estimator fidelity | The r11 synthetic learned-vs-oracle match is estimator fidelity. Round A is real-data eigenvalue-curve reproducibility (a separate calibration item). Terminology change required. | (calibration section; separate items in draft) |
| EL23 | Gate 0c archetype-consensus statement of "K_eff = 1" as an intrinsic property of scJDO on marrow Ery | Gate 0c used single-linkage agglomerative clustering with absolute cosine — methodologically flawed (chained everything into 1 cluster). Gate 0d v1 Hungarian showed 4/5 archetypes are consensus-stable at cos ≥ 0.7. Gate 0d v2 held-out gain confirms K_eff = 1 under block-permutation null — but via a rigorous held-out reconstruction test, not clustering. | EL15 |
| EL24 | "Sensitivity genes" language for leading-J projections | Replaced by "leading-direction loadings" throughout the manuscript per the wrap-up terminology rule. | — |
| EL25 | "Velocity prior" language for the additive V_ref slot | Replaced by "additive pseudotime-gradient prior" throughout. | — |

---

## Cohort / substrate notes (background rows for figures)

| ID | Note | Source | Commit |
|:---:|:---|:---:|:---:|
| EL30 | LARRY in vitro cohort for fate prediction: 154 day-2 cells across 106 clones after (a) subsampling substrate to 12,550 cells due to compute constraints, (b) requiring ≥ 2 late Neu ∪ Mono cells per clone with ≥ 80% purity. | Gate 1 v2 | `2d5f60e` |
| EL31 | LARRY in vitro cohort for soft-mode falsification: 520 clones with ≥ 1 barcoded day-2 cell AND ≥ 5 mature descendants at day 4/6; positive fraction of "mixed" = 0.275 under rarefied major-lineage-groups labelling; day-2 features on FULL 28,249 day-2 cells (no compute subsample). | Gate 3 strict | `6edbb1e` |
| EL32 | scNT-seq neuron labelling cohort: 3,060 cells, all `Neu`; time-since-KCl 0..120 min; 4sU pulse fixed. Dynamo kinetic pipeline collapses ("4 genes have finite velocity") on this dataset; R reformulated as `log1p(M_n_hvg) @ PCs_hvg`. | Gate 2 v1 + amendment | `8afef29` |
| EL33 | Marrow Ery calibration cohort: 1,151 cells × 16,106 genes, r24 pinned pipeline (Palantir 1.4.4 + MAGIC imputation + FA-30 rep). Reused across Gates 0a/0b/0c/0d. | (setup) | `130642f` |
