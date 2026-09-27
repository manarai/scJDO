# REPORT — Gate 2: scNT-seq metabolic labelling

**Prereg**: `PREREG_Gate2.md` (frozen through amendment `6f1d1cb`).
**Compute**: ~22 min (dynamo preproc + moments 12 s; 3 × G scJDO fits ~11 min; 3 × L scJDO fits ~11 min; metrics <1 min).

## Verdict — **FAIL Gate 2**

All three pre-declared predictions failed in the predicted direction.

| Prediction | Threshold | Observed | Pass? |
|:---|:---:|:---:|:---:|
| P1a G median cos(leading eigvec, R) | ≥ 0.50 | 0.340 | no |
| P1b Spearman(Re λ_max_G, Re λ_max_R) | ≥ 0.50 | 0.125 | no |
| P2 mean cos(vel_L, R) > mean cos(vel_G, R) with CI excl. 0 | > 0, CI positive | −0.234 vs −0.176; Δ_L−G = −0.057, 95 % CI [−0.072, −0.042] | no (reverse) |
| P3 median var_L / var_G < 1 with 95 % CI < 1 | < 1 | 16.62, 95 % CI [16.36, 16.80] | no (reverse) |

## Detailed numbers

- **P1 — data-determined output alignment (Jacobian leading eigvec + Re λ_max curve)**:
  - G (vel_scale = 0) vs R: median cos(leading eigvec across 150 bins) = 0.340; Spearman(Re λ_max_G, Re λ_max_R) = 0.125.
  - L (vel_scale = 2, V_ref = R) vs R: median cos = 0.030; Spearman = −0.578. Adding the V_ref prior does not move the Jacobian leading eigvec closer to R — in fact it drifts orthogonal to R and inverts the Re λ_max curve's monotonicity.
- **P2 — prior-determined output (per-cell velocity direction cosine to R)**:
  - mean cos(vel_G, R) = −0.176; mean cos(vel_L, R) = −0.234.
  - Both arms produce per-cell drift essentially uncorrelated with (or anti-correlated with) R's labelling-based velocity.
  - Δ = L − G = −0.057 with 95 % CI [−0.072, −0.042] (paired bootstrap over cells, N = 3,060, 2 000 samples). CI is entirely negative — the L arm is significantly WORSE than G.
- **P3 — seed stability of per-cell drift**:
  - Cross-seed variance of per-cell velocity: var_G = 0.016, var_L = 0.306. Ratio = 16.6.
  - 95 % CI on median ratio (bootstrap over cells): [16.36, 16.80]. Entirely > 1.
  - The V_ref anchor increases seed-to-seed spread by an order of magnitude rather than reducing it.

## Interpretation

The scJDO operator on this substrate does not align with a labelling-derived reference at either the data-determined level (P1) or via V_ref-anchored supervision (P2, P3). The three failures are not independent:

1. The scJDO Jacobian leading eigvec on this all-Neuron substrate is essentially orthogonal to the labelling-based leading eigvec (median |cos| = 0.34). Whatever geometric operator scJDO recovers from kNN + kernel windowing is not the same as the biosynthesis-rate direction indicated by new mRNA.

2. Injecting the labelling-based V_ref via monkey-patch of `_pseudotime_velocity` — the same mechanism `fit_drift_branches` uses for its terminal bias — does not reproduce Dynamo-style supervision. The DriftField's additive `vel_scale × gate × V_ref` term at magnitudes |v_ref| ≈ 13 (much larger than the k-NN pseudotime gradient's unit scale scJDO expects) destabilises the fit: L's per-cell drift becomes seed-random (var_L ≫ var_G).

3. The scale mismatch is a real implementation weakness. However, P1 also fails on shape (median cos = 0.34 < 0.5), which is scale-invariant. Even with a correctly scaled V_ref, the leading eigvec disagreement would remain: the labelling operator and the scJDO Jacobian operator on this substrate are structurally different objects.

## Compute-scale amendments (all pre-declared, all recorded before Gate 2 metrics)

- R reformulated to `v_ref = log1p(M_n_hvg) @ PCs_hvg` (labelling ratio projected to PCA-30). Dynamo's kinetic estimation collapsed to "4 genes have finite velocity" on this dataset — `experiment_type='one-shot'` expects `time` to encode labelling duration but this dataset's `time` is post-stimulation time. Amendment `6f1d1cb` records this fix before any metric.
- `time` values converted from minutes to hours before dynamo preproc (numeric stability).
- scJDO fits at `vel_scale=0` (G) and `vel_scale=2` (L), `n_epochs=3000`, `grid_size=150`, `n_archetypes=5`, matching Gate 0c defaults.

**Sensitivity of the P1 failure to these amendments**: the P1 result is invariant to the specifics of R's velocity magnitude (leading eigvec + Re λ_max curve are direction-only metrics on the operator). The operator disagreement at 0.34 median cos would survive any V_ref rescaling.

## Downstream decision per protocol

Combined outcome:
- Gate 0: PASS.
- Gate 1: FAIL.
- Gate 2: FAIL.

Per the frozen decision table this places us in the {Pass, Fail, Fail} cell:
> **Calibration paper with negatives.**

## Deliverables

- Prereg: `PREREG_Gate2.md` (frozen through amendment `6f1d1cb`).
- Runner: `run_gate2.py`.
- Machine-readable summary: `gate2_summary.json`.
- Arm-level artefacts: `gate2_arms.npz` (J_R, R.v_ref, X_pca, tau, t_centers, G_seed{0,1,2}_J, L_seed{0,1,2}_J).
- Log: `run_gate2.log`.

**Next**: consolidate all four gate reports into a single top-level decision document, then execute the deferred manuscript corrections that were held pending gate outcomes.
