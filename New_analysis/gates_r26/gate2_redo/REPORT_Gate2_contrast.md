# REPORT — Task 1: Gate 2 redo temporal-contrast reinterpretation

**Reading B applies**: velocity supervision yields a time-varying operator that removes seed dependence.

**Prereg**: `PREREG_Task1_contrast.md` (frozen `168a4f3`, before compute).
**Compute**: 2.5 s (metrics from cached Jacobian tensors + figure + held-out ridge test).

## M1 — temporal contrast per fit

`contrast_i = ‖J(τ_i) − J̄‖_F / ‖J̄‖_F` at each grid point (T = 150 bins).

| Arm | Median contrast | Across-seed SD |
|:---|:---:|:---:|
| R (reference)  | 0.2339 | — |
| G (geom-only)  | 0.3160 | 0.0213 (over 3 seeds) |
| **L (matching loss)** | **0.5315** | **0.0105** (over 3 seeds) |

- `median contrast(L) / median contrast(R) = 2.27` — L's operator is more τ-variable than R's, not less.
- L's across-seed spread (0.0105) is HALF of G's (0.0213), confirming the matching loss suppresses seed-to-seed contrast variation.

## M2 — per-window L vs R

Leading-eigenvector `|cos|` and Frobenius distance at each of 150 grid points, averaged over 3 L seeds:

- `|cos(v_L(τ), v_R(τ))|`: median-over-τ = **0.974**, mean-over-τ = 0.895.
- Frobenius `‖J_L(τ) − J_R(τ)‖_F`: median = 1.59.

L's leading direction matches R's leading direction at almost every τ (median cos 0.97). The two Jacobian tensors' leading directions agree per-window; only the magnitudes and non-leading structure differ.

## M3 — global linear baseline J_lin

`LinearRegression().fit(X_pca, R_v_ref)` on all 3060 cells → `J_lin = coef_.T` shape (30, 30).

| Comparison | mean `|cos(v_lin, v_arm(τ))|` | mean Frobenius correlation |
|:---|:---:|:---:|
| J_lin vs R (per τ) | 0.880 | +0.920 |
| J_lin vs G (per τ) | 0.304 | −0.691 |
| J_lin vs L (per τ) | 0.801 | +0.383 |

- J_lin captures R's leading direction (cos ≈ 0.88, Frob corr 0.92): R is nearly a linear map of X_pca, as noted in the Gate 2 redo P2 caveat.
- G (geometry-only, no velocity supervision) is essentially orthogonal to J_lin (cos = 0.30) and even anti-correlated in Frobenius (−0.69). G learns something totally different.
- L (velocity-matching) partially recovers J_lin's leading direction (cos = 0.80) but has weak overall Frobenius correlation (0.38) — L is time-varying whereas J_lin is constant.

## Pre-registered reading

- Threshold A required BOTH `median contrast(L) / contrast(R) ≤ 1.2` AND `cos(J_lin, J_L) > 0.95`.
- Observed: `contrast(L)/contrast(R) = 2.27` (> 1.2) AND `cos(J_lin, J_L) = 0.80` (< 0.95).
- **Reading B** applies: "velocity supervision yields a time-varying operator that removes seed dependence."

## Downstream test (Reading B → run held-out prediction)

Predict held-out cells' labelled velocity `v_ref` from either (a) per-cell J_L features vs (b) `RidgeCV(X_pca)`. 5-fold KFold on cells; 3 scJDO seeds for J_L.

- **R² from J_L features → v_ref**: mean **0.4344** (n = 15 fold-seed values). 95 % bootstrap CI [+0.4260, +0.4417].
- **R² from RidgeCV(X_pca) → v_ref**: mean **0.9197** (n = 5 folds). 95 % bootstrap CI [+0.9185, +0.9207].

RidgeCV(X_pca) beats J_L features by ~0.49 R² — a plain linear map on expression is a much better predictor of the labelling velocity reference than the labelling-supervised scJDO Jacobian features. The velocity-supervision-learned operator does yield a time-varying, seed-stable Jacobian, but it does NOT capture the labelling velocity better than a naive linear regression on expression.

## P2 status (frozen adjustment per prereg)

**P2 (cell-wise `cos(vel_L, R) > cos(vel_G, R)`) is REMOVED from Gate 2's scored predictions.**

Reason (verbatim from prereg): "R = `log1p(M_n_hvg) @ PCs_hvg` is nearly a linear function of `X_pca` on this dataset; the P2 comparison is dominated by a trivial linear-map baseline (Gate 2 redo Baseline_E reached cos = 0.99) rather than by scJDO structure. Retaining P2 in scored predictions would mis-attribute a linear-map identity to scJDO's operator."

Updated Gate 2 redo scored predictions: P1 (data-determined) and P3 (seed stability) only. P1 pass on the L arm improves substantively per M2 above (cos(v_L, v_R) median = 0.97 per-window); P3 improves substantively per M1 above (L across-seed contrast SD 0.011 vs G's 0.021, halved). Neither meets the original prereg thresholds — Gate 2 redo verdict remains FAIL — but the direction of the correction is clear.

## Interpretation

The velocity-matching loss produces a Jacobian tensor that:
1. Varies MORE across τ than R itself (contrast ratio 2.27) — the operator has substantive time-varying structure.
2. Has HALF the across-seed variance of the geometry-only arm — the supervision stabilises the fit as intended.
3. Aligns with R's leading direction per-window at cos ≈ 0.97 median — the direction of the labelling operator is well recovered.
4. But does NOT match the constant linear baseline J_lin closely (Frob corr 0.38), because L is time-varying while J_lin is constant.
5. And it does NOT beat a plain `RidgeCV(X_pca)` at held-out labelling-velocity prediction (0.43 vs 0.92 R²).

So the substantive positive from Reading B: velocity supervision on scJDO does what the loss says — produces a time-varying seed-stable operator whose leading direction agrees with R per-window. The substantive negative: this operator does not capture what a plain expression regression captures at the labelling-velocity task, because the labelling velocity on this dataset is largely a linear function of expression.

## Deliverables

- Prereg: `PREREG_Task1_contrast.md` (168a4f3).
- Runner: `task1_contrast.py`.
- This report.
- JSON: `gate2_task1_contrast.json`.
- Figure: `fig_gate2_contrast.pdf` + `.png` (3-panel: M1 contrast profiles, M2 per-window cos, M3 J_lin vs arms).
- Log: `task1_contrast.log`.
