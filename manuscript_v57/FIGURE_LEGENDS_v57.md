# FIGURE_LEGENDS_v57.md

Legends for the five main-text figures of `manuscript_v57/scJDO_v57_draft1.md`.

## Fig 1 — Lyapunov gauge, price schedule, and cross-comparison

**A.** Schematic of the Lyapunov-gauge decomposition of the drift Jacobian: `J(x) = (−D/2 + A) Σ^{−1}`. Colour-coded: symmetric part (blue) — a covariance-derived object, identifiable from the snapshot; antisymmetric part `A` (red) — not identified from the snapshot, encodes flow direction, oscillation phase and saddle timing.

**B.** Price of the antisymmetric part: schematic bars showing which experimental augmentation recovers which class of output. Left to right, top to bottom: `A(x)` for any point requires `d − 1` static perturbations of independent inputs; the sign of rotation requires one time-resolved perturbation; per-cell velocity is recovered by metabolic labelling (provided the reference is not a linear function of the input representation, see §2.5); lineage displacement is recovered by clonal barcoding. Bars are schematic (not scaled to experimental cost).

**C.** Estimator cross-comparison on the marrow Ery calibration substrate. Rows compare scJDO's Jacobian archetypes / whitened symmetric part to: local covariance, local precision, the Lyapunov-gauge prediction `−½ Σ^{−1/2} D̂ Σ^{−1/2}`, and local precision. Numbers cite ledger IDs EL01–EL03. Whitened symmetric part matches the Lyapunov prediction moderately (|cos| 0.454, Frobenius corr +0.468) and matches local precision poorly (|cos| 0.224, corr −0.368) — theory-predicted symmetric structure is what the estimator recovers.

## Fig 2 — Synthetic estimator fidelity on toggle-SDE ramp

**A.** Learned Re λ_max as a function of pseudotime (red, kernel-aggregated Jacobian tensor at bandwidth h* = 0.05, averaged over 3 seeds) overlaid on the analytic curve (black). The eigenvalue crossing at τ_crit = 0.5 is localised within one bandwidth (interior-crossing benchmark threshold `|τ̂ − τ_crit| ≤ 0.05`).

**B.** Bandwidth selection via `S(h) = R(h) · C(h) · L(h)`: bootstrap reproducibility (blue), peak contrast (green), peak localisation (orange), and their product (black dashed). The optimum h* = 0.05 (grey line) is chosen automatically.

Numbers derived from V2-fixed benchmark protocol; see §2.2 and Methods.

## Fig 3 — Real-data stability on marrow Ery

**A.** Fraction of matched signed cosines ≥ 0.85 (Hungarian one-to-one matching to per-fold reference) for REAL fits (6 fits, 3 folds × 2 seeds), block-permutation null (30 fits, 10 draws × 3 folds), and circular-shift null (same). Both nulls yield HIGHER stability fractions than REAL because kernel-averaging over a marginally-preserved pseudotime distribution collapses to a trivially consistent operator (diagnostic; see §2.3 text) [EL15].

**B.** Held-out reconstruction gain(K) as a function of dictionary size K on the discovery fold, evaluated on the two validation folds' Jacobian tensors via non-negative least-squares activations. Blue: real gain per K. Orange dashed: block-null 95th percentile per K. Only K = 1 exceeds the null band (real 0.938 vs null 0.937). K_eff = 1 [EL15].

**C.** Whitened symmetric part of the fold-0 seed-0 Jacobian compared to the Lyapunov prediction `−½ Σ^{−1/2} D̂ Σ^{−1/2}` (left pair) and to local precision `Σ^{−1}` (right pair). Green: median leading-eigenvector |cos| across τ. Red: mean Frobenius correlation across τ. Lyapunov prediction is the closer match on both metrics; local precision is anti-correlated in Frobenius sense [EL03].

## Fig 4 — LARRY lineage ground truth

**A.** Day-2 fate prediction (Neu vs Mono of the eventual clone) — mean AUROC and error bars per feature set (5-fold clone-grouped CV × 3 CV seeds; per-fold-value error bars). N = 154 cells across 106 clones. Feature sets: E_PCA (30 principal components), E_FA (30 factor-analysis components), E_scVI (30-dim scVI latent) — all reach ~0.83. S_FA (6 scJDO features per cell, from FA-space fit) reaches 0.51 with wide across-scJDO-seed spread. E_FA + S_FA equals E_FA within noise [EL10].

**B.** Gate 3 strict — soft-mode falsification on 520 LARRY clones with ≥ 5 mature descendants. Arm 1 (FA30 + N) AUROC 0.646; Arms 2–4 add local covariance features (C, top-3 eigvals of Σ_local + softness ratio) and Mojtahedi Ic. All arms are indistinguishable; 95% paired-bootstrap CIs on ΔAUROC vs Arm 1 include zero [EL11].

**C.** Sanity: per-day-2-cell softness ratio scattered against day-2 Palantir pseudotime. Spearman correlation ≈ +0.47; features track pseudotime but are redundant with expression at the clone level. Confirms failure in Panels A/B is not a broken pipeline.

## Fig 5 — Labelling ground truth and known rotation

**A.** Temporal contrast `‖J(τ) − J̄‖_F / ‖J̄‖_F` per grid point on scNT-seq neuron labelling data, for R (labelling reference, black), G (geometry-only scJDO, dashed blue), L (velocity-matching scJDO, dashed red). L median 0.532 vs R 0.234 (ratio 2.27) — L is more time-varying than R itself. Across-seed spread of L (0.011) is half of G's (0.021) [EL06].

**B.** Per-window absolute cosine between L's leading Jacobian eigenvector and R's leading eigenvector, averaged over 3 L seeds. Median over τ = 0.974; L's leading direction agrees with R per-window [EL06].

**C.** Antisymmetric-Jacobian component projected on the cycle plane (A_12), median over τ, per arm on the synthetic 30-D circular substrate (N = 1500 cells). At vel_scale = 0, both forward-τ and reverse-τ arms yield magnitudes ≤ 0.05 (grey bars, no reversal). At vel_scale = 2, forward-τ yields +0.149 and reverse-τ yields −0.204 (red / blue bars, clean sign flip). Error bars = across-3-seed spread [EL05, EL14].

**D.** Interpretive summary: the velocity-matching loss produces a time-varying seed-stable operator whose leading direction agrees with R per-window; but on this dataset R is nearly a linear function of X_pca (RidgeCV(X_pca) → v_ref reaches R² 0.92 vs scJDO J_L features 0.43 [EL13]). Rotation direction is prior-selected, not snapshot-determined.
