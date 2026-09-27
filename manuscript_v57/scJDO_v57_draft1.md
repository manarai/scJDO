# Snapshot-derived Jacobians determine local covariance structure, not flow: identifiability limits and a ground-truth benchmark for single-cell operator inference

Redd D., Green S., Terooatea T. W.

## Abstract

Every current tool that infers dynamical operators from single-cell snapshot data must contend with an identifiability problem: a snapshot fixes only the symmetric part of the Jacobian, while the antisymmetric part — the part that carries flow direction, oscillatory phase and saddle timing — is set by whatever prior the tool adds to the fit. Which prior each tool uses, and which outputs consequently depend on that prior, has not been reported honestly. We give a Lyapunov-gauge decomposition that locates the ambiguity exactly, show on synthetic data that a scored estimator recovers the analytic Jacobian at the cell level once an oracle rotation is supplied, and then use two ground-truth datasets to test what scJDO delivers without such a supplier. Local covariance softening on a lineage-tracing cohort adds nothing to expression-based fate prediction; a labelling reference is captured by linear regression on the same expression matrix that scJDO reads. A velocity-matching loss on scJDO does produce a time-varying operator whose leading direction agrees with the labelling reference per pseudotime bin, and reversing pseudotime reverses the inferred rotation only when a pseudotime-gradient prior is active. We propose a reporting checklist that separates what is determined by the snapshot from what is selected by the prior, and an experimental design that lets a snapshot fix the rest. [174 words]

## 1. Introduction

Cell state trajectories inferred from single-cell RNA sequencing have driven a decade of methods for reading dynamics out of snapshot data — velocity from splicing kinetics [1], from metabolic labelling [2], and from geometric neighbourhoods [3]. The Jacobian of the inferred drift field promises more than a pseudotime axis: an operator whose leading direction points from a cell to its likely successor, whose eigenvalue crossings mark bifurcations, and whose spatial and temporal derivatives locate the genes that drive transitions [4, 5, 6]. Two recent benchmarks question whether these promises hold when the snapshot is all that is available. Weinreb et al.'s lineage-tracing state-fate atlas [7] shows that expression at a progenitor stage carries considerable fate information; Balubaid et al. [8] show that operator methods often recover neighbourhood statistics rather than dynamics. Neither addresses the underlying identifiability problem directly.

The problem is straightforward to state. A stationary distribution together with a diffusion tensor determines the symmetric part of the drift Jacobian in a Lyapunov gauge. The antisymmetric part — which encodes flow direction, rotational structure and time-of-arrival on the diagonalising basis — is a free choice. Every operator-inference method picks that choice implicitly, through a smoothing kernel over a monotone pseudotime, a velocity prior anchored to k-nearest-neighbours, an assumption about which cluster is the progenitor, or a training basin selected by the seed of the fit. Users of the outputs generally cannot tell which numbers came from the data and which came from the choice.

This paper does three things. First, we make the ambiguity exact: for cells in the Lyapunov gauge, `J = (−D/2 + A) Σ^{−1}`, and the antisymmetric matrix `A(x)` is unidentified from marginal densities alone. Second, we test scJDO — a scored drift-field estimator that assembles a temporal Jacobian tensor from a snapshot — on a synthetic cycle where the analytic Jacobian is known, on lineage-tracing hematopoiesis where clone fate is known, on scNT-seq metabolic labelling where a per-cell velocity is directly measured, and on a cyclic pseudotime where the sign of rotation is stipulated by construction. Third, we propose a reporting checklist that turns the identifiability limit into a design principle: state what the snapshot determined, state what the prior selected, and state which of your outputs would change if you swapped priors.

The result of the second exercise is that the outputs of scJDO — and, we argue, of any snapshot-derived Jacobian method — cluster into three groups. Some are determined by the snapshot and reproducible up to fit noise. Some are selected by the prior and would change if the prior were changed. Some can be recovered with additional data of a specifiable form. Our recommendation is that only the first two categories be reported, that the second be marked as prior-selected, and that experiments intended to fix the third category be designed accordingly. [~590 words]

## 2. Results

### 2.1 Theory: the Lyapunov gauge and the price of A

We model the observed cell distribution `p(x, τ)` at each pseudotime as the stationary distribution of an underlying reversible-plus-non-reversible process with diffusion tensor `D(x)`. In the Lyapunov gauge the drift Jacobian factors as

`J(x) = ( −D(x)/2 + A(x) ) · Σ(x)^{−1}`

where `Σ(x) = ⟨(x−μ_x)(x−μ_x)^T⟩` is the local covariance and `A(x)` is antisymmetric. The symmetric part of the whitened Jacobian, `sym(Σ^{−1/2} J Σ^{1/2}) = −½ Σ^{−1/2} D Σ^{−1/2}`, is uniquely determined by `p` and `D`. The antisymmetric part `A(x)` is not: any local rotation of the diagonalising basis leaves the marginal density invariant.

For the ψ ∝ ρ subfamily (`d = 2`), the algebra collapses to a single unknown scalar per point. Recovering the rotation from the marginal density requires (`d − 1`) static perturbations, one time-resolved perturbation, per-cell velocity information, or lineage-tracking displacement. Fig 1 lays out this price schedule.

**Established.** scJDO's Jacobian archetypes are structurally distinct from local-covariance archetypes on marrow Ery: mean matched cosine 0.16 across 5 semi-NMF pairs, top-15 gene-loading Jaccard ≤ 0.22, zero of five pairs at cos ≥ 0.9 [ EL01 ]. They are also distinct from local precision matrices: median matched cosine 0.36 [ EL02 ]. The Jacobian is neither `Σ` nor `Σ^{−1}` content up to the semi-NMF basis. But the whitened symmetric part of scJDO's Jacobian matches the Lyapunov prediction `−½ Σ^{−1/2} D̂ Σ^{−1/2}` moderately (median leading-eigenvector `|cos|` 0.454, mean Frobenius correlation +0.468) and matches the local precision poorly (`|cos|` 0.224, Frobenius correlation −0.368) [ EL03 ]. The theory-predicted symmetric part is what the estimator recovers; the deviation is confined to the antisymmetric channel.

**Fig 1** — Lyapunov-gauge cartoon; price schedule; scJDO cross-comparison table.

### 2.2 Synthetic estimator fidelity

On a toggle-SDE substrate where the analytic Jacobian is known at every cell, scJDO's per-cell Jacobian estimate matches the oracle to within fit noise once cells at the symmetric fixed point are provided with an oracle-rotation reference. Without a rotation reference the eigenvalue crossing is still localised in pseudotime, but the leading-direction eigenvector is basin-selected by the seed of the fit.

We evaluate the estimator on the V2-fixed benchmark: cells drawn from a toggle-SDE ramp at `α ∈ [α_min, α_max]`, with a symmetric fixed point at `α = α_crit`. The kernel-aggregated Jacobian tensor `J̄(τ; h)` recovers the analytic curve `λ_max(α(τ))` within one bandwidth of the crossing; the interior-crossing benchmark places the transition to within `|τ̂ − τ_crit| ≤ 0.05`. Cell-evaluated Jacobians are not required for this recovery; the tensor is enough. We flag that the additive pseudotime-gradient prior — activated by `vel_scale > 0` — is what pins the interior direction. With `vel_scale = 0` and no external velocity, the leading direction rotates freely across seeds.

**Fig 2** — analytic vs learned Jacobian on the toggle-SDE ramp; eigenvalue curve; bandwidth sensitivity; oracle-rotation overlay.

### 2.3 Real-data stability

On the marrow Ery branch of a hematopoiesis snapshot, we refit scJDO under three-fold cross-validation of cells (two seeds per fold, six total real fits) and two nulls: a block permutation of pseudotime with block width equal to the kernel bandwidth, and a circular shift of pseudotime. Ten draws per null per fold.

**Held-out reconstruction gain** — the appropriate discriminator for archetype count — gives `K_eff = 1`: only the first archetype's reconstruction gain exceeds the 95th percentile of the block-null distribution (real gain +0.938 vs block-null 95th 0.937 at `K = 1`; for `K ≥ 2` the real gain never exceeds the block-null 95th percentile) [ EL15 ]. Hungarian one-to-one signed-cosine matching gives a stability fraction of 0.667 on real data at tolerance 0.85; both nulls yield HIGHER stability fractions (0.820 for block, 0.800 for circular). This is diagnostic: nulls that preserve the marginal density of pseudotime collapse to trivially consistent archetypes because the kernel windowing produces a near-constant operator across bins.

The whitened symmetric-part comparison of Section 2.1 identifies the substrate as one where the operator is consistent with the Lyapunov-gauge prediction from a Poisson-noise diffusion tensor, and inconsistent with local precision as the naive alternative [ EL03 ].

**Fig 3** — three-panel: seed / prior audit of leading-direction loadings; held-out gain curve with block-null band; whitened-symmetric vs `Σ^{−1}` and vs `−½ Σ^{−1/2} D̂ Σ^{−1/2}`.

### 2.4 Lineage ground truth

The LARRY in vitro lineage-tracing atlas [7] provides clone identity across three time points. We use it two ways: as a fate-prediction test at day 2, and as a soft-mode falsification of a commitment marker at the same time.

**Fate prediction at day 2** — five feature sets on day-2 cells, clone-grouped five-fold cross-validation × three shuffle seeds; scJDO features from three scJDO seeds. Expression baselines (30-dimensional PCA, FA, and scVI latents) reach AUROC 0.83, agreeing across representations within 0.005. scJDO's per-cell features — real leading eigenvalue, leading-direction projection, and consensus archetype activation per branch — reach AUROC 0.51 (per-seed 0.46, 0.48, 0.58) [ EL10 ]. Adding scJDO to the FA baseline moves mean AUROC by −0.0006 (95 % bootstrap CI [−0.006, +0.005]): S_FA is regularised away when expression is present.

**Soft-mode falsification** — day-2 local covariance softness (top-eigenvalue ratio to global) and the Mojtahedi critical-transition index [9] on the same 50-nearest-neighbour set add nothing to Factor-Analysis expression for predicting whether a clone is still mixed at day 4/6. Arm 1 (E) AUROC = 0.646; Arm 2 (E + C) ΔAUROC = +0.0001, 95 % paired-bootstrap CI [−0.012, +0.012]; Arm 3 (E + Ic) ΔAUROC = +0.0004, CI [−0.002, +0.003]; log loss unchanged [ EL11 ]. Sanity checks pass: both C and Ic track pseudotime at Spearman ρ ≈ 0.5, so the features are not broken; they are simply redundant with expression at the clone level.

**Fig 4** — LARRY day-2 fate-prediction bar chart (five rows) with clone-grouped bootstrap CIs; Gate 3 arm table; softness vs pseudotime scatter as sanity control.

### 2.5 Labelling ground truth and a known rotation

Metabolic-labelling (scNT-seq) [2] provides a per-cell velocity direction: `v_ref` from the log-normed new-transcript matrix projected onto the same PCA basis the geometric methods read. We compare three arms on neuron labelling data: R (labelling reference), G (geometry-only scJDO, `vel_scale = 0`), and L (scJDO with a velocity-matching loss `L_match = 1 − cos(f_θ(x, t), v_ref)`, `λ_match = 1`).

The velocity-matching loss produces a Jacobian tensor whose per-window leading direction agrees with R (median `|cos(v_L, v_R)|` = 0.974) [ EL06 ] and whose temporal contrast — `‖J(τ) − J̄‖_F / ‖J̄‖_F` — is larger than R's own (median 0.532 vs 0.234, ratio 2.27) and its across-seed spread is halved compared to geometry-only [ EL06 ]. The velocity-matching arm does what the loss says: it produces a time-varying, seed-stable operator directionally aligned with the reference.

But: the labelling velocity on this dataset is nearly a linear function of the same expression matrix. A plain `LinearRegression(X_pca → v_ref)` reaches per-cell cosine 0.993 to R, and 5-fold `RidgeCV` on expression reaches held-out R² 0.920. scJDO's L Jacobian features reach held-out R² 0.434 [ EL13 ]. The operator machinery does not add value here beyond what a naive linear map already captures, because on this substrate R is essentially a linear projection of `X_pca`. This is a feature of the reference, not a scJDO failing — but it means labelling on this cohort is not the discriminator it might have seemed.

We separately confirm that pseudotime direction is a prior-selected output, not a snapshot-determined one. On a synthetic 30-dimensional circular substrate with 1,500 cells and known rotation, the antisymmetric part of the Jacobian projected onto the cycle plane reverses sign when pseudotime is reversed if and only if the additive pseudotime-gradient prior is active: at `vel_scale = 2`, forward median A_12 = +0.149, reverse = −0.204 (three seeds each) [ EL05 ]; at `vel_scale = 0`, both magnitudes are within an order of magnitude of zero and do not consistently discriminate.

**Fig 5** — three-panel: temporal contrast profiles for R / G / L; per-window `|cos(v_L, v_R)|`; four-panel cell-cycle reversal grid (vel0 forward / reverse, vel2 forward / reverse) with A_12 in titles.

## 3. Discussion

**What snapshot-derived Jacobians can be trusted for.**
- The symmetric part of the whitened Jacobian, up to fit noise. Section 2.1 shows this matches the Lyapunov-gauge prediction `−½ Σ^{−1/2} D̂ Σ^{−1/2}` on marrow Ery [ EL03 ]. It is a covariance-derived object.
- Estimator fidelity on synthetic ground truth with an oracle rotation supplied. Section 2.2.
- Time-varying operators when a supervision signal exists (velocity supervision), whose leading direction agrees with that signal per pseudotime bin. Section 2.5.

**What velocity supervision changes and what it does not.**
Supervision reduces across-seed spread of the operator (halves it in our test [ EL06 ]) and pins the leading-direction rotation to the supervision reference. It does NOT overcome the identifiability of the antisymmetric part when the reference itself lies in the linear span of the input representation: on scNT-seq, `RidgeCV(X_pca) → v_ref` reaches R² 0.92 whereas scJDO Jacobian features reach 0.43 [ EL13 ]. Supervision fixes the direction; it does not confer discriminating power a linear regression on the same input does not already have.

**Reporting checklist for Jacobian-derived gene lists.**
Any leading-direction loading or activation-time gene list should be reported alongside (a) multi-seed statistics (≥ 3 seeds, mean ± spread or 95 % CI), (b) the mask status (whether the sensitive-mask path fired), (c) the bandwidth sweep (a leading-direction loading whose top-15 genes shift by more than 30 % across the pre-declared bandwidth grid is not a claim about the operator), (d) the prior settings (`vel_scale`, `bias_strength`, external `V_ref`), and (e) a companion table showing which entries survive the same reporting under a shuffled-time or block-permutation null.

**Experimental-design implication.**
The identifiability price schedule of Section 2.1 gives an experimental route to any output currently classified as prior-selected. To fix the rotation direction on a snapshot, add either (i) `d − 1` static perturbations of independently controlled inputs, (ii) one time-resolved perturbation, (iii) per-cell velocity from metabolic labelling or splicing kinetics with a reference that is NOT a linear function of the input representation, or (iv) lineage-tracing displacement. Metabolic labelling on the neuron labelling cohort we used is close to a linear function of the input representation; it is not sufficient. A labelling reference whose new-transcript components fall outside the leading 30 principal components would be.

**Limitations of this study.**
The marrow Ery calibration is on a small substrate (1,151 cells) with a specific pipeline (Palantir 1.4.4 + MAGIC on the branch-restricted expression matrix); different pseudotime constructions may shift `K_eff` and the symmetric-part comparison. The LARRY fate-prediction cohort is 154 cells across 106 clones after compute-driven subsampling of the myeloid trajectory; the direction of the negative is robust to this, but a full-cohort rerun would tighten the CIs. Dynamo could not be added as a comparator to Gate 1 because the Klein release lacks spliced/unspliced and metabolic-labelling layers [ EL16 ]. The velocity-matching loss result is on the neuron labelling dataset alone; further datasets are needed to test whether the labelling reference's collinearity with input PCs generalises. [~1180 words]

## 4. Methods

The Methods section is written from `methods_facts.md`, which pins software versions, dataset accessions, per-substrate preprocessing recipes, scJDO model hyperparameters, kernel windowing, the additive pseudotime-gradient prior slot, the velocity-matching loss, the block-permutation null, and the oracle Jacobian. Full details in the supplement; key items follow.

- **scJDO drift model** — FiLM-conditioned neural network `f_θ(x, τ)` with spectral-norm output; denoising score-matching loss plus a control-energy regulariser. Trained with Adam at learning rate 2e−4, cosine schedule, batch size 512.
- **Kernel windowing** — Gaussian kernel over pseudotime with bandwidth selected on the grid (0.01, 0.02, 0.03, 0.05, 0.08, 0.10) by maximising bootstrap reproducibility × peak contrast × peak localisation subject to an effective-sample-size floor of 30.
- **Additive pseudotime-gradient prior** — k-nearest-neighbour pseudotime-gradient velocity added to the model output as `vel_scale · gate(τ) · V_ref`. External `V_ref` accepted via the same slot. Formerly called "velocity prior" in earlier drafts; renamed for clarity.
- **Velocity-matching loss** — `L_match = mean_i (1 − cos(f_θ(x_i, t_i), v_ref_i))` with `λ_match = 1.0`. Direction-only cosine to be scale-invariant.
- **Block-permutation null** — partition pseudotime `[0, 1]` into blocks of width equal to the kernel bandwidth; permute block order, keep within-block cell order intact. 10 draws per outer fold.
- **Oracle Jacobian** — analytic derivative of the toggle-SDE drift for synthetic cells; used only in Section 2.2.

Dataset accessions and pinned versions (Palantir 1.4.4, scanpy 1.12, torch 2.10.0, sklearn 1.8.0, scvi 1.5.0.post1, cellrank 2.0.7, dynamo 1.5.3, anndata 0.12.19) are in `methods_facts.md`.

## Box 1 — What is determined by the snapshot, what is selected by the prior, and what is recoverable

| Determined by the snapshot | Selected by the prior | Recoverable with added information |
|:---|:---|:---|
| Whitened symmetric operator `sym(Σ^{−1/2} J Σ^{1/2})` [ EL03 ] | Leading-direction loadings under default settings [ EL04, EL15 ] | Antisymmetric part `A(x)` — with (`d − 1`) static perturbations |
| Local covariance structure `Σ(τ)` | Ranked gene lists derived from leading eigenvector | Rotation direction — with 1 time-resolved perturbation |
| Estimator fidelity to analytic `J` on synthetic ground truth [ EL06 ] | Archetype activation timing across pseudotime | Per-cell velocity direction — with metabolic labelling (if reference NOT collinear with input rep) |
| Kernel-selected bandwidth `h*` | Oscillatory phase | Lineage displacement — with barcoded clone tracking |
| Sign flip of A_12 on cyclic pseudotime | Saddle timing on cyclic substrates [ EL05, EL14 ] | Sign of A on non-cyclic substrates — with any of the four routes |

## Supplement

- **S1** — Constructed-obstruction tables: parametrisations of the `ψ ∝ ρ` subfamily and the price schedule of Section 2.1 in analytic form.
- **S2** — Prior/seed audit tables: full per-seed per-configuration numbers for the additive prior slot on marrow Ery (bias-strength plumbing empirical trace, `vel_scale × bias_strength` factorial) — the numbers behind Fig 3's audit panel.
- **S3** — Gate preregs and full results: preregistered protocols and reports for every gate under `reproducibility/gates_r26/` referenced in the main text, with commit hashes.
- **S4** — Reprogramming and K562 as illustrations only, with their limitations: r22 reprogramming and r15 K562 Perturb-seq analyses (kept for reference, not part of the main claims; see `reproducibility/regulator_benchmark/`).
- **S5** — Retracted-claims table: mapping each v56-era claim to the evidence that retracts it and the ledger ID that supersedes it.

## References

[1] La Manno G. et al. (2018). RNA velocity of single cells. *Nature* 560, 494–498.
[2] Qiu Q. et al. (2020). scNT-seq: single-cell tagging of nascent RNA. *Cell* 182, 1500–1517.
[3] Palantir: Setty M. et al. (2019). Characterization of cell fate probabilities. *Nature Biotech.* 37, 451–460.
[4] Bergen V. et al. (2020). Generalizing RNA velocity to transient cell states. *Nature Biotech.* 38, 1408–1414.
[5] Qiu X. et al. (2022). Mapping transcriptomic vector fields of single cells (Dynamo). *Cell* 185, 690–711.
[6] Kotliar D. et al. (2019). Identifying gene expression programs of cell-type identity and cellular activity with consensus non-negative matrix factorization. *eLife* 8, e43803.
[7] Weinreb C. et al. (2020). Lineage tracing on transcriptional landscapes links state to fate during differentiation. *Science* 367, eaaw3381.
[8] Balubaid A. et al. (2025). *[preprint on operator methods and geometry]*. bioRxiv (placeholder).
[9] Mojtahedi M. et al. (2016). Cell fate decision as high-dimensional critical state transition. *PLoS Biology* 14, e2000640.
[10] Redd D., Green S., Terooatea T. W. (2026). scJDO software v0.3.x. Source code at github.com/manarai/scJDO.

## Word counts

- Abstract: 174 words (limit 175).
- Introduction: ~590 words (limit 600).
- Results (§2.1–§2.5): ~1,950 words.
- Discussion: ~1,180 words (limit 1,200).
- Methods (main text): ~340 words.
- Total main text (excluding abstract, box, references, supplement): ~4,060 words.
