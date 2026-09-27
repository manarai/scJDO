# PREREG — Task 3: Gate 0d v2 (proper null + criteria)

**Status**: pre-registered before compute. Frozen against `227df70` (Task 2 commit).

## Purpose

Replace the fully-shuffled null of Gate 0d v1 (which was degenerate: kernel averaging makes every window the global mean at random τ) with (a) block permutation of τ (block width = kernel bandwidth) and (b) circular shift of τ. Report Hungarian 1-to-1 stability against each null, held-out gain K_eff, and a correctly whitened precision comparison.

## Substrate (frozen)

- Marrow Ery branch (r24 pinned pipeline, 1,151 cells, FA-30 rep, Palantir pseudotime).
- Prepared via `run_operator_claims.prepare_marrow_ery` (same as Gates 0a/0b/0c/0d v1).

## Fits (frozen)

- **Outer folds**: `KFold(n_splits=3, shuffle=True, random_state=0)` on cells → 3 folds.
- **REAL condition, seeds per fold**: 2 (`{0, 1}`) → 6 real fits.
- **NULLs**, per fold: 10 draws each × 1 seed (seed 0) = 10 fits per null type per fold. Two null types → 20 null fits per fold. Across 3 folds: 60 null fits total.
- **Total fits**: 6 real + 60 null = **66**.
- **scJDO configuration** (unchanged from Gate 0d v1): `vel_scale = 0`, `bias_strength = 1.5`, `hidden = 256`, `depth = 4`, `n_epochs = 3000`, `grid_size = 150`, `n_archetypes = K = 5`, kernel windowing with adaptive bandwidth.

## Null definitions (frozen)

- **Block permutation**: partition the sorted τ range [0, 1] into non-overlapping blocks of width `h = adaptive bandwidth` from the reference fold-0 seed-0 REAL fit. Group each cell's τ into a block. Permute the block ORDER (random permutation of block indices per null draw) while keeping τ order WITHIN each block intact. This preserves the local structure of τ (nearby cells stay nearby in τ) but destroys the alignment between global position and τ.
- **Circular shift**: `τ_shifted = (τ + δ) mod 1`, where `δ ~ Uniform([0, 1))` per null draw. Preserves the RELATIVE ordering of all cells but breaks the anchor between τ=0 and any biological reference (Progenitor cluster at low τ). Because scJDO's Progenitor / Ery groupby depends on τ thresholds, shifted τ generates different Progenitor/Terminal masks per draw; these masks are recomputed identically to REAL (30th percentile = Progenitor, 70th = Terminal).

Both null types have 10 draws each per fold (random state derived from `fold_idx * 100 + draw_idx`).

## Stability metric — Hungarian 1-to-1 signed cosine, medoid reporting (frozen)

- For each fit, sign-normalise the 5 patterns per prior convention (flip so leading-|magnitude| element positive), flatten to 900-D unit vectors.
- **Reference for each fold**: fold's seed-0 REAL fit's 5 patterns.
- For each REAL non-reference fit (fold_i seed_1, i.e. 1 per fold, plus the 5 non-fold-0 REAL fits) match to reference via Hungarian on `1 - signed_cosine`. Signed cosine of matched pairs is the stability signal.
- For each NULL fit (10 per null type per fold): Hungarian-match its 5 patterns to REAL reference, take signed cosine.
- **Stability threshold**: signed cos ≥ **0.85** (frozen; stricter than Gate 0d v1's 0.7).
- **Medoid reporting**: for each fold, the "medoid REAL fit" = the fit whose patterns' summed matched cos to the other REAL fits is largest. Report medoid's per-archetype `min signed cos` across the 5 non-medoid REAL fits.

**Reported per condition per fold**: for each of 5 reference archetypes, fraction of matched fits with signed cos ≥ 0.85, plus the full distribution.

## Held-out gain K_eff (frozen)

- **Dictionary from discovery folds**: for each K ∈ {1, ..., 5}, semi-NMF-decompose the DISCOVERY fold's real J_tensor into K archetypes (`scjdo.archetypes.decompose.jacobian_modes(rank=K, n_restarts=5, seed=0)`). For the discovery step, use fold-0's fold-0 seed-0 REAL fit's J_tensor as the training tensor.
- **NNLS activations on validation fold**: for each of the 2 non-discovery folds' seed-0 REAL J_tensors, solve `activations = scipy.optimize.nnls(V, J_val_flat)` per grid point, where V = discovery patterns stacked as (K, D²) and J_val_flat = per-grid-point-flattened validation J. Compute reconstruction `J_recon = V @ activations`.
- **Reconstruction error**: `err(K) = mean_over_grid(‖J_val − J_recon‖²_F) / mean_over_grid(‖J_val‖²_F)`.
- **Gain**: `gain(K) = err(K − 1) − err(K)`. `err(0)` = 1.0 (no reconstruction).
- **Null gain distribution**: repeat NNLS with dictionary from a random block-permutation null fit at fold-0. Report the 95th percentile of `gain_null(K)` across 10 null draws.
- **K_eff = largest K such that `gain(K) > gain_null(K)_95pct`**. If no K passes, K_eff = 0.

## Precision comparison, whitened (frozen)

For the REAL fold-0 seed-0 fit:
- Compute the kernel-aggregated local covariance `Σ(τ)` from the same substrate, same grid, same bandwidth as scJDO's J_tensor (identical to Gate 0b's covariance tensor).
- Compute `Σ^{−1/2}(τ)` via eigendecomposition + inverse-square-root, `Σ^{1/2}(τ)` similarly.
- **Whitened J**: `W(τ) = Σ^{−1/2}(τ) · J(τ) · Σ^{1/2}(τ)`.
- **Symmetric part**: `sym_W(τ) = 0.5 · (W(τ) + W(τ)^T)`.
- **Count-noise diffusion `D̂`**: constructed from HVG counts (marrow Ery, 1,151 cells × 16,106 genes → 2,000 HVGs → projected to FA-30). `D̂ = diag(gene-wise Poisson variance)` in gene space, projected into FA-30 via loadings: `D̂_FA = L^T · diag(σ²_gene) · L` where `L` is the (n_genes, 30) `fa_loadings` matrix and `σ²_gene = mean(counts_gene, axis=0)` (Poisson approx). This is CONSTANT in τ.
- **Predicted symmetric part under Lyapunov gauge**: `predicted_sym(τ) = −0.5 · Σ^{−1/2}(τ) · D̂_FA · Σ^{−1/2}(τ)`.
- **Local precision**: `precision(τ) = pinv(Σ(τ) + λI)` with `λ = 1e−3 · trace(Σ(τ)) / D` (Gate 0d v1 formula).

### Comparisons

Two `sym_W(τ)`-versus-baseline comparisons, per grid point τ and averaged:
1. `sym_W(τ)` vs `predicted_sym(τ)` — leading-eigvec |cos| (median over τ) + Frobenius correlation (mean over τ).
2. `sym_W(τ)` vs `local_precision(τ)` — same two statistics.

## Frozen readouts

- Fraction of REAL archetypes stable at signed cos ≥ 0.85 per fold + averaged over folds.
- Same for block-null and circular-null (mean fraction across the 10 draws per fold per null type).
- gain(K) profile for K = 1..5; K_eff.
- Precision comparison: median |cos| and mean Frob corr for both baselines.

## DECISION.md update (frozen)

After Task 3 completes, append EXACTLY this line, filling in K_eff and nothing else:

> Gate 0c's consensus was methodologically flawed (absolute cosine, single linkage). Gate 0d with a block-permutation null and held-out gain gives K_eff = [n].

Write NOTHING about archetypes being "supported" unless K_eff ≥ 2 by held-out criterion. If K_eff = 0 or 1, DECISION.md notes only the fact of that K_eff.

## Deliverables

- `PREREG_Task3_Gate0d_v2.md` (this file).
- `run_gate0d_v2.py` — single deterministic entry point.
- `REPORT_Gate0d_v2.md` — verdict + all numbers.
- `gate0d_v2_summary.json` — machine-readable.

## Compute budget

- 66 fits × ~2 min = **~130 min**.
- NNLS + precision comparison + figure: ~5 min.
- **Total: ~2 h 15 min.**

If block-permutation null draws consume more compute than expected, reduce to 5 null draws per null type per fold (30 null fits + 6 REAL = 36 fits). This is the ONLY frozen fallback; no other design changes.
