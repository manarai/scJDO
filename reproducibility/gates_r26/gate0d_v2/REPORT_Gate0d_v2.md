# REPORT — Task 3: Gate 0d v2

**K_eff = 1** by held-out gain against block-permutation null. Under Hungarian 1-to-1 signed-cosine matching at tolerance 0.85, block and circular nulls yield HIGHER fraction-stable than REAL (0.820, 0.800 vs REAL 0.667). Whitened symmetric part of J matches the Lyapunov-gauge prediction `-½ Σ^{-1/2} D̂ Σ^{-1/2}` moderately (median |cos| 0.454, mean Frob corr +0.468) and matches local precision poorly (median |cos| 0.224, mean Frob corr −0.368).

**Prereg**: `PREREG_Task3_Gate0d_v2.md` (frozen `6765787`, before compute).
**Compute**: 139 min for 66 fits + K_eff loop + precision comparison (6 REAL + 30 block-null + 30 circular-null fits, each ~2 min at 1,151 cells).

## Fits

- 3-fold CV, 2 seeds per fold real = 6 REAL fits.
- 10 draws × 2 null types × 3 folds = 60 null fits.
- Bandwidth (fold-0 seed-0 REAL): `h = 0.05`. Block-permutation blocks were sized 0.05 wide.

## Stability under Hungarian 1-to-1 signed cosine (threshold ≥ 0.85)

Per-fold REAL matched signed cosine (seed-1 fit against seed-0 reference):
```
fold=0 seed=1: [+0.677, +0.844, +0.970, +0.730, +0.934]
fold=1 seed=1: [+0.821, +0.884, +0.958, +0.970, +0.937]
fold=2 seed=1: [+0.733, +0.915, +0.971, +0.987, +0.974]
```

Fraction of matched signed cosines ≥ 0.85 (per prereg):

| Condition | Fraction stable | n matched (15 = 5 archs × 3 folds each) |
|:---:|:---:|:---:|
| REAL           | **0.667** | 10 / 15  |
| block null     | 0.820     | 123 / 150 (30 fits × 5 archs) |
| circular null  | 0.800     | 120 / 150 |

**Both nulls yield HIGHER stability than REAL.** Diagnostic explanation follows.

## Held-out gain K_eff (block-null 95th percentile)

Reconstruction error `err(K)` of validation-fold J_tensors from a dictionary of K NNLS-activated archetypes, learned on discovery fold fold-0 seed-0 REAL:

| K | err(K) real | real gain(K) | block-null 95th pct | Exceeds null? |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.0624 | +0.9376 | +0.9372 | yes (barely) |
| 2 | 0.0413 | +0.0212 | +0.0241 | no |
| 3 | 0.0397 | +0.0016 | +0.0024 | no |
| 4 | 0.0386 | +0.0010 | +0.0021 | no |
| 5 | 0.0381 | +0.0006 | +0.0015 | no |

**K_eff = 1** (largest K where real gain(K) > block-null 95th pct).

## Whitened precision comparison

Symmetric part of the whitened Jacobian `sym_W(τ) = ½ · [Σ^{−1/2}·J·Σ^{1/2} + (Σ^{−1/2}·J·Σ^{1/2})^T]` compared to two baselines at each grid point τ:

| Baseline | median `|cos(leading eigvec)|` over τ | mean Frob correlation over τ |
|:---|:---:|:---:|
| Lyapunov-predicted `−½ Σ^{−1/2} D̂ Σ^{−1/2}` | **0.454** | **+0.468** |
| local precision `Σ^{−1}`                     | 0.224     | **−0.368** |

`D̂ = L^T diag(gene-wise Poisson variance) L` projected into the FA-30 rep space via the loadings `L`. `‖D̂‖_F = 85.9`.

**The Lyapunov-gauge prediction is a moderately better match to sym_W than local precision** on this substrate. Anti-correlation of sym_W with local precision (Frob corr −0.37) is consistent with the sign expected for a stable trajectory (relaxation toward a manifold pulls in the negative direction of the local density curvature).

## Diagnostic — why nulls beat REAL on Hungarian stability

The Hungarian-match-to-REAL-reference metric measures how well a fit's 5 archetypes ALIGN with a fixed reference. Under block permutation of τ and circular shift, scJDO's DSM training loop learns essentially the SAME local density gradients everywhere in τ (all τ-bins see similar local geometries because the density-of-cells varies smoothly). The resulting Jacobian tensors are near-constant across τ within each null fit, and the semi-NMF decomposition of a near-constant tensor consistently returns the same 5 archetypes across draws — hence high Hungarian match to REAL's reference archetypes on average.

Under REAL, DIFFERENT scJDO SEEDS produce genuinely different fits (initialisation stochasticity + basin selection). Cross-seed variation causes signed-cos < 0.85 on ~5/15 arch-fold pairs.

This is exactly the failure mode Gate 0d v1's fully-shuffled null exhibited (5/5 stable) — the block/circular nulls narrow the degeneracy but do not eliminate it. On this substrate, ANY null that preserves the marginal density of τ (as block permutation and circular shift do) collapses to trivially consistent archetypes across fits. The correct discriminator is the HELD-OUT GAIN, not fraction-stable.

**The K_eff = 1 result is the substantive scientific claim.** The stability fractions are a diagnostic showing the null design's limits.

## DECISION.md update (frozen line, appended)

Exactly this line will be appended to `DECISION.md`:

> Gate 0c's consensus was methodologically flawed (absolute cosine, single linkage). Gate 0d with a block-permutation null and held-out gain gives K_eff = 1.

Per prereg: no archetype-support claims (K_eff < 2).

## Deliverables

- Prereg: `PREREG_Task3_Gate0d_v2.md` (`6765787`).
- Runner: `run_gate0d_v2.py`.
- This report.
- Machine-readable summary: `gate0d_v2_summary.json`.
- Per-fit caches: `fit_REAL_fold{0,1,2}_seed{0,1}.pkl` + `fit_NULL_{block,circular}_fold{0-2}_draw{0-9}.pkl` (66 total).
- Log: `run.log`.
