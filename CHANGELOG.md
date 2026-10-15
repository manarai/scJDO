# Changelog

## [Unreleased] — v2 preview

### Phase B (2026-10-15) — minimum field version (Steps 1, 2, 5, 7, 8)

Operator field layer on top of v1's drift. v1 is unchanged; v2 lives in
`scjdo.v2`.

- `scjdo/v2/embed.py` — Step 1 positional layer with `scvi | fa | pca`
  backends and a frozen `transform` for projecting new data.
- `scjdo/v2/drift.py` — Step 2 wrapper on v1's `DriftField` with
  `clock_calibrated=False` by default; no velocity prior.
- `scjdo/v2/field.py` — Step 5 neighbourhood Jacobian field: per-cell
  Gaussian-weighted OLS over kNN in latent space, with a Jacobi-smoothed
  tie between adjacent cells' Jacobians. `k` selectable, `k='auto'`
  sweeps `{15, 25, 50, 100}` on held-out Gaussian log-likelihood. Cells
  below the recovery radius are left `NaN` — no silent fallbacks.
- `scjdo/v2/opspace.py` — Step 7 descriptors: sorted eigenvalues, mode
  counts, spectral gap, Henrici index, reactivity, transient gain,
  eigenvector condition number.
- `scjdo/v2/archetypes.py` — Step 8 semi-NMF wrapper on the stacked
  per-cell Jacobians; `K='auto'` sweeps held-out reconstruction;
  archetype naming (`holder / damped / drift / fork / growing`) with a
  non-normal `sensitive` flag.
- `scjdo/v2/_bench/pitchfork.py` — ported pitchfork synthetic from
  `reproducibility/code/toggle_truth.py` (default τ_crit ≈ 0.00437).

**Acceptance (tests/v2/):**

- `test_pitchfork_default_tau_crit_recovered[42,43]` — the field recovers
  a positive leading eigenvalue localised within ±0.05 of τ_crit on both
  seeds. v1's slicing cannot (its default grid starts at τ=0.02).
- `test_pitchfork_halfsplit_spearman` — half-split binned λ_max curves
  agree at Spearman ρ ≥ 0.8.
- `test_pitchfork_outlier_marked_unrecovered` — far-outlier cells have
  their Jacobians left `NaN` and `recovered=False`.
- `test_pitchfork_el08_variant_diagnostic` — non-gating diagnostic
  across α remappings (τ_crit ∈ {0.1, 0.3}); results logged to
  `benchmarks/v2/pitchfork_el08.csv`. τ_target = 0.5 was removed from
  the diagnostic — low post-bifurcation signal makes the minimum
  crossing detector unreliable there; Phase C will revisit with Fisher
  weighting.
- Archetype + descriptor smoke tests on the pitchfork field.

### Phase A (2026-10-15) — audit + plan

- New `V2_PLAN.md` at repo root — module-mapping table, phase plan,
  acceptance checks, and three deviations from the pasted spec surfaced for
  the maintainer (τ_crit grid for the pitchfork benchmark, embedding default,
  package layout).
- `tests/v2/` scaffold with its own `conftest.py` carrying the pitchfork
  benchmark fixtures (EL08 τ_crit grid, ±0.05 tolerance).
- `benchmarks/v1_baseline/pytest_report.txt` — snapshot of the v1 test suite
  (152 passed, 11 failed, 2 skipped, 2 errors; failures/errors are
  pre-existing in untracked WIP tests, not caused by v2).
- `benchmarks/v1_baseline/README.md` — pointers to the pre-rendered v1
  notebook outputs under `examples/results/`, used as the Phase E comparison
  baseline so the five full tutorials do not need to be re-fit from scratch.
- Repaired the editable install (`pip install -e .`) — the active env's
  `__editable__.scjdo-0.3.0.pth` previously pointed at a defunct
  `/Users/terooatt/Downloads/scJDO`.

## [Unreleased] — 2026-09-27 (calibration-paper wrap-up)

### Fixed (from r25 review)

- **`infer_regulators` empty-instability guard** (`scjdo/tl/_regulators.py`) — added the `instab_matrix.ndim == 2 and instab_matrix.shape[1] > 0` check at line 356 and the bounds check on `gi_list` before indexing. Prevents the `IndexError` when the sensitive mask leaves the instability matrix with zero columns (observed on marrow Ery and DC datasets when using the r25 masked path with narrow gene sets).
- **`infer_regulators` empty-result return** (`scjdo/tl/_regulators.py`) — replaced the `ValueError("No regulator results")` raise at line 867 with an empty-DataFrame return + `UserWarning`. Downstream code no longer needs a try/except around the call when no path produces results.
- **`sjd.tl.rank_fusion` public export** (`scjdo/tl/__init__.py`, r20) — now importable as `sjd.tl.rank_fusion` for RRF-based ensemble scoring; matches the docs.

### Documented — Known limitations (v0.3.x)

Written into `README.md` (Known limitations section). Summary:
- Leading-direction loadings are seed-dependent at default settings; multi-seed reporting is mandatory (Gate 0a; EVIDENCE_LEDGER EL04, EL15).
- The sensitive mask does not fire on the manuscript benchmark datasets; all shown gene lists come from the unmasked fallback path.
- The additive pseudotime-gradient prior slot (previously called "velocity prior") is wired to the model; `vel_scale × bias_strength` outcomes are training basins of the SGD fit, not dynamical-system attractors (Gate 0a; EL04, EL20, EL21).
- Fig 3 reproduction requires Palantir 1.4.4 + MAGIC imputation on the branch-restricted expression matrix.

### Changed — terminology (paper-and-README-only, no API change)

- "Two attractors" → "training basins" throughout narrative documentation.
- "Sensitivity genes" → "leading-direction loadings" throughout narrative documentation.
- "Velocity prior" → "additive pseudotime-gradient prior" throughout narrative documentation.

The Python API symbols (`vel_scale`, `bias_strength`, `V_ref`, `sensitive_mask`) are unchanged for backward compatibility.

## [0.3.0] — 2026-05-15

### Added — high-level API

Scanpy-style `sjd.pp / sjd.tl / sjd.pl` namespaces. Users no longer need to
construct model objects, training loops, or plotting code manually.

**`sjd.pp`**
- `prepare_trajectory()` — normalize → HVG → PCA → kNN → DPT in one call.
  Auto-handles NaN pseudotime, normalizes output to [0, 1].

**`sjd.tl`**
- `fit_drift()` — trains DriftField, builds Jacobian tensor, runs semi-NMF archetype
  decomposition, computes instability gene scores, stores everything in `adata.uns`.
  Automatically derives a pseudotime-gradient velocity prior when no RNA velocity
  is available — eliminates the directional ambiguity of DSM-only training.
- `fit_bridge()` — trains Schrödinger Bridge between source/target populations,
  computes Jacobian tensors for both forward and backward directions, runs semi-NMF
  per direction, extracts instability genes.
- `get_instability_genes()` — extracts ranked instability gene table from drift results.
- `get_bridge_instability_genes()` — same for bridge, returns (df_fwd, df_bwd).
- `infer_regulators()` — links instability genes to upstream TF regulators.
  Scores each TF on six metrics: weighted out-degree, mean target instability,
  regulon enrichment (hypergeometric), branch specificity (entropy), database
  confidence, pseudotime lead-lag. Infers de novo co-instability edges from
  Jacobian eigenvector co-loading. Works with both drift and bridge results.
- `load_network()` — loads CollecTRI (decoupler) → TRRUST v2 → built-in
  hematopoiesis network.

**`sjd.pl`**
- `summary_figure()` — four-panel drift field summary.
- `drift_field()` — streamplot (default) or quiver on PCA/UMAP. Uses
  pseudotime-gradient velocity for PCA plots to avoid noisy 50D→2D projection.
- `sensitivity()`, `archetypes()`, `coordination()`, `instability_genes()`.
- `bridge_summary()`, `bridge_trajectories()`, `bridge_instability()`,
  `bridge_archetypes()`, `bridge_genes()`, `bridge_gene_comparison()`.
- `regulator_summary()`, `regulator_barplot()`, `regulator_heatmap()`,
  `regulator_scatter()`, `regulator_profiles()`, `regulator_network()`.

**CLI**
- `scjdo drift` and `scjdo bridge` — full pipeline in one command.
  Produces annotated `adata.h5ad`, ranked CSV tables, and all figures.

---

### Changed — model architecture

**`DriftField` (drift.py)**
- Replaced `MLPScore` (sinusoidal time concatenation) with `FiLMNet` (FiLM
  conditioning at every hidden layer). Time now modulates hidden representations
  rather than being appended to the input — stronger inductive bias.
- Added `use_spectral_norm=True` — spectral normalization on output layer for
  Jacobian stability.
- Extended `vel_time_mode` gate options: `flat` (new default), `root`, `rise`, `mid`.
  The old default `mid` zeroed the prior at the root where directionality matters
  most; `flat` applies a constant gate throughout.
- Changed `vel_scale` default from `1.0` → `2.0` and `vel_k` from `32` → `15`.

**Archetype decomposition (decompose.py)**
- Replaced SVD with **semi-NMF** (non-negative activations, signed patterns).
  Activations are now guaranteed ≥ 0, making temporal profiles directly
  interpretable as "how active is this archetype." Old SVD kept as
  `jacobian_modes_svd()` for backward compatibility.

**Losses (losses.py)**
- `denoising_score_matching()` now accepts per-sample sigma tensors for
  local adaptive noise (denser regions get finer-grained score).
- Added `local_sigma()` utility — per-cell sigma from kNN density.

**Jacobian computation**
- Fixed averaging: now computes Jacobians at individual cells within each window
  and averages the operators, rather than averaging cells first and computing one
  Jacobian. `J(mean(x)) ≠ mean(J(x))` for nonlinear f.
- Window construction enforces `min_cells=5` per window to avoid degenerate
  representative states.

---

### Fixed

- `direction[:3]` slicing bug in `_regulators.py` that mapped `"forward"` → `"for"`
  (missing key in bridge result dict).
- Numpy array ambiguity in `pl/_regulators.py` — replaced `a or b` with
  `_first_not_none(d, "a", "b")` helper for dict lookups returning arrays.
- Empty DataFrame crash in `_score_regulators()` when no TF met the `min_targets`
  threshold — now returns an empty DataFrame with a descriptive warning.
- Bridge gene set expansion — `infer_regulators` now projects eigenvectors through
  PCA loadings to recover all 2000 HVGs rather than only the top-20 stored in the
  bridge table.
- `obsm` shape error — removed invalid `adata.obsm["X_bridge_src"]` assignment
  (shape was (n_traj, D) not (n_obs, D)).
- Notebook 03 import error: `compute_ot_plan` was imported from the wrong module
  (`coupling` → `sinkhorn`).

---

## [0.2.0] — initial public release

Core models: `DriftField`, `SchrodingerBridge`, `FourierScoreNet`, `HybridGRN`.
CLI entry points: `scjdo-train`, `scjdo-bridge`, `scjdo-atlas`.
