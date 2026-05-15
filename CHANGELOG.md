# Changelog

## [0.3.0] — 2026-05-15

### Added — high-level API

Scanpy-style `sqd.pp / sqd.tl / sqd.pl` namespaces. Users no longer need to
construct model objects, training loops, or plotting code manually.

**`sqd.pp`**
- `prepare_trajectory()` — normalize → HVG → PCA → kNN → DPT in one call.
  Auto-handles NaN pseudotime, normalizes output to [0, 1].

**`sqd.tl`**
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

**`sqd.pl`**
- `summary_figure()` — four-panel drift field summary.
- `drift_field()` — streamplot (default) or quiver on PCA/UMAP. Uses
  pseudotime-gradient velocity for PCA plots to avoid noisy 50D→2D projection.
- `sensitivity()`, `archetypes()`, `coordination()`, `instability_genes()`.
- `bridge_summary()`, `bridge_trajectories()`, `bridge_instability()`,
  `bridge_archetypes()`, `bridge_genes()`, `bridge_gene_comparison()`.
- `regulator_summary()`, `regulator_barplot()`, `regulator_heatmap()`,
  `regulator_scatter()`, `regulator_profiles()`, `regulator_network()`.

**CLI**
- `scqdiff drift` and `scqdiff bridge` — full pipeline in one command.
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
CLI entry points: `scqdiff-train`, `scqdiff-bridge`, `scqdiff-atlas`.
