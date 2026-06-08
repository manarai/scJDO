# Release notes

## Version 0.3.0 — 2026-05-15

Version 0.3.0 introduces the Scanpy-style `sjd.pp`, `sjd.tl`, and `sjd.pl` API. Users can now run complete drift, bridge, plotting, and regulator-inference workflows without manually constructing model objects or training loops.

| Area | New or changed behavior |
|---|---|
| Preprocessing | `sjd.pp.prepare_trajectory` performs normalization, HVG selection, PCA, neighbors, UMAP, and DPT pseudotime preparation. |
| Drift analysis | `sjd.tl.fit_drift` trains the drift field, computes Jacobians, runs semi-NMF archetype decomposition, and stores results in `adata.uns`. |
| Bridge analysis | `sjd.tl.fit_bridge` computes forward and backward bridge dynamics and direction-specific instability gene tables. |
| Regulator inference | `sjd.tl.infer_regulators` scores transcription factors using six metrics and can infer de novo co-instability edges. |
| Plotting | `sjd.pl` includes drift, bridge, and regulator summary figures plus standalone plotting panels. |
| CLI | `scjdo drift` and `scjdo bridge` run complete pipelines from `.h5ad` files. |

### Model and implementation changes

The drift model now uses FiLM conditioning, spectral normalization is available for Jacobian stability, the default velocity gate is `flat`, and archetype decomposition uses semi-NMF with non-negative activation profiles. Jacobian averaging is performed at the per-cell level before temporal aggregation.

## Version 0.2.0 — initial public release

Version 0.2.0 provided the initial public implementation of the core drift, bridge, Fourier-score, and hybrid-GRN components, along with earlier command-line entry points.
