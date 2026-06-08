# scJDO — single-cell Jacobian drift operators in Python

<div class="scjdo-hero">
<p class="scjdo-tagline"><strong>scJDO</strong> infers time-varying dynamical operators from single-cell transcriptomic data. It learns a drift field over pseudotime, computes local Jacobian operators, decomposes them into recurrent regulatory archetypes, and links instability-driving genes to upstream transcription factors.</p>
</div>

```python
import scanpy as sc
import scjdo as sjd

adata = sc.datasets.paul15()
sjd.pp.prepare_trajectory(adata, groupby='paul15_clusters', root='7MEP')
sjd.tl.fit_drift(adata, n_archetypes=5, n_epochs=5000)
sjd.pl.summary_figure(adata, save='figure3.pdf')
```

scJDO follows the familiar Scanpy-style namespace convention: preprocessing lives in `sjd.pp`, analysis methods live in `sjd.tl`, and plotting functions live in `sjd.pl`. This documentation site is structured for a ReadTheDocs deployment similar to the stable Scanpy documentation, with a landing page, tutorial guide, usage principles, and an API reference grouped by namespace.[^scanpy]

| Workflow | Primary call | Typical question |
|---|---|---|
| Drift-field analysis | `sjd.tl.fit_drift` | Which local regulatory operators and instability genes drive a pseudotime trajectory? |
| Schrödinger Bridge analysis | `sjd.tl.fit_bridge` | How do forward and backward endpoint-constrained dynamics differ between source and target populations? |
| Regulatory inference | `sjd.tl.infer_regulators` | Which transcription factors explain instability-driving genes and archetype-specific programs? |

## Public API

The public API is intentionally compact. Most users can complete a full analysis with four calls: `prepare_trajectory`, one model-fitting function, one gene/regulator extraction function, and one summary plotting function.

| Namespace | Purpose | Examples |
|---|---|---|
| `sjd.pp` | Single-cell preprocessing and pseudotime preparation | `prepare_trajectory` |
| `sjd.tl` | Drift, bridge, instability-gene, and regulator analysis | `fit_drift`, `fit_bridge`, `infer_regulators` |
| `sjd.pl` | Reusable plotting panels and summary figures | `summary_figure`, `bridge_summary`, `regulator_network` |

```{toctree}
:maxdepth: 2
:caption: Getting started

installation
tutorials/index
usage-principles
how-to/index
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/index
mathematical-background
release-notes
citation
contributing
```

## References

[^scanpy]: Scanpy documentation, "Scanpy – Single-Cell Analysis in Python," https://scanpy.readthedocs.io/en/stable/.
