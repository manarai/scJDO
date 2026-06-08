# Tools: `tl`

The tools namespace contains model-fitting, gene extraction, regulator inference, and network-loading functions.

## Drift analysis

```{eval-rst}
.. autofunction:: scjdo.tl.fit_drift

.. autofunction:: scjdo.tl.fit_drift_branches

.. autofunction:: scjdo.tl.branch_drift_analysis

.. autofunction:: scjdo.tl.get_instability_genes
```

## Schrödinger Bridge analysis

```{eval-rst}
.. autofunction:: scjdo.tl.fit_bridge

.. autofunction:: scjdo.tl.fit_bridge_branches

.. autofunction:: scjdo.tl.get_bridge_instability_genes
```

## Regulatory inference

```{eval-rst}
.. autofunction:: scjdo.tl.infer_regulators

.. autofunction:: scjdo.tl.infer_regulators_branches

.. autofunction:: scjdo.tl.load_network
```

## Stored drift results

| Key | Shape | Description |
|---|---|---|
| `J_tensor` | `(T, D, D)` | Temporal Jacobian operator (T = `grid_size` in kernel mode, `n_windows` in fixed mode) |
| `t_centers` | `(T,)` | Grid pseudotime centers |
| `patterns` | `(K, D, D)` | Archetype Jacobian patterns |
| `activations` | `(T, K)` | Non-negative temporal activations |
| `max_real_eig` | `(T,)` | Maximum real eigenvalue per grid point |
| `instability_scores` | `(T, n_genes)` | Gene-level instability scores |
| `top_instability_genes` | `list` | Globally ranked instability genes |
| `windowing` | `str` | `'kernel'` (default) or `'fixed'` |
| `bandwidth` | `float` or `(T,)` | Selected $h^*$ (`None` in fixed mode) |
| `n_eff` | `(T,)` | Effective sample size per grid point (kernel only) |
| `kernel_score` | `dict` | `{R, C, L, S}` for the selected bandwidth |
| `kernel_sweep` | `list[dict]` | Per-bandwidth scores (kernel + `bandwidth='auto'` only) |
| `lam_bootstrap` | `(n_boot, T)` | Bootstrap $\lambda_{\max}$ curves (kernel only) |

## Regulator scoring metrics

| Metric | Output column | Interpretation |
|---|---|---|
| Weighted out-degree | `weighted_score` | Total instability explained by TF targets |
| Mean target instability | `mean_instability` | Quality of target instability signal |
| Regulon enrichment | `enrichment_score` | Hypergeometric enrichment score |
| Branch specificity | `branch_specificity` | Archetype preference on a 0–1 scale |
| Database confidence | `db_confidence` | Mean confidence of source-network edges |
| Pseudotime lead | `pseudotime_lead` | Optional TF-before-target timing score |
