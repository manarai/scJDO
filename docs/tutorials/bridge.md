# Schrödinger Bridge analysis

Schrödinger Bridge analysis fits endpoint-constrained stochastic dynamics between source and target populations. In scJDO, the bridge workflow computes both forward and backward Jacobian tensors, enabling asymmetric instability analysis across a transition.

## Quantile-based source and target populations

```python
import scanpy as sc
import scjdo as sjd

adata = sc.datasets.paul15()
sjd.pp.prepare_trajectory(adata, groupby='paul15_clusters', root='7MEP')

sjd.tl.fit_bridge(
    adata,
    src_quantile=0.20,
    tgt_quantile=0.80,
    n_archetypes=4,
)

sjd.pl.bridge_summary(adata, save='results/bridge_summary.pdf')

df_fwd, df_bwd = sjd.tl.get_bridge_instability_genes(adata)
df_fwd.to_csv('results/genes_forward.csv', index=False)
df_bwd.to_csv('results/genes_backward.csv', index=False)
```

## Label-based source and target populations

When the biological comparison is defined by a condition or cell-type label, use `src_group`, `tgt_group`, and `groupby` instead of pseudotime quantiles.

```python
sjd.tl.fit_bridge(
    adata,
    groupby='condition',
    src_group='young',
    tgt_group='old',
    n_archetypes=4,
)
```

## Direction-specific regulator inference

Bridge results can be passed to `infer_regulators` by setting `key='scjdo_bridge'`. Direction can be `forward`, `backward`, or `both`.

```python
sjd.tl.infer_regulators(
    adata,
    key='scjdo_bridge',
    direction='forward',
    key_added='scjdo_regulators_fwd',
)

sjd.pl.regulator_network(
    adata,
    key='scjdo_regulators_fwd',
    scjdo_key='scjdo_bridge',
    save='results/regulator_network_forward.pdf',
)
```

| Bridge result | Meaning |
|---|---|
| `max_eig_fwd`, `max_eig_bwd` | Forward and backward maximum real eigenvalue trajectories |
| `pat_fwd`, `pat_bwd` | Direction-specific operator archetype patterns |
| `act_fwd`, `act_bwd` | Direction-specific archetype activations |
| `df_fwd`, `df_bwd` | Direction-specific instability gene tables |
| `fwd_traj_2d`, `bwd_traj_2d` | Simulated 2D trajectories for visualization |
