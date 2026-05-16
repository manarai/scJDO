# Plotting panels

All major summary figures are composed of standalone plotting functions. This design makes it easy to build publication panels without copying internal plotting code.

## Drift-field panels

```python
sqd.pl.drift_field(adata, save='drift_field.pdf')
sqd.pl.sensitivity(adata, save='sensitivity.pdf')
sqd.pl.archetypes(adata, save='archetypes.pdf')
sqd.pl.coordination(adata, save='coordination.pdf')
sqd.pl.instability_genes(adata, n_genes=15, save='instability_genes.pdf')
```

## Bridge panels

```python
sqd.pl.bridge_trajectories(adata, direction='forward', save='traj_fwd.pdf')
sqd.pl.bridge_trajectories(adata, direction='backward', save='traj_bwd.pdf')
sqd.pl.bridge_instability(adata, save='bridge_instability.pdf')
sqd.pl.bridge_archetypes(adata, save='bridge_archetypes.pdf')
sqd.pl.bridge_genes(adata, save='bridge_genes.pdf')
sqd.pl.bridge_gene_comparison(adata, save='bridge_gene_comparison.pdf')
```

## Regulator panels

```python
sqd.pl.regulator_barplot(adata, save='regulator_barplot.pdf')
sqd.pl.regulator_heatmap(adata, save='regulator_heatmap.pdf')
sqd.pl.regulator_scatter(adata, save='regulator_scatter.pdf')
sqd.pl.regulator_profiles(adata, save='regulator_profiles.pdf')
sqd.pl.regulator_network(adata, save='regulator_network.pdf')
```

| Plot type | Best use |
|---|---|
| Summary figures | First-pass review and complete workflow reports |
| Standalone panels | Manuscript figures and supplementary panels |
| Network plots | Interpreting regulator hypotheses and de novo co-instability edges |
| Gene heatmaps | Comparing archetype-specific instability signatures |
