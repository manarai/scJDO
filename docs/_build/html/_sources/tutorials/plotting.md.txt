# Plotting panels

All major summary figures are composed of standalone plotting functions. This design makes it easy to build publication panels without copying internal plotting code.

## Drift-field panels

```python
sjd.pl.drift_field(adata, save='drift_field.pdf')
sjd.pl.sensitivity(adata, save='sensitivity.pdf')
sjd.pl.archetypes(adata, save='archetypes.pdf')
sjd.pl.coordination(adata, save='coordination.pdf')
sjd.pl.instability_genes(adata, n_genes=15, save='instability_genes.pdf')
```

## Bridge panels

```python
sjd.pl.bridge_trajectories(adata, direction='forward', save='traj_fwd.pdf')
sjd.pl.bridge_trajectories(adata, direction='backward', save='traj_bwd.pdf')
sjd.pl.bridge_instability(adata, save='bridge_instability.pdf')
sjd.pl.bridge_archetypes(adata, save='bridge_archetypes.pdf')
sjd.pl.bridge_genes(adata, save='bridge_genes.pdf')
sjd.pl.bridge_gene_comparison(adata, save='bridge_gene_comparison.pdf')
```

## Regulator panels

```python
sjd.pl.regulator_barplot(adata, save='regulator_barplot.pdf')
sjd.pl.regulator_heatmap(adata, save='regulator_heatmap.pdf')
sjd.pl.regulator_scatter(adata, save='regulator_scatter.pdf')
sjd.pl.regulator_profiles(adata, save='regulator_profiles.pdf')
sjd.pl.regulator_network(adata, save='regulator_network.pdf')
```

| Plot type | Best use |
|---|---|
| Summary figures | First-pass review and complete workflow reports |
| Standalone panels | Manuscript figures and supplementary panels |
| Network plots | Interpreting regulator hypotheses and de novo co-instability edges |
| Gene heatmaps | Comparing archetype-specific instability signatures |
