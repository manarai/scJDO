# Plotting: `pl`

The plotting namespace provides standalone Matplotlib panels and combined summary figures. Most functions accept `ax=None` for standalone plotting and `save=None` for optional file output.

## Drift-field plots

```{eval-rst}
.. autofunction:: scqdiff.pl.summary_figure

.. autofunction:: scqdiff.pl.drift_field

.. autofunction:: scqdiff.pl.sensitivity

.. autofunction:: scqdiff.pl.archetypes

.. autofunction:: scqdiff.pl.coordination

.. autofunction:: scqdiff.pl.instability_genes
```

## Bridge plots

```{eval-rst}
.. autofunction:: scqdiff.pl.bridge_source_target

.. autofunction:: scqdiff.pl.bridge_trajectories

.. autofunction:: scqdiff.pl.bridge_instability

.. autofunction:: scqdiff.pl.bridge_archetypes

.. autofunction:: scqdiff.pl.bridge_genes

.. autofunction:: scqdiff.pl.bridge_summary

.. autofunction:: scqdiff.pl.bridge_gene_comparison
```

## Regulator plots

```{eval-rst}
.. autofunction:: scqdiff.pl.regulator_barplot

.. autofunction:: scqdiff.pl.regulator_heatmap

.. autofunction:: scqdiff.pl.regulator_scatter

.. autofunction:: scqdiff.pl.regulator_profiles

.. autofunction:: scqdiff.pl.regulator_network

.. autofunction:: scqdiff.pl.regulator_summary
```
