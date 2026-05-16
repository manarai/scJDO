# API reference

The public API is organized by namespace, following the same convention used by Scanpy. Preprocessing functions are in `sqd.pp`, analysis tools are in `sqd.tl`, and plotting functions are in `sqd.pl`.

```{toctree}
:maxdepth: 2

Preprocessing: pp <preprocessing>
Tools: tl <tools>
Plotting: pl <plotting>
Core models <models>
Command line interface <cli>
```

| Namespace | Role |
|---|---|
| `scqdiff.pp` | Trajectory preparation and pseudotime preprocessing |
| `scqdiff.tl` | Drift, bridge, instability-gene, and regulator inference tools |
| `scqdiff.pl` | Plotting helpers for fitted drift, bridge, and regulator results |
| `scqdiff.models` | Lower-level neural drift and bridge model classes |
