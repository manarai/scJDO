# API reference

The public API is organized by namespace, following the same convention used by Scanpy. Preprocessing functions are in `sjd.pp`, analysis tools are in `sjd.tl`, and plotting functions are in `sjd.pl`.

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
| `scjdo.pp` | Trajectory preparation and pseudotime preprocessing |
| `scjdo.tl` | Drift, bridge, instability-gene, and regulator inference tools |
| `scjdo.pl` | Plotting helpers for fitted drift, bridge, and regulator results |
| `scjdo.models` | Lower-level neural drift and bridge model classes |
