# Tutorials

These tutorials mirror the project workflows used in the repository examples. They are written as copy-pasteable guides rather than notebook outputs, so they can be used as the basis for executable notebooks, ReadTheDocs tutorial pages, or lab protocols.

| Tutorial | Dataset | Main outputs |
|---|---|---|
| [Drift field analysis](drift.md) | Scanpy Paul15 hematopoiesis | Drift vectors, Jacobian tensor, archetypes, instability genes, regulator figures |
| [Schrödinger Bridge analysis](bridge.md) | Paul15 or any source/target populations | Forward and backward trajectories, instability curves, gene comparisons |
| [Plotting panels](plotting.md) | Any fitted scJDO object | Individual figures for manuscripts and supplementary panels |

```{toctree}
:maxdepth: 1

Drift field analysis <drift>
Schrödinger Bridge analysis <bridge>
Plotting panels <plotting>
```
