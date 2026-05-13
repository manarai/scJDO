# Examples

This directory contains four Jupyter tutorial notebooks covering the main scQDiff workflows.

| Notebook | Description |
|----------|-------------|
| `01_paul15_hybrid_drift_tutorial.ipynb` | Hybrid drift field on the Paul15 hematopoiesis dataset |
| `02_paul15_fourier_tutorial.ipynb` | Fourier-domain score network extension |
| `03_schrodinger_bridge_tutorial.ipynb` | Schrödinger Bridge for aging / optimal transport |
| `04_scopatlas_complete_workflow.ipynb` | Full SCOPAtlas stable-operator pipeline |

## Generated outputs

Large files produced by running the notebooks (model checkpoints `*.pt`, processed
datasets `*.h5ad`, raw data `*.h5`) are excluded from version control via `.gitignore`.
Run the notebooks in order to regenerate them:

```bash
conda activate scQDiff
jupyter lab
```

The Paul15 source data (`data/paul15/paul15.h5`) is fetched automatically by the
first notebook via `scanpy.datasets.paul15()` — no manual download required.
