# Installation

scQDiff is a Python package for single-cell dynamical analysis. The package requires Python 3.9 or later and depends on the scientific Python and single-cell ecosystem, including PyTorch, NumPy, SciPy, AnnData, Scanpy, and Matplotlib.

## Install from GitHub

The current development version can be installed directly from the GitHub repository.

```bash
git clone https://github.com/manarai/scQDiff.git
cd scQDiff
python -m pip install -e .
```

For development and documentation work, install the additional development requirements.

```bash
python -m pip install -r requirements-dev.txt
```

## Optional documentation environment

The documentation source in `docs/` is designed for ReadTheDocs and local Sphinx builds. A minimal local documentation environment can be prepared with the following commands.

```bash
python -m pip install -r docs/requirements.txt
python -m pip install -e . --no-deps
sphinx-build -b html docs docs/_build/html
```

The `--no-deps` flag is useful for documentation-only builds because Sphinx can mock heavy runtime dependencies while still importing the scQDiff package structure.

| Component | Recommendation |
|---|---|
| Python | 3.9 or newer |
| Hardware | GPU recommended for publication-scale drift or bridge training |
| Core data structure | `anndata.AnnData` |
| Preferred preprocessing ecosystem | Scanpy |
| Documentation theme | `pydata-sphinx-theme` |

## Verify the installation

After installation, verify that the Scanpy-style namespaces are available.

```python
import scqdiff as sqd

print(sqd.__version__)
print(sqd.pp.prepare_trajectory)
print(sqd.tl.fit_drift)
print(sqd.pl.summary_figure)
```
