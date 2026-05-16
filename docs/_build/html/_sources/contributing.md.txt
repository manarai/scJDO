# Contributing

Contributions should preserve the compact Scanpy-style public API while improving model reliability, documentation clarity, and reproducibility.

## Documentation development

Build the documentation locally before opening a pull request.

```bash
python -m pip install -r docs/requirements.txt
python -m pip install -e . --no-deps
sphinx-build -b html docs docs/_build/html
```

## Documentation style

Documentation pages should use complete examples, clear tables, and explicit result-storage descriptions. API pages should document where each function reads from and writes to `AnnData`.

| Contribution type | Expected evidence |
|---|---|
| New public function | API page, tutorial snippet, and tests |
| New plot | Example output and description of required `adata.uns` fields |
| New model option | Parameter documentation and reproducibility guidance |
| Bug fix | Regression test and release-note entry |
