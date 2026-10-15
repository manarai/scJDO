# v1 baseline — snapshot at `main @ a0b8ef4`

Captured on 2026-10-15 at the start of the v2 remodel. This directory is the
reference v2 is compared against in Phase E.

## pytest baseline

See [pytest_report.txt](pytest_report.txt).

- **152 passed, 11 failed, 2 skipped, 2 errors** in 17.56 s.
- The 11 failures and 2 errors are **pre-existing** in untracked / WIP test
  files (`test_prior_toggles.py`, `test_constructed_obstruction.py`,
  `test_recovery_path.py`, parts of `test_hybrid_grn.py`,
  `test_training_smoke.py`, `test_corrections.py`); they reference
  modules/features that have not landed on `main` (`scjdo.validation.constructed_obstruction`,
  gradient-only drift parameterisation, etc.).
- They are **not** caused by v2 — recorded here only to establish the baseline.

## Notebook outputs

The five canonical v1 tutorials (`01, 02, 03, 04, 05` per
[`examples/README.md`](../../examples/README.md)) have **existing rendered
outputs** at these paths from prior runs; v2's Phase E compares against them
rather than re-running the heavy fits (notebook 01 trains 5000 epochs; 04
trains a Schrödinger Bridge; 08 is an 8-method benchmark):

| Notebook | Outputs |
|---|---|
| 01 Paul15 hybrid drift | [`examples/results/01_paul15/`](../../examples/results/01_paul15/) |
| 02 Paul15 Fourier | [`examples/results/02_fourier/`](../../examples/results/02_fourier/) |
| 03 Schrödinger bridge (2D synthetic) | [`examples/results/03_bridge/`](../../examples/results/03_bridge/) and [`results/03_bridge/`](../../results/03_bridge/) |
| 04 Paul15 bridge | [`examples/results/04_paul15_bridge/`](../../examples/results/04_paul15_bridge/) and [`results/04_paul15_bridge/`](../../results/04_paul15_bridge/) |
| 05 scOpAtlas | depends on 01's checkpoint; outputs live under [`examples/results/`](../../examples/results/) |

Also usable as v1 reference:

- [`examples/results/benchmark/`](../../examples/results/benchmark/) — embedding benchmark (notebook 08).
- [`examples/results/fa_benchmark/`](../../examples/results/fa_benchmark/) — FA vs. PCA vs. scVI (notebook 09).
- [`examples/results/10_koopman/`](../../examples/results/10_koopman/) — Koopman tutorial.
- [`Figures_notebook/`](../../Figures_notebook/) — manuscript-figure notebooks (not tutorials but authoritative for v1).

## What Phase E compares against

Phase E's `compare/v1_vs_v2.py` reads from the paths above for v1 numbers and
from `benchmarks/v2/<dataset>/` for v2 numbers. For datasets where v1 outputs
pre-exist (all five tutorials), the compare pipeline reuses them. Only
datasets without pre-stored v1 outputs would trigger an on-demand v1 rerun;
none of the canonical five falls in that bucket.
