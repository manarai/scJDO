# scJDO v2 — Implementation plan

Phase A audit output, written before any v2 code. Executed by the implementing AI against the spec in the user's `v2_instructions.md` (pasted). Everything below is checked against the repo at commit `a0b8ef4` (branch `main`).

> **Status:** v1 imports cleanly (`scjdo 0.3.0`). `scvi-tools 1.5.0.post1` and `dynamo-release 1.5.3` are already installed in the active environment — no new deps required for Phases A–C beyond what the spec already names. `torch` is 2.10.0.

---

## 1. Where v1 lives — module map

| v1 subpackage | Primary files | What it does |
|---|---|---|
| `scjdo.pp` | `_trajectory.py` | `prepare_trajectory`: normalize → HVG → PCA → kNN → DPT in one call. **Returns normalized expression + PCA.** This is the only normalization layer in v1. |
| `scjdo.models.drift` | `drift.py` (297 LOC) | `DriftField` (FiLM-conditioned MLP, spectral-norm output, DSM-trainable); `DriftConfig`. The **neural-ODE drift field** v2 keeps unchanged. |
| `scjdo.models.schrodinger_bridge` | `schrodinger_bridge.py` | Bridge model + config. Out-of-scope for v2 core, but notebooks 03/04 use it. |
| `scjdo.models.fourier_score_network` | `fourier_score_network.py` | Fourier-domain score net. Notebook 02 only; private symbols. |
| `scjdo.tl._drift` | `_drift.py` (1359 LOC) | `fit_drift` end-to-end: trains drift, computes per-cell Jacobians via autograd, calls `build_temporal_operator`, runs semi-NMF, scores instability, stores in `adata.uns['scjdo']`. **This is the function v2 forks from.** |
| `scjdo.tl._bridge` | `_bridge.py` | `fit_bridge`; parallel to `_drift.py` for bridge. |
| `scjdo.tl._regulators` | `_regulators.py` | `infer_regulators`, CollecTRI/TRRUST wiring, Jacobian-eigenvector-derived hybrid GRN. Downstream of fit; v2 reuses. |
| `scjdo.archetypes.windowing` | `windowing.py` (283 LOC) | `kernel_jacobian_tensor` (adaptive Gaussian kernel), `select_bandwidth` (R·C·L score), `build_temporal_operator`. **Pseudotime-slicing step v2 replaces.** |
| `scjdo.archetypes.decompose` | `decompose.py` (210 LOC) | `_seminmf` (column-wise NNLS, Lawson–Hanson), `jacobian_modes` (semi-NMF + ALS restarts + optional TV), `jacobian_modes_svd` (legacy). **v2 Step 8 wraps this.** |
| `scjdo.archetypes.koopman` | `koopman.py` (624 LOC) | Windowed linear-operator fit on vectorised Jacobian sequence; eigenvalues, growth rates, Nyquist, Henrici index, reactivity, transient gain, eigvec conditioning. **v2 Step 7 reuses these spectral descriptors.** |
| `scjdo.archetypes.cellrank` | `cellrank.py` | CellRank branch-prob reader. Keep. |
| `scjdo.atlas` | 11 files (regime_classifier, operator_metrics, flow_topology, visualization, etc.) | scOpAtlas: regime labels + flow topology from the Jacobian field. v2 operator-space naming (holder/fork/drift/damped-osc/growing-osc) partly overlaps with regime_classifier.py — reuse where possible. |
| `scjdo.cli.main` | `main.py` (207 LOC) | `scjdo drift` and `scjdo bridge` subcommands. v2 adds `scjdo2`. |
| `scjdo.pipeline` | `train_from_anndata.py`, `train_bridge.py`, `train_hybrid_grn.py` | Lower-level entry points behind the console scripts (`scjdo-train`, `scjdo-bridge`, `scjdo-atlas`). |
| `scjdo.validation` | `constructed_obstruction.py`, etc. | Synthetic obstructions for identifiability tests. Reused in Phase B. |
| `scjdo.simulate` | `trajectories.py` (28 LOC) | Only `euler_integrate` and `apply_archetype_modulation`. **No pitchfork synthetic lives here** — see §4 below. |

### Dependencies already available for v2

- **`scvi-tools 1.5.0.post1`** — used for Step 3 (count-likelihood rate model). Will register a custom `BaseModuleClass` subclass; the latent `z_c` is a *fixed* input (not an encoder output) so most of the standard VI pattern is replaced by a decoder-only training loop. Keep this module in `scjdo/v2/rates.py`.
- **`dynamo-release 1.5.3`** — spectral/Jacobian analysis helpers (e.g. `dynamo.vf.rank_jacobian_genes`, `dynamo.vf.jacobian`, `dynamo.pl.jacobian`). Borrow where they fit Step 7; do not vendor.

---

## 2. Mapping — v2 steps ↔ v1 modules ↔ v2 files

| v2 step | What it is | Reuses from v1 | New v2 file |
|---|---|---|---|
| 1. Positional layer | scVI encoder (default) / FA / PCA → frozen `z_c` | `scjdo.pp._trajectory` for the PCA path; `scvi.model.SCVI` for scVI; sklearn FA | `scjdo/v2/embed.py` |
| 2. Drift field | Keep v1 `DriftField`; relative timescale with `clock_calibrated=False` flag | `scjdo.models.drift.DriftField`, `scjdo.tl._drift.fit_drift` training inner loop | `scjdo/v2/drift.py` (thin wrapper) |
| 3. Gene rate model | scvi-tools NB module; **fixed `z` input, size factor as offset**, autodiff-friendly decoder | NEW (no v1 analogue — v1 differentiates log-expression) | `scjdo/v2/rates.py` |
| 4. Rate along drift | `d μ_g/dt = ∇_z μ_g(z_c) · f(z_c)` via torch autograd on the decoder | `scjdo.models.drift.DriftField` for `f`; autograd (**no finite differences**) | `scjdo/v2/rates.py` (same file) |
| 5. Neighbourhood Jacobian field | kNN in latent space; per-cell likelihood-weighted regression of rates on levels; **smoothness penalty** tying adjacent cells; sparse by construction | **Replaces** `scjdo.archetypes.windowing.build_temporal_operator`. Reuses only autograd and nn helpers. | `scjdo/v2/field.py` |
| 6. Fisher ledger | Diagonal Fisher at the fitted `J`; per-row (per-gene) independent regressions; reason codes `no_counts` / `no_variation` / `not_in_ball` | NEW | `scjdo/v2/fisher.py` |
| 7. Operator space | Eigenvalue stats (sorted by Re), stable/unstable/near-zero/complex-pair counts, spectral gap, Henrici index, transient gain. Fisher-weighted. | **Reuses** descriptors from `scjdo.archetypes.koopman` (henrici, reactivity, transient_gain, eigvec_cond). | `scjdo/v2/opspace.py` |
| 8. Archetypes | Semi-NMF on operator-space coords, Fisher-weighted cells, K by held-out likelihood, named from eigenvalues | **Wraps** `scjdo.archetypes.decompose._seminmf` / `jacobian_modes`. | `scjdo/v2/archetypes.py` |
| 9. Grammar (minimal) | Archetype sequences along drift; pairwise commutator norms `‖[A_k, A_{k+1}]‖` | NEW | `scjdo/v2/grammar.py` |
| IO | AnnData-based; `J` field + ledger as sparse arrays in `.obsm`/`.uns`; optional zarr for large N | Uses `scjdo.io.anndata.tensors_from_anndata` as a reference pattern | `scjdo/v2/io.py` |
| CLI | `scjdo2 run`, `scjdo2 compare` | Mirrors `scjdo.cli.main` shape | `scjdo/v2/cli.py` |
| Comparison report | Runs v1 and v2 on same dataset, writes `benchmarks/compare/<ds>/report.md` and `SUMMARY.md` | Uses both pipelines | `compare/v1_vs_v2.py` |

### Layout decision

v1 **stays in place** (do not move it under `scjdo/v1/`). Reason: v1 is pinned at `0.3.0` and is imported publicly as `scjdo.pp` / `scjdo.tl` / `scjdo.pl` from manuscript figures, external notebooks, the installed console entry points, and the EVIDENCE_LEDGER. Moving it under `scjdo/v1/` would require shims and would risk silent import failures in Figures_notebook/ and reproducibility/code/. v2 lives at `scjdo/v2/` as a sibling tree; its console entry point is `scjdo2`. This is explicitly permitted by the spec ("…or left in place with v2 added — choose whichever keeps v1 imports working").

### Package-level wiring

- `scjdo/v2/__init__.py` — exports `run`, `compare`, `embed`, `fit_rates`, `build_field`, `archetypes` convenience functions. Does **not** import v1 symbols into its own namespace; the two trees are callable in the same session but independent.
- `pyproject.toml` — add `scjdo2 = "scjdo.v2.cli:main"` to `[project.scripts]`. Deps unchanged (scvi-tools, dynamo-release are already in the active env; add as `install_requires` here only when the user confirms the broader install story — see §5 open question).
- `README.md` — add a "v2 (field-based operators)" section + a "why v2" paragraph pointing to EVIDENCE_LEDGER EL08 (the slicing failure the field layer resolves).
- `CHANGELOG.md` — new `[Unreleased] — v2 preview` block per phase commit.

---

## 3. Phase plan with acceptance checks

Each phase is a separate commit with a `CHANGELOG.md` entry.

### Phase A — Audit and plan (**in progress**)

- [x] Read repo structure, drift workflow, windowing, decompose, Koopman, CLI, notebooks.
- [x] Confirm v1 `import scjdo as sjd` works and `__version__ == "0.3.0"`.
- [x] Write this file (`V2_PLAN.md`).
- [ ] Add `tests/v2/` scaffold (`__init__.py`, `conftest.py` sharing v1's reproducibility seed fixture).
- [ ] Run existing v1 pytest suite; record pass/fail counts and durations under `benchmarks/v1_baseline/pytest_report.txt`.
- [ ] Execute (nbconvert) the five canonical v1 notebooks `01, 02, 03, 04, 05` from `examples/` and save their rendered HTML + any produced figure/CSV under `benchmarks/v1_baseline/<notebook_name>/`.

**Acceptance:** `V2_PLAN.md` exists (this file); v1 baseline is captured on disk; v1 tests pass or any failures are documented and acknowledged as pre-existing (not caused by v2).

### Phase B — Minimum field version (Steps 1, 2, 5, 7, 8) on existing rates

Build the operator-field machinery on top of v1's log-expression rate (what `fit_drift` already differentiates) so the field, operator-space, and archetype code is testable **without** the count model. Count-likelihood work is deferred to Phase C.

- Implement `scjdo/v2/embed.py` (scVI | FA | PCA selector; freezes encoder/loadings).
- Implement `scjdo/v2/drift.py` (thin wrapper on `scjdo.models.drift.DriftField`; `clock_calibrated: false` flag; passes seed through).
- Implement `scjdo/v2/field.py` with:
  - kNN in latent space (sklearn `NearestNeighbors` default; `faiss-cpu` optional, matches v1's dep posture).
  - per-cell regression of `d μ/dt ≈ J (μ − μ̄)` with likelihood weights.
  - **column mask**: for each cell, keep only genes whose within-ball variance exceeds a floor → sparse `J_c`.
  - **smoothness penalty** `λ · Σ_{(c, c') ∈ kNN} ‖J_c − J_{c'}‖²_F` tying neighbouring Jacobians. Implemented as a joint ridge solve on the stacked system, not an iterative smoother on data.
  - `k` selected by held-out NB (or Gaussian in this phase) log-likelihood over `k ∈ {15, 25, 50, 100}`.
  - Cells with fewer than `k` neighbours within the sane radius are flagged `recovered: false` — **no fabricated Jacobian**.
- Implement `scjdo/v2/opspace.py` (eigen sort, mode counts, spectral gap, Henrici / reactivity / transient gain / eigvec cond — ported directly from `scjdo.archetypes.koopman`).
- Implement `scjdo/v2/archetypes.py` (wraps `_seminmf`; Fisher-weight hook left as unit weights in Phase B; K via held-out log-likelihood).

**Phase B acceptance tests** (`tests/v2/test_phase_b_pitchfork.py`):

1. **Saddle recovery on the repo's 2-branch pitchfork benchmark.** The toggle-switch ground truth lives in `reproducibility/code/toggle_truth.py`; the V2-fixed benchmark is in `reproducibility/blind_saddle_benchmark/` with `τ_crit ∈ {0.1, 0.3, 0.5}` and a predeclared `±0.05` tolerance window (EVIDENCE_LEDGER EL08). **v2 must localise a positive leading real eigenvalue to within `±0.05` of `τ_crit` on at least 4 of 6 (τ_crit, seed) conditions**, where v1's slicing (`fit_drift` default) produces 0 of 6.
2. **Reproducibility across half-splits.** Random 50/50 cell split, run v2 on each half, compute per-cell leading eigenvalue, Spearman the two per-cell arrays. **ρ ≥ 0.8.**
3. **No silent fallbacks.** Cells flagged `recovered: false` must have their field entries left `NaN` and their obs-column `scjdo2_recovered == False`.

### Phase C — Count-likelihood rate model (Steps 3, 4, 6)

- Implement `scjdo/v2/rates.py`:
  - `scvi-tools` module subclassing `BaseModuleClass`.
  - **Input:** fixed `z_c` tensor (no encoder trained; `z_c` is frozen from Step 1).
  - **Decoder:** small shared MLP (penalised; dropout + weight decay) + gene-specific linear heads. Optional GP head reserved for a later task.
  - **Likelihood:** negative binomial, gene-specific dispersion `θ_g`.
  - **Size factor:** entered as an **offset** inside the NB rate — `log μ_cg = log s_c + h_g(z_c)`. Never divide.
  - **Autodiff hook:** `rates(mu_fn, f_fn, z)` returns `(mu, d_mu_dt)` using `torch.autograd.functional.jacobian` on `h_g` composed with the drift. Finite differences are prohibited.
- Switch `scjdo/v2/field.py` to use these rates and NB-likelihood weights in the regression step.
- Implement `scjdo/v2/fisher.py`:
  - Row-by-row: for each gene `g`, the regression that produced row `g` of `J_c` has its own log-likelihood; take its curvature w.r.t. the entries of row `g` at the fitted value. Diagonal only.
  - **Reason codes:** `no_counts` (<`min_counts` reads in the ball), `no_variation` (within-ball variance < floor), `not_in_ball` (gene masked out of this cell's column set). Stored as a per-cell, per-entry `int8` array alongside the sparse `J`.

**Phase C acceptance tests** (`tests/v2/test_phase_c_counts.py`):

1. **NB beats Poisson** on held-out log-likelihood on a planted-synthetic (NegativeBinomial.sample with known θ_g, z_c). Threshold: Δ(NB − Poisson) > 0 on 95% of genes.
2. **Ledger flags silent & constant genes.** Plant `n_silent = 50` all-zero genes and `n_constant = 50` genes with identical counts across cells; after fit, the Fisher ledger must flag all 100 with reason codes `no_counts` / `no_variation` respectively (recall ≥ 0.95; false-positive rate on the other genes < 0.05).
3. **Driver recovery on the synthetic fork.** Plant 10 driver genes whose rates diverge across the fork branches. Precision@10 after Fisher filtering of the top archetype's leading eigenvector ≥ 0.8.

### Phase D — Grammar scaffold + CLI

- `scjdo/v2/grammar.py`: order archetypes along the drift (parameterise by diffusion-pseudotime quantile of weighted cells); for each consecutive pair `(A_k, A_{k+1})` compute `‖[A_k, A_{k+1}]‖_F / (‖A_k‖‖A_{k+1}‖)` as the commutator norm. Nothing else — this is scaffolding.
- `scjdo/v2/cli.py`:
  - `scjdo2 run --input <h5ad> --embed scvi|fa|pca --k auto --out <dir>` → runs Steps 1–9, writes `adata_v2.h5ad`, `operator_field.npz`, `fisher_ledger.npz`, `archetypes.json`, figures.
  - `scjdo2 compare --v1 <dir> --v2 <dir> --dataset <name>` → produces `benchmarks/compare/<name>/report.md`.
- Register `scjdo2` in `pyproject.toml` entry points.

### Phase E — Rerun tutorials

- Create `notebooks/v2/` with one notebook per canonical v1 example (01–05 per README; **10** optional as a Koopman sanity check — see §5 open question). Same datasets, same seeds, v2 pipeline.
- Run `compare/v1_vs_v2.py` on each.
- Write `benchmarks/compare/<dataset>/report.md` (section template per spec §4).
- Write `benchmarks/compare/SUMMARY.md` — plain prose about where v2 improved, where it did not, and anything that got worse.

**The side-by-side figure per dataset:** v1 slicing archetype sequence (reusing `scjdo.pl.archetypes`) vs v2 operator field (new `scjdo.v2.viz.operator_field_scatter`) coloured by leading eigenvalue, fork marked.

---

## 4. Risks and open questions

### Deviations/clarifications from spec

1. **`τ_crit ≈ 0.004` disagrees with the repo's actual pitchfork benchmark.** The spec says "the synthetic pitchfork dataset in the repo … τ_crit ≈ 0.004". The repo's 2-branch pitchfork lives in `reproducibility/blind_saddle_benchmark/` (ground truth in `reproducibility/code/toggle_truth.py`) and uses `τ_crit ∈ {0.1, 0.3, 0.5}` with a predeclared ±0.05 tolerance window (EVIDENCE_LEDGER EL08). I will target that benchmark and interpret the spec's "≈ 0.004" as either a typo or a reference to a smaller tolerance band that is tighter than EL08's ±0.05. **Flagging; not blocking.**
2. **"factor analysis (v1 default)" disagrees with v1.** The spec says FA is the v1 embedding default; v1's `prepare_trajectory` actually defaults to PCA (notebook 01 and the CLI confirm). Phase B will offer `scvi | fa | pca` with the same semantics as the spec intends; v2 can default to `scvi` for the new pipeline while FA/PCA remain selectable. **Flagging; proceeding with scvi default per spec intent.**
3. **No `scjdo/simulate/pitchfork.py` exists.** Only `scjdo/simulate/trajectories.py` (28 LOC, an Euler integrator). The pitchfork benchmark will be pulled from `reproducibility/code/toggle_truth.py` into a lightweight new module `scjdo/v2/_bench/pitchfork.py` so tests can import it without reaching into `reproducibility/`.

### Technical risks

- **Smoothness penalty cost.** Phase B joint ridge over all cells is `O(N · k · d² · g_mean)` where `g_mean` is the median within-ball gene count. For `N = 2500` (Paul15), `k = 25`, `d = 20`, `g_mean = 2000` this is a few billion ops — tractable but needs sparse solve. Fall-back: block solve per weakly-connected kNN component.
- **scvi-tools 1.5 breaking changes.** The `BaseModuleClass` API in 1.5 differs from 1.3; v2's custom module must inherit from the current (1.5) base and use the Lightning-based training loop. Will pin `scvi-tools >= 1.5` once Phase C stabilises.
- **Dynamo spectral helpers run on their own data model.** `dynamo.vf.jacobian` expects a `dyn.read` AnnData with specific obsm layers. Rather than reshape data to Dynamo's model, Phase C reuses only the pure numerical descriptors from `scjdo.archetypes.koopman` (which already implement Henrici / reactivity / transient gain). Dynamo stays available but un-wired unless a specific function (e.g. `rank_jacobian_genes`) becomes necessary — flagged if so.
- **Private-API notebooks (02, 03, 05).** The three notebooks that import private symbols (`MultiBandScoreNet`, `SchrodingerBridge`, `StableOperatorAtlas`) will break if v2 renames anything in those modules. v2 does not touch v1 code, so these should keep running. **Add a smoke test that imports each private symbol used by the five canonical notebooks before every Phase B+ commit.**

### Open question for the user

**(Q1)** The spec says "five tutorial notebooks". `examples/README.md` promises 01–05; notebook 10 (Koopman) was added later and is also in the README table. Should Phase E rerun exactly the five (01–05), or add 10 as a sixth? **Default: run 01–05 strictly; include 10 only if time permits.**

**(Q2)** The install story — should the v2 refactor add `scvi-tools >= 1.5` and `dynamo-release >= 1.5` as hard runtime deps of `scjdo`? They are present in the active env but not yet in `pyproject.toml`. **Default: add them as `install_requires` once Phase C is green; before that, keep them optional-extras `[v2]`.**

**(Q3)** Phase B acceptance test calibration: EL08's existing benchmark uses `τ_crit ∈ {0.1, 0.3, 0.5}` with `±0.05`. The spec writes `τ_crit ≈ 0.004`. **Confirm the acceptance window: either (a) EL08's `±0.05` on `{0.1, 0.3, 0.5}` conditions (what the repo supports), or (b) a tighter `±0.004` window that the repo's benchmark cannot resolve at current grid resolution.** Choosing (a) by default.

---

## 5. Definition of done (restated for this plan)

- v1 untouched (`scjdo 0.3.0`) and passing its existing tests.
- `scjdo2 run` completes on all five canonical tutorial datasets (01–05).
- Phase B and Phase C acceptance tests encoded in `tests/v2/` and passing.
- `benchmarks/compare/SUMMARY.md` + per-dataset reports + side-by-side figures.
- `README.md` and `CHANGELOG.md` updated; this `V2_PLAN.md` updated with what actually got built and any deviations, each with a one-line reason.

---

*This plan is checked against the repo at `main @ a0b8ef4`. If the user approves it, Phase A finishes by scaffolding `tests/v2/` and capturing the v1 baseline; implementation of Phase B begins immediately after.*
