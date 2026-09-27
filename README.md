# scJDO — single-cell Jacobian drift operators

**scJDO** infers how local dynamical sensitivity evolves during cell fate transitions.
It learns a drift field from scRNA-seq data, computes temporal Jacobian operators along
a trajectory, decomposes them into recurrent regulatory archetypes, and identifies the
genes and transcription factors that drive instability at each transition point.

```python
import scanpy as sc
import scjdo as sjd

adata = sc.datasets.paul15()
sjd.pp.prepare_trajectory(adata, groupby='paul15_clusters', root='7MEP')
sjd.tl.fit_drift(adata, n_archetypes=5, n_epochs=5000)
sjd.pl.summary_figure(adata, save='figure3.pdf')
```

```bash
# Same analysis from the command line
scjdo drift paul15.h5ad --groupby paul15_clusters --root 7MEP --out results/
```

---

## Install

```bash
git clone https://github.com/manarai/scJDO
cd scJDO
pip install -e .
```

Requires Python ≥ 3.9, PyTorch ≥ 2.2.

**Optional extras:**

```bash
pip install decoupler          # CollecTRI regulatory network (recommended)
pip install networkx           # regulator_network() graph figure
pip install faiss-cpu          # faster kNN for velocity prior
```

---

## Three workflows

| Workflow | When to use | Key call |
|---|---|---|
| **Drift field** | Any scRNA-seq with pseudotime | `sjd.tl.fit_drift` |
| **Schrödinger Bridge** | Two defined populations (young/old, treated/ctrl) | `sjd.tl.fit_bridge` |
| **Regulatory inference** | After either — link instability genes to TF regulators | `sjd.tl.infer_regulators` |

---

## API overview

### `sjd.pp` — Preprocessing

| Function | What it does |
|---|---|
| `prepare_trajectory(adata, groupby, root)` | Normalize → HVG → PCA → kNN → DPT pseudotime in one call |

### `sjd.tl` — Analysis

| Function | What it does |
|---|---|
| `fit_drift(adata, ...)` | Train drift field, compute Jacobian tensor, decompose into archetypes |
| `fit_bridge(adata, ...)` | Train Schrödinger Bridge between source/target populations |
| `decompose_archetypes(J_tensor, method=...)` | Backend-agnostic decomposition — `snmf` (default) and `svd` return archetypes; `koopman` returns eigen-**modes** with a spectral / geometry diagnostics dict |
| `get_instability_genes(adata)` | Extract top instability-driving genes per archetype (drift) |
| `get_bridge_instability_genes(adata)` | Same for forward and backward bridge directions |
| `infer_regulators(adata, ...)` | Link instability genes to upstream TF regulators via network database |

### `sjd.pl` — Figures

**Drift field:**

| Function | Figure |
|---|---|
| `summary_figure(adata)` | 4-panel: drift field, sensitivity, archetypes, coordination |
| `drift_field(adata)` | Streamplot on PCA embedding |
| `sensitivity(adata)` | Max Re(λ) across pseudotime |
| `archetypes(adata)` | Archetype activation profiles |
| `coordination(adata)` | Temporal correlation heatmap |
| `instability_genes(adata)` | Top instability genes across pseudotime + heatmap |

**Schrödinger Bridge:**

| Function | Figure |
|---|---|
| `bridge_summary(adata)` | 7-panel summary |
| `bridge_trajectories(adata)` | PCA + trajectory paths (forward / backward / both) |
| `bridge_instability(adata)` | Forward vs backward instability curves |
| `bridge_archetypes(adata)` | Archetype activation for both directions |
| `bridge_genes(adata)` | Gene × archetype heatmaps |
| `bridge_gene_comparison(adata)` | Forward vs backward unique gene lists |

**Regulatory network:**

| Function | Figure |
|---|---|
| `regulator_summary(adata)` | 4-panel: bar chart, heatmap, scatter, profiles |
| `regulator_barplot(adata)` | Ranked bar chart colored by mean instability |
| `regulator_heatmap(adata)` | TF × archetype instability heatmap |
| `regulator_scatter(adata)` | Quality vs quantity (n_targets vs mean_instability) |
| `regulator_profiles(adata)` | Target instability across pseudotime for top TFs |
| `regulator_network(adata)` | Hybrid graph: solid=reference, dashed=de novo |

### CLI

```bash
scjdo drift  input.h5ad --groupby CLUSTER_COL --root ROOT_CLUSTER --out DIR/
scjdo bridge input.h5ad --groupby CLUSTER_COL --root ROOT_CLUSTER --out DIR/
```

---

## What gets stored

Both `fit_drift` and `fit_bridge` store all results in `adata.uns` so every plotting
function can read directly without recomputing:

```
adata.uns['scjdo']          ← drift results
adata.uns['scjdo_bridge']   ← bridge results
adata.uns['scjdo_regulators'] ← regulator inference results
adata.obsm['X_drift']         ← per-cell drift vectors
adata.obsm['X_velocity_pseudo'] ← pseudotime-gradient velocity prior
```

---

## Notebooks

Five end-to-end tutorials are in [`examples/`](examples/README.md):

| Notebook | Analysis |
|---|---|
| `01_paul15_hybrid_drift_tutorial` | Drift field + archetypes + instability genes + regulators |
| `02_paul15_fourier_tutorial` | Fourier-domain score network, spectral validation |
| `03_schrodinger_bridge_tutorial` | Bridge on 2D synthetic data, forward/backward instability |
| `04_paul15_schrodinger_tutorial` | Bridge on Paul15 PCA space, forward vs backward gene lists |
| `05_scopatlas_complete_workflow` | Operator atlas on pre-trained model |

Figure-generating notebooks for the manuscript are in [`Figures_notebook/`](Figures_notebook/).

---

## scOpAtlas

`scjdo.atlas` classifies each cell by the local dynamical regime of the trained
drift field — stable / plastic / unstable / deeply-stable — from the eigenvalue
spectrum of the Jacobian. See:

- [`docs/scopatlas/README.md`](docs/scopatlas/README.md) — full documentation
- [`docs/scopatlas/QUICKSTART.md`](docs/scopatlas/QUICKSTART.md) — 5-minute walkthrough
- [`docs/scopatlas/DESIGN.md`](docs/scopatlas/DESIGN.md) — design notes and open issues

---

## Optional spectral backend — Koopman

The default archetype decomposition is semi-NMF (sparse, non-negative,
interpretable as regulatory programs). scJDO also ships a **Koopman**
backend that treats the vectorised Jacobian sequence as a trajectory of
observables and fits a windowed linear operator on it — a complementary
spectral lens. The output is Koopman **modes** (eigenvectors — not
non-negative programs), continuous-time growth / decay rates, candidate
oscillation frequencies bounded by an explicit Nyquist limit, and
branch-free operator-geometry descriptors (departure-from-normality,
reactivity, transient gain, eigenvector conditioning).

```python
sjd.tl.fit_drift(
    adata,
    archetype_method="koopman",   # or "snmf" (default)
    koopman_kwargs=dict(mode="local", window_half=8, ridge=1e-4,
                        whiten=False),   # reduced-space metric — see KOOPMAN.md §2
)
koop = adata.uns["scjdo"]["koopman"]
koop["eigenvalues"], koop["growth_rates"], koop["freqs"]
koop["delta_tau"], koop["nyquist_ang_freq"], koop["nyquist_cycle_freq"]
koop["geometry"]     # henrici, reactivity, transient_gain, eigvec_cond
koop["whiten"]       # metric echo — geometry descriptors depend on this
```

The Koopman output slots into the same downstream code as semi-NMF
(shared `(K, D, D)` return shape and per-time activations; gene scores
and instability tables just work). Best on dense time-resolved
trajectories; less suited to sparse branch-heavy atlases. See
[`KOOPMAN.md`](KOOPMAN.md) for the full derivation, per-window operator
model, eigenvalue interpretation, and caveats specific to snapshot
scRNA-seq.

---

## Mathematical background

For the full mathematical derivation see [`MATH.md`](MATH.md). For the
Koopman spectral backend see [`KOOPMAN.md`](KOOPMAN.md).

**Using scVI, Palantir, Harmony, or Slingshot?**
scJDO accepts any latent space or pseudotime from any tool — one parameter
change connects them. See [`INTEROPERABILITY.md`](INTEROPERABILITY.md).

**Core idea:** model cell dynamics as a stochastic differential equation

$$dX_t = f_\theta(X_t, t)\,dt + \sigma\,dW_t$$

where the drift field $f_\theta$ is parameterized by a FiLM-conditioned neural network
trained via denoising score matching. Local Jacobians $J(x,t) = \nabla_x f_\theta$
are aggregated across pseudotime by **adaptive Gaussian kernel windowing**

$$\bar J(\tau;h) = \frac{\sum_i e^{-(\tau-\tau_i)^2/2h^2}\,J_i}
                        {\sum_i e^{-(\tau-\tau_i)^2/2h^2}}$$

with the bandwidth $h$ selected by maximising
$S(h) = R(h)\!\cdot\!C(h)\!\cdot\!L(h)$ — bootstrap reproducibility, peak
contrast, and peak localisation — subject to an effective-sample-size floor.
The resulting temporal Jacobian tensor is decomposed by semi-NMF into $K$
recurrent operator archetypes with non-negative temporal activation profiles.
The legacy fixed-window scheme is available via `windowing='fixed'`; see
[`Manuscript/adaptive_kernel_windowing.ipynb`](Manuscript/adaptive_kernel_windowing.ipynb)
for the derivation and side-by-side validation.

---

## Known limitations (v0.3.x)

scJDO is a snapshot-derived Jacobian estimator. What that means, in terms measured against real datasets during r22–r26 calibration and the wrap-up tasks:

- **Leading-direction loadings are seed-dependent under default settings.** The 30 "leading-direction loadings" produced by a single `fit_drift` run are not a stable object across random seeds at default `n_epochs=5000` and `bandwidth='auto'`. Report multi-seed statistics (≥3 seeds, spread or CI), the mask status, the bandwidth-sweep result, and the prior settings whenever gene lists derived from the Jacobian are shown.
- **The sensitive mask does not fire on the benchmark datasets used in the manuscript.** All gene lists in the figures come from the unmasked fallback path in `sjd.tl.infer_regulators`. Consumers who rely on the mask should treat it as opt-in, not opt-out, and verify per dataset.
- **`bias_strength` and `vel_scale` behaviour as measured in r24 / Gate 0a.** The additive pseudotime-gradient prior slot (formerly "velocity prior") *is* wired into the model (`V_ref` and 50/53 state_dict tensors differ between `bias_strength=1.5` and `bias_strength=0` at the same seed), but which basin the training lands in is dominated by the seed at these fit parameters. Do not read `vel_scale × bias_strength` interactions as attractor structure — they are training basins of a non-convex fit.
- **Fig 3 requires Palantir 1.4.4 and MAGIC imputation to reproduce.** The regulator table and archetype panels of Fig 3 use `palantir.utils.run_diffusion_maps` / `determine_multiscale_space` at 1.4.4, followed by MAGIC on the branch-restricted expression matrix, per the r24-pinned pipeline. Different Palantir versions may give different terminal-cell assignments and slightly different pseudotime; the downstream operator statistics inherit this dependence.
- **What snapshot-derived Jacobians CAN be trusted for**: covariance structure of local operators; estimator fidelity on synthetic ground truth; and (with velocity supervision) time-varying operators whose leading direction agrees per-window with a supervision reference. See `EVIDENCE_LEDGER.md` for the full list of established, qualified, and negative findings.
- **What they CANNOT be trusted for** on snapshot data alone: predicting downstream fate or velocity beyond what an expression baseline recovers; identifying commitment before differentiation onset from covariance softening; recovering rotation direction from pseudotime without an additive prior.

---

## Citation

If you use scJDO, please cite:

> Redd D., Green S., Terooatea T.W. (2026). scJDO: Inferring time-varying dynamical
> operators from single-cell transcriptomic data. *[journal]*.

---

## Version

Current version: **0.3.0** — see [`CHANGELOG.md`](CHANGELOG.md) for what's new.
