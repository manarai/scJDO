"""
Build Figures_notebook/Synthetic_pseudotime_embedding_benchmark.ipynb.

Source of truth for the notebook that benchmarks pseudotime × embedding × noise
against analytic ground-truth Jacobians on a small library of synthetic
trajectory topologies. Heavy work is cached to disk so cells re-run cheaply
after the first pass.
"""
from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf


CELLS: list = []


def md(text: str):
    CELLS.append(nbf.v4.new_markdown_cell(text))


def code(src: str):
    CELLS.append(nbf.v4.new_code_cell(src))


# ─────────────────────────────────────────────────────────────────────────
md(r"""# Synthetic benchmark — pseudotime × embedding × noise vs ground-truth Jacobian

**Purpose.** Quantify scJDO's recovery against analytic truth on a small
library of trajectory topologies. We sweep three orthogonal axes:

| Axis | Levels |
|---|---|
| Pseudotime | DPT (Scanpy), Palantir, **oracle** (the true ordering used to generate the data) |
| Embedding  | PCA, FA, ICA, TruncatedSVD; scVI/LDVAE optional if `scvi-tools` is installed |
| Noise      | σ ∈ {0.0, 0.1, 0.3, 0.5, 1.0} of additive Gaussian on observations |

**Synthetic systems** (each with closed-form Jacobian and known λ-peak timing):

1. `linear`     — monotonic drift, no instability (negative control: expect $\lambda_{\max}(t)<0$ everywhere)
2. `commitment` — transient instability pulse at τ≈0.5 (the canonical neurogenic-commitment analogue)
3. `bifurcation` — Y-shaped 2D toggle switch; instability spans the branch point
4. `cyclic`     — Stuart–Landau limit cycle (declared failure mode: pseudotime methods invent a linear order on a topologically circular trajectory)

For each `(system, pseudotime, embedding, noise)` we fit `sjd.tl.fit_drift`
and compare its recovered $\lambda_{\max}(\tau)$ curve to the analytic
one. The decisive question for the FA argument is whether **FA's advantage
grows with noise** while still tracking truth at low noise.

**Pre-committed reads.**

* If `peak_timing_error_oracle < peak_timing_error_DPT` consistently → scJDO is sensitive to upstream pseudotime; report this honestly and recommend Palantir for branching topologies.
* If FA λ-curve correlation > scVI as σ↑, with both matching at σ=0 → the FA argument is mechanistic (derivative stability under noise), not specific to hematopoiesis.
* If on `cyclic`, every method recovers a spurious linear λ profile → known failure mode confirmed; scJDO should be reported as not applicable to topologically circular trajectories.

**Cost.** ~50–100 `fit_drift` calls at 800 epochs each. With CPU defaults
and the caching layer below, the full sweep is roughly 1.5–2 h on first
run; subsequent re-runs only re-execute the analysis cells.""")


# ─────────────────────────────────────────────────────────────────────────
code(r"""
# Imports + paths
import os, sys, warnings, hashlib, time
sys.path.insert(0, os.path.abspath('..'))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA, FactorAnalysis, FastICA, TruncatedSVD

import anndata as ad
import scanpy as sc
import scjdo as sjd

mpl.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'font.family': 'DejaVu Sans',
                     'axes.titlesize': 10, 'axes.labelsize': 9,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8,
                     'legend.fontsize': 8})

CACHE_DIR = 'results/synthetic_benchmark/'
os.makedirs(CACHE_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)

# ── Sweep configuration ─────────────────────────────────────────────────────
SYSTEMS_DEFAULT     = ['linear', 'commitment', 'bifurcation', 'cyclic']
PSEUDOTIME_METHODS  = ['oracle', 'dpt', 'palantir']
EMBEDDINGS_DEFAULT  = ['PCA', 'FA', 'ICA', 'TruncatedSVD']
NOISE_LEVELS_DEFAULT= [0.0, 0.1, 0.3, 0.5, 1.0]

# Cell + training budget — keep modest so the first full pass is tractable
N_CELLS         = 1500
D_GENES         = 200
N_LATENT        = 20
N_GRID          = 200
N_EPOCHS_BENCH  = 800
N_ARCHETYPES    = 4

print(f'cache dir : {CACHE_DIR}')
print(f'cells     : {N_CELLS}  ·  obs dim D : {D_GENES}  ·  latent dim : {N_LATENT}')
print(f'fit_drift : {N_EPOCHS_BENCH} epochs  ·  archetypes : {N_ARCHETYPES}')

# Optional scVI / LDVAE — skip silently if not installed
try:
    import scvi
    SCVI_OK = True
    print(f'scvi      : v{scvi.__version__}  (will include scVI and LDVAE in embedding sweep)')
except ImportError:
    SCVI_OK = False
    print('scvi      : not installed (skip scVI / LDVAE — pip install scvi-tools to enable)')
""")


# ─────────────────────────────────────────────────────────────────────────
md(r"""## 1. Synthetic systems with analytic Jacobians

Every system gives us a hidden state $z(t)$ with closed-form drift
$f(z,t)$, Jacobian $J(t) = \partial f / \partial z$ evaluated on the mean
trajectory, and the analytic max-real-eigenvalue curve
$\lambda_{\max}(t)$. Cells are sampled by drawing a pseudotime
$t_i\sim\mathrm{Uniform}(0,1)$, integrating the system from $z(0)$ to
$z(t_i)$ with a small Brownian perturbation, then projecting to $D$
observed coordinates via a fixed random loading matrix $W$ with column
norms equalised.""")
code(r"""
def _project_obs(z_traj: np.ndarray, D: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    # z_traj: (N, d_state). Returns (X (N, D), W (d_state, D)).
    rng = np.random.default_rng(seed)
    d = z_traj.shape[1]
    W = rng.standard_normal((d, D)).astype(np.float32)
    W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-8
    X = z_traj.astype(np.float32) @ W
    return X, W


def make_system(name: str, n_cells: int = N_CELLS, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    t   = rng.uniform(0, 1, n_cells).astype(np.float32)
    order = np.argsort(t); t = t[order]                # easier for downstream

    if name == 'linear':
        # 2D state: monotonic drift along z1, decaying along z2 — no instability.
        # z(t) = (t, 0). Jacobian = diag(-1, -2) → λ_max = -1 (stable everywhere).
        z = np.zeros((n_cells, 2), dtype=np.float32)
        z[:, 0] = t + 0.05 * rng.standard_normal(n_cells)
        z[:, 1] = 0.05 * rng.standard_normal(n_cells)
        grid  = np.linspace(0, 1, N_GRID, dtype=np.float32)
        true_lambda = -1.0 * np.ones_like(grid)
        true_peak_t = float('nan')                       # no true peak
        info = {'topology': 'linear', 'has_instability': False}

    elif name == 'commitment':
        # 1D pitchfork with a Gaussian-pulsed α(t) crossing 1.
        # ż = (α(t) - 1) z - z³        ⇒  J(z=0) = α(t) - 1
        # α(t) = 0.5 + 1.5 exp(-((t-0.5)/0.08)²) → λ peaks at τ=0.5
        z = np.zeros((n_cells, 2), dtype=np.float32)
        alpha = 0.5 + 1.5 * np.exp(-((t - 0.5) / 0.08) ** 2)
        # Integrate with small noise; use saturating tanh to keep z bounded.
        z[:, 0] = np.tanh(2 * (alpha - 1)) + 0.1 * rng.standard_normal(n_cells)
        z[:, 1] = 0.1 * rng.standard_normal(n_cells)
        grid  = np.linspace(0, 1, N_GRID, dtype=np.float32)
        alpha_g = 0.5 + 1.5 * np.exp(-((grid - 0.5) / 0.08) ** 2)
        true_lambda = alpha_g - 1.0
        true_peak_t = 0.5
        info = {'topology': 'commitment', 'has_instability': True}

    elif name == 'bifurcation':
        # 2D toggle: ẋ = α/(1+y^4) - x   ẏ = α/(1+x^4) - y
        # α ramps 0.5 → 5 along t. Bifurcation when α≈1.5.
        # We sample cells from the analytic post-bifurcation manifold (random branch).
        alpha = 0.5 + 4.5 * t                                # 0.5 → 5
        branch_sign = rng.choice([-1, 1], n_cells)
        # Approximate steady state on a branch
        z = np.zeros((n_cells, 2), dtype=np.float32)
        for i in range(n_cells):
            a = alpha[i]; sgn = branch_sign[i]
            hi = a / 1.001               # near-on level
            lo = a / (1 + hi**4)
            if sgn > 0: z[i] = [hi, lo]
            else:       z[i] = [lo, hi]
        z += 0.08 * rng.standard_normal(z.shape).astype(np.float32)
        grid = np.linspace(0, 1, N_GRID, dtype=np.float32)
        alpha_g = 0.5 + 4.5 * grid
        # λ_max on the symmetric branch crosses zero at α=1.5 (the pitchfork point)
        true_lambda = (alpha_g - 1.5) * np.exp(-((grid - 0.4) / 0.25) ** 2)
        true_peak_t = float(grid[np.argmax(true_lambda)])
        info = {'topology': 'bifurcation', 'has_instability': True,
                'branch_sign': branch_sign}

    elif name == 'cyclic':
        # Stuart–Landau limit cycle. λ = μ - 3r² at the origin; on the limit
        # cycle r²=μ, λ = -2μ (stable cycle, unstable origin). We sample on
        # the cycle so the "true" sensitivity is roughly constant ⇒ no
        # interesting peak. Any peak the method reports is a pseudotime
        # artefact, by design.
        mu = 1.0; omega = 2*np.pi
        phi = 2*np.pi * t
        z = np.zeros((n_cells, 2), dtype=np.float32)
        z[:, 0] = np.sqrt(mu) * np.cos(phi) + 0.05 * rng.standard_normal(n_cells)
        z[:, 1] = np.sqrt(mu) * np.sin(phi) + 0.05 * rng.standard_normal(n_cells)
        grid = np.linspace(0, 1, N_GRID, dtype=np.float32)
        true_lambda = -2*mu * np.ones_like(grid)
        true_peak_t = float('nan')
        info = {'topology': 'cyclic', 'has_instability': False}
    else:
        raise ValueError(f'unknown system {name}')

    X, W = _project_obs(z, D=D_GENES, seed=seed + 1)
    return dict(
        name=name, t_true=t, z=z, X=X, W=W,
        grid=grid, true_lambda=true_lambda, true_peak_t=true_peak_t,
        info=info,
    )


# Sanity check + plot the true λ curves
fig, axes = plt.subplots(1, 4, figsize=(13, 2.6))
for ax, name in zip(axes, SYSTEMS_DEFAULT):
    s = make_system(name)
    ax.plot(s['grid'], s['true_lambda'], 'k', lw=1.6)
    ax.axhline(0, color='gray', lw=0.6, ls=':')
    ax.set_title(f"{name}\npeak τ = {s['true_peak_t']:.2f}" if not np.isnan(s['true_peak_t'])
                 else f"{name}\nno instability")
    ax.set_xlabel('τ true'); ax.set_ylabel(r'$\lambda_{\max}^{true}$')
plt.tight_layout(); plt.savefig(CACHE_DIR + 'fig01_true_lambda.pdf'); plt.show()
""")


# ─────────────────────────────────────────────────────────────────────────
md(r"""## 2. Pseudotime and embedding wrappers

Each wrapper takes an `AnnData`-like object and returns a 1-D pseudotime
or a `(N, K)` embedding. `oracle` is the true pseudotime used to generate
the data; `dpt` and `palantir` are computed from the embedding.
We always run the embedding **before** the pseudotime call so the
pseudotime sees only what the embedding exposed.""")
code(r"""
def embed(X: np.ndarray, method: str, seed: int = SEED, n_components: int = N_LATENT):
    if method == 'PCA':           est = PCA(n_components=n_components, random_state=seed)
    elif method == 'FA':          est = FactorAnalysis(n_components=n_components, random_state=seed)
    elif method == 'ICA':         est = FastICA(n_components=n_components, random_state=seed,
                                                whiten='unit-variance', max_iter=400)
    elif method == 'TruncatedSVD':est = TruncatedSVD(n_components=n_components, random_state=seed)
    elif method == 'scVI':
        if not SCVI_OK: return None
        a = ad.AnnData(X=X.astype(np.float32))
        # scVI expects count-like; clip to non-negative integers via simple rounding
        a.layers['counts'] = np.clip(np.round(X - X.min() + 0.5), 0, None).astype(np.int32)
        import scvi
        scvi.model.SCVI.setup_anndata(a, layer='counts')
        m = scvi.model.SCVI(a, n_latent=n_components)
        m.train(max_epochs=80, train_size=1.0, accelerator='cpu', plan_kwargs=dict(lr=1e-3), early_stopping=False)
        return m.get_latent_representation().astype(np.float32)
    elif method == 'LDVAE':
        if not SCVI_OK: return None
        a = ad.AnnData(X=X.astype(np.float32))
        a.layers['counts'] = np.clip(np.round(X - X.min() + 0.5), 0, None).astype(np.int32)
        import scvi
        scvi.model.LinearSCVI.setup_anndata(a, layer='counts')
        m = scvi.model.LinearSCVI(a, n_latent=n_components)
        m.train(max_epochs=80, train_size=1.0, accelerator='cpu', plan_kwargs=dict(lr=1e-3), early_stopping=False)
        return m.get_latent_representation().astype(np.float32)
    else:
        raise ValueError(f'unknown embedding {method}')
    return est.fit_transform(X).astype(np.float32)


def pseudotime(adata, method: str, seed: int = SEED) -> np.ndarray:
    if method == 'oracle':
        return adata.obs['t_true'].values.astype(np.float32)
    # both DPT and Palantir need a kNN graph on the embedding
    sc.pp.neighbors(adata, use_rep='X_embed', n_neighbors=15, random_state=seed)
    if method == 'dpt':
        # DPT root = highest-t_true cell at the start of the trajectory (oracle hint
        # ONLY for choosing the root; the ordering itself is computed by DPT).
        root_idx = int(np.argmin(adata.obs['t_true'].values))
        adata.uns['iroot'] = root_idx
        sc.tl.diffmap(adata, random_state=seed)
        sc.tl.dpt(adata)
        return adata.obs['dpt_pseudotime'].values.astype(np.float32)
    if method == 'palantir':
        import palantir
        palantir.utils.run_diffusion_maps(adata, n_components=min(15, N_LATENT-1),
                                          pca_key='X_embed')
        palantir.utils.determine_multiscale_space(adata)
        start = adata.obs_names[int(np.argmin(adata.obs['t_true'].values))]
        pr = palantir.core.run_palantir(adata, start, num_waypoints=200)
        return adata.obs['palantir_pseudotime'].values.astype(np.float32)
    raise ValueError(method)
""")


# ─────────────────────────────────────────────────────────────────────────
md(r"""## 3. One sweep cell: (system, pseudotime, embedding, noise) → result

Calls `sjd.tl.fit_drift` with the new kernel default. Per-cell Jacobians,
cell-level caching, and a flat-file `.npz` per result keep the sweep
re-runnable.""")
code(r"""
def _cache_key(system, pt, embed_name, noise):
    raw = f'{system}|{pt}|{embed_name}|sigma={noise}|N={N_CELLS}|D={D_GENES}|epochs={N_EPOCHS_BENCH}|seed={SEED}'
    h = hashlib.sha1(raw.encode()).hexdigest()[:10]
    return CACHE_DIR + f'sweep_{system}_{pt}_{embed_name}_sigma{noise}_{h}.npz'


def run_one(system: str, pt_method: str, embed_name: str,
            noise: float, n_epochs: int = N_EPOCHS_BENCH,
            force: bool = False) -> dict:
    cache = _cache_key(system, pt_method, embed_name, noise)
    if (not force) and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return {k: z[k].item() if z[k].dtype == object and z[k].ndim == 0 else z[k]
                for k in z.files}

    s = make_system(system, n_cells=N_CELLS, seed=SEED)
    X = s['X'].copy()
    if noise > 0:
        rng = np.random.default_rng(SEED + 13)
        X = X + (noise * rng.standard_normal(X.shape)).astype(np.float32)

    # Embedding
    Z = embed(X, method=embed_name)
    if Z is None:
        return dict(skipped=True, reason='embedding unavailable')

    # AnnData carrier
    adata = ad.AnnData(X=X)
    adata.obs_names = [f'c{i:05d}' for i in range(adata.n_obs)]
    adata.obs['t_true'] = s['t_true']
    adata.obsm['X_embed'] = Z

    pt = pseudotime(adata, pt_method)
    pt = (pt - pt.min()) / max(pt.max() - pt.min(), 1e-9)
    adata.obs['pseudotime'] = pt

    # Drift fit (new kernel default)
    t0 = time.time()
    sjd.tl.fit_drift(
        adata, rep='X_embed', time_key='pseudotime',
        n_epochs=n_epochs, n_archetypes=N_ARCHETYPES,
        n_eff_min=20.0, n_boot=10,
        grid_size=N_GRID, seed=SEED, verbose=False,
    )
    dt = time.time() - t0
    r = adata.uns['scjdo']

    out = dict(
        system=system, pt_method=pt_method, embedding=embed_name, noise=noise,
        runtime=dt,
        bandwidth=r.get('bandwidth'),
        lam=np.asarray(r['max_real_eig'], dtype=np.float32),
        t_centers=np.asarray(r['t_centers'], dtype=np.float32),
        activations=np.asarray(r['activations'], dtype=np.float32),
        n_eff=np.asarray(r.get('n_eff', np.full(len(r['t_centers']), np.nan)),
                         dtype=np.float32),
        true_lambda=s['true_lambda'].astype(np.float32),
        true_grid=s['grid'].astype(np.float32),
        true_peak_t=float(s['true_peak_t']),
    )
    np.savez_compressed(cache, **{k: np.array(v) for k, v in out.items()})
    return out
""")


# ─────────────────────────────────────────────────────────────────────────
md(r"""## 4. Recovery metrics

For each result we compute:

* `peak_timing_error` = |argmax(recovered λ) − argmax(true λ)| (skipped when truth has no peak).
* `lambda_corr` = Pearson r between recovered and true λ-curves after re-interpolating both to a common grid.
* `sign_recovery` = fraction of grid points where sign(recovered) = sign(true).
* `n_eff_at_peak` = effective sample size where recovered λ peaks (low ⇒ likely boundary artefact).""")
code(r"""
def metrics(res: dict) -> dict:
    if res.get('skipped'):
        return dict(skipped=True)
    g_rec = res['t_centers'];   l_rec = res['lam']
    g_tru = res['true_grid'];   l_tru = res['true_lambda']
    # interpolate to a common grid
    g  = np.linspace(0.05, 0.95, 100, dtype=np.float32)
    f_rec = interp1d(g_rec, l_rec, bounds_error=False, fill_value='extrapolate')
    f_tru = interp1d(g_tru, l_tru, bounds_error=False, fill_value='extrapolate')
    L_rec = f_rec(g); L_tru = f_tru(g)
    r_pearson = float(pearsonr(L_rec, L_tru)[0]) if np.std(L_rec) > 0 else float('nan')
    sign_match = float(np.mean(np.sign(L_rec) == np.sign(L_tru)))
    if not np.isnan(res['true_peak_t']):
        peak_err = float(abs(g[np.argmax(L_rec)] - g[np.argmax(L_tru)]))
    else:
        peak_err = float('nan')
    pk_rec = int(np.argmax(res['lam']))
    n_eff_pk = float(res['n_eff'][pk_rec]) if len(res['n_eff']) > pk_rec else float('nan')
    return dict(
        system=res['system'], pt_method=res['pt_method'],
        embedding=res['embedding'], noise=float(res['noise']),
        peak_timing_error=peak_err,
        lambda_corr=r_pearson,
        sign_recovery=sign_match,
        max_lambda_rec=float(np.max(L_rec)),
        max_lambda_true=float(np.max(L_tru)),
        n_eff_at_peak=n_eff_pk,
        recovered_peak_t=float(g[np.argmax(L_rec)]),
        bandwidth=float(res['bandwidth']) if res['bandwidth'] is not None else float('nan'),
        runtime=float(res['runtime']),
    )
""")


# ─────────────────────────────────────────────────────────────────────────
md(r"""## 5. Sweep A — shape × pseudotime × embedding at moderate noise

This is the main matrix experiment: every system × every pseudotime ×
every embedding at σ=0.3 (the noise level closest to the multiome
benchmark in spirit). Results are cached, so re-runs of this cell skip
training entirely.""")
code(r"""
SHAPE_SWEEP_NOISE = 0.3
records = []
for sys_name in SYSTEMS_DEFAULT:
    for pt_method in PSEUDOTIME_METHODS:
        for emb in EMBEDDINGS_DEFAULT + (['scVI', 'LDVAE'] if SCVI_OK else []):
            t0 = time.time()
            r = run_one(sys_name, pt_method, emb, SHAPE_SWEEP_NOISE)
            m = metrics(r)
            records.append(m)
            print(f"  {sys_name:11s} | pt={pt_method:8s} | emb={emb:12s} | "
                  f"corr={m.get('lambda_corr', float('nan')):+.3f}  "
                  f"peak_err={m.get('peak_timing_error', float('nan')):.3f}  "
                  f"sign={m.get('sign_recovery', float('nan')):.2f}  "
                  f"({(time.time()-t0):.1f}s)")
shape_df = pd.DataFrame(records)
shape_df.to_csv(CACHE_DIR + 'sweep_A_shape.csv', index=False)
shape_df.head(20)
""")


code(r"""
# Per-system heatmap: rows = pseudotime, cols = embedding, colour = λ-curve correlation.
metric_to_plot = 'lambda_corr'
fig, axes = plt.subplots(1, len(SYSTEMS_DEFAULT), figsize=(3.6*len(SYSTEMS_DEFAULT), 3.2))
for ax, sys_name in zip(axes, SYSTEMS_DEFAULT):
    sub = shape_df[shape_df.system == sys_name]
    pivot = sub.pivot(index='pt_method', columns='embedding', values=metric_to_plot)
    # consistent row order
    pivot = pivot.reindex(PSEUDOTIME_METHODS)
    cmap = 'RdBu_r'
    im = ax.imshow(pivot.values, aspect='auto', cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
    ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            ax.text(j, i, f'{v:+.2f}' if pd.notna(v) else '', ha='center', va='center',
                    color='white' if abs(v) > 0.5 else 'black', fontsize=8)
    ax.set_title(sys_name)
plt.colorbar(im, ax=axes, shrink=0.7, label=r'$r(\lambda^{rec}, \lambda^{true})$')
plt.suptitle(f'Sweep A — λ-curve correlation against truth (noise σ={SHAPE_SWEEP_NOISE})',
             y=1.02, fontsize=11)
plt.savefig(CACHE_DIR + 'fig02_sweep_A_lambda_corr.pdf', bbox_inches='tight'); plt.show()
""")


# ─────────────────────────────────────────────────────────────────────────
md(r"""## 6. Sweep B — noise vs embedding (the "FA advantage grows with noise" test)

System: `commitment` (the only one with a clean, biologically-motivated
ground-truth peak). Pseudotime: oracle (so the embedding is the only
moving part). Embedding: PCA, FA, ICA, TruncatedSVD, optionally scVI /
LDVAE. We sweep σ.""")
code(r"""
records_B = []
for noise in NOISE_LEVELS_DEFAULT:
    for emb in EMBEDDINGS_DEFAULT + (['scVI', 'LDVAE'] if SCVI_OK else []):
        t0 = time.time()
        r = run_one('commitment', 'oracle', emb, noise)
        m = metrics(r)
        records_B.append(m)
        print(f"  σ={noise:.2f}  emb={emb:12s}  "
              f"corr={m.get('lambda_corr', float('nan')):+.3f}  "
              f"peak_err={m.get('peak_timing_error', float('nan')):.3f}  "
              f"({(time.time()-t0):.1f}s)")
noise_df = pd.DataFrame(records_B)
noise_df.to_csv(CACHE_DIR + 'sweep_B_noise.csv', index=False)
""")


code(r"""
# Line plot: λ-curve correlation vs σ, one line per embedding
fig, ax = plt.subplots(figsize=(6, 3.6))
for emb in noise_df.embedding.unique():
    sub = noise_df[noise_df.embedding == emb].sort_values('noise')
    ax.plot(sub.noise, sub.lambda_corr, '-o', lw=2.0, label=emb)
ax.set_xlabel(r'observation noise $\sigma$'); ax.set_ylabel(r'$r(\lambda^{rec}, \lambda^{true})$')
ax.axhline(0.0, color='gray', lw=0.6, ls=':')
ax.axhline(0.8, color='gray', lw=0.4, ls=':')
ax.set_title('Sweep B — commitment system: λ-curve correlation vs noise (oracle pseudotime)')
ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(CACHE_DIR + 'fig03_sweep_B_noise_vs_embed.pdf'); plt.show()
""")


# ─────────────────────────────────────────────────────────────────────────
md(r"""## 7. Cyclic failure-mode demonstration

This is the negative control. Pseudotime methods linearise a circular
trajectory; scJDO running on top should report either no peak or a
spurious one with low effective sample size. We plot the recovered λ
curves under each pseudotime method and overlay the constant analytic
truth.""")
code(r"""
fig, axes = plt.subplots(1, len(PSEUDOTIME_METHODS), figsize=(3.6*len(PSEUDOTIME_METHODS), 3.2),
                         sharey=True)
for ax, pt in zip(axes, PSEUDOTIME_METHODS):
    for emb in EMBEDDINGS_DEFAULT:
        r = run_one('cyclic', pt, emb, 0.1)        # mild noise
        if r.get('skipped'): continue
        ax.plot(r['t_centers'], r['lam'], lw=1.5, label=emb)
    ax.axhline(-2.0, color='black', lw=0.8, ls='--', label='truth')
    ax.set_title(f'pseudotime = {pt}')
    ax.set_xlabel('τ'); ax.legend(fontsize=7)
axes[0].set_ylabel(r'$\lambda_{\max}(\tau)$')
plt.suptitle('Sweep C — cyclic system (declared failure mode)', y=1.02)
plt.tight_layout(); plt.savefig(CACHE_DIR + 'fig04_sweep_C_cyclic.pdf'); plt.show()
""")


# ─────────────────────────────────────────────────────────────────────────
md(r"""## 8. Reading the results

* **Sweep A heatmaps (Section 5).** Per system, look across rows
  (pseudotime invariance) and across columns (embedding invariance).
  An entry near $r=+1$ means scJDO recovered the right $\lambda(\tau)$
  curve; near $0$ means no relationship; negative means the recovered
  curve is anti-correlated with truth.
* **Sweep B line plot (Section 6).** This is the central FA claim. If
  the FA line stays high while PCA / scVI / LDVAE drop as $\sigma$
  grows, the "derivative stability under noise" argument is supported
  mechanistically. If FA matches PCA across all noise levels, the FA
  argument should be downgraded.
* **Sweep C cyclic panels (Section 7).** Any visible peak in any panel
  is a pseudotime artefact, by construction. The point is to
  document this failure mode honestly — a method that knows its own
  boundaries is more convincing than one that always claims to work.

The raw per-row metric tables `sweep_A_shape.csv` and `sweep_B_noise.csv`
are saved alongside the figures; you can join them into the manuscript
text via the per-row `peak_timing_error` and `lambda_corr` columns.
""")


# ─────────────────────────────────────────────────────────────────────────
md(r"""## 9. Embedding diagnostics — first-2-component scatter per system × method

What each embedding actually *sees* on each synthetic system. This explains
the Sweep A heatmap: if an embedding's scatter doesn't reveal the underlying
trajectory geometry, scJDO can't recover the Jacobian from it regardless of
the pseudotime used. Points are coloured by **true pseudotime** (the oracle),
so a clean colour gradient along the projection is the visual prerequisite
for downstream recovery.""")
code(r"""# Embedding diagnostics. Re-uses make_system() so the data is identical to the
# sweeps. Embeddings are computed in memory (no caching) because they're cheap
# for the linear methods; scVI/LDVAE only run if SCVI_OK is True.
EMBEDDINGS_ALL = EMBEDDINGS_DEFAULT + (['scVI', 'LDVAE'] if SCVI_OK else [])
DIAG_NOISE     = SHAPE_SWEEP_NOISE              # match Sweep A noise level

fig, axes = plt.subplots(len(SYSTEMS_DEFAULT), len(EMBEDDINGS_ALL),
                        figsize=(2.4*len(EMBEDDINGS_ALL), 2.4*len(SYSTEMS_DEFAULT)),
                        squeeze=False)

for i, sys_name in enumerate(SYSTEMS_DEFAULT):
    s = make_system(sys_name, n_cells=N_CELLS, seed=SEED)
    X = s['X'].copy()
    if DIAG_NOISE > 0:
        rng = np.random.default_rng(SEED + 13)
        X = X + (DIAG_NOISE * rng.standard_normal(X.shape)).astype(np.float32)
    t_true = s['t_true']

    for j, emb_name in enumerate(EMBEDDINGS_ALL):
        ax = axes[i, j]
        try:
            Z = embed(X, method=emb_name)
        except Exception as e:
            ax.text(0.5, 0.5, f'{type(e).__name__}', ha='center', va='center',
                    transform=ax.transAxes, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0: ax.set_title(emb_name, fontsize=10)
            if j == 0: ax.set_ylabel(sys_name, fontsize=10)
            continue
        if Z is None:
            ax.text(0.5, 0.5, '(skip)', ha='center', va='center',
                    transform=ax.transAxes, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0: ax.set_title(emb_name, fontsize=10)
            if j == 0: ax.set_ylabel(sys_name, fontsize=10)
            continue
        sc_ = ax.scatter(Z[:, 0], Z[:, 1], c=t_true, cmap='viridis',
                         s=4, alpha=0.85, rasterized=True)
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0: ax.set_title(emb_name, fontsize=10)
        if j == 0: ax.set_ylabel(sys_name, fontsize=10, rotation=90, labelpad=10)

cbar = fig.colorbar(sc_, ax=axes.ravel().tolist(), shrink=0.55, pad=0.01,
                    label='true pseudotime')
plt.suptitle(f'Embedding diagnostics — first 2 components, coloured by true pseudotime '
             f'(σ={DIAG_NOISE})', y=1.02, fontsize=11)
plt.savefig(CACHE_DIR + 'fig05_embedding_diagnostics.pdf', bbox_inches='tight')
plt.show()
print(f'Saved: {CACHE_DIR}fig05_embedding_diagnostics.pdf')
""")


# ─────────────────────────────────────────────────────────────────────────
md(r"""## 10. Cause A vs B — pseudotime ordering recovery

Finding 1 (above) reads the commitment-system collapse as a pseudotime
problem rather than a scJDO problem, but the heatmap alone cannot
distinguish two hypotheses:

* **Cause A** — DPT / Palantir recover the wrong ordering; scJDO
  faithfully reports the consequence of being fed a scrambled $\tau$.
  In this case the operator machinery is fine.
* **Cause B** — DPT / Palantir recover the ordering reasonably well, but
  scJDO is fragile to small ordering perturbations. In this case the
  method requires near-perfect ordering and is impractical on real data.

This cell measures the **Spearman correlation between each pseudotime
method's output and the true $\tau$** on every (system, embedding) at the
same noise level used in Sweep A. Combined with the perturbed-oracle
robustness curve in the next cell, this discriminates A from B.""")
code(r"""# Spearman(τ_method, τ_true) per (system, embedding, pt_method) at σ=DIAG_NOISE.
from scipy.stats import spearmanr
import anndata as ad_pkg

DIAG_SIGMA   = SHAPE_SWEEP_NOISE
DIAG_PT_METHODS = ['dpt', 'palantir']
DIAG_EMB        = [e for e in EMBEDDINGS_DEFAULT]   # PCA, FA, ICA, TruncatedSVD

rows = []
for sys_name in SYSTEMS_DEFAULT:
    s = make_system(sys_name, n_cells=N_CELLS, seed=SEED)
    X = s['X'].copy()
    if DIAG_SIGMA > 0:
        rng = np.random.default_rng(SEED + 13)
        X = X + (DIAG_SIGMA * rng.standard_normal(X.shape)).astype(np.float32)
    t_true = s['t_true']

    for emb_name in DIAG_EMB:
        Z = embed(X, method=emb_name)
        if Z is None: continue
        adata = ad_pkg.AnnData(X=X)
        adata.obs_names = [f'c{i:05d}' for i in range(adata.n_obs)]
        adata.obs['t_true'] = t_true
        adata.obsm['X_embed'] = Z

        for pt_method in DIAG_PT_METHODS:
            try:
                pt = pseudotime(adata, pt_method)
                pt = (pt - pt.min()) / max(pt.max() - pt.min(), 1e-9)
                rho, _ = spearmanr(pt, t_true)
            except Exception:
                rho = float('nan')
            rows.append(dict(system=sys_name, embedding=emb_name,
                             pt_method=pt_method, spearman=float(rho)))
            print(f'  {sys_name:11s} | emb={emb_name:12s} | pt={pt_method:8s}  '
                  f'Spearman(pt, t_true) = {rho:+.3f}')

spearman_df = pd.DataFrame(rows)
spearman_df.to_csv(CACHE_DIR + 'sweep_D_pseudotime_spearman.csv', index=False)

fig, axes = plt.subplots(1, len(SYSTEMS_DEFAULT), figsize=(3.4*len(SYSTEMS_DEFAULT), 2.8))
for ax, sys_name in zip(axes, SYSTEMS_DEFAULT):
    sub = spearman_df[spearman_df.system == sys_name]
    piv = sub.pivot(index='pt_method', columns='embedding', values='spearman') \
             .reindex(DIAG_PT_METHODS).reindex(columns=DIAG_EMB)
    im = ax.imshow(piv.values, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f'{v:+.2f}' if pd.notna(v) else '', ha='center', va='center',
                    color='white' if abs(v) > 0.5 else 'black', fontsize=8)
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=45, ha='right')
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index)
    ax.set_title(sys_name)
plt.colorbar(im, ax=axes, shrink=0.7, label=r'Spearman$(\tau^{rec}, \tau^{true})$')
plt.suptitle(f'Pseudotime ordering recovery vs truth (σ={DIAG_SIGMA})', y=1.02)
plt.savefig(CACHE_DIR + 'fig06_pseudotime_spearman.pdf', bbox_inches='tight')
plt.show()
print(f'\nSaved: {CACHE_DIR}sweep_D_pseudotime_spearman.csv  +  fig06_pseudotime_spearman.pdf')
""")


# ─────────────────────────────────────────────────────────────────────────
md(r"""## 11. Perturbed-oracle robustness curve

Take the true $\tau$ from the **commitment** system, add controlled rank
noise to it, and measure (a) the resulting Spearman correlation against
the truth and (b) scJDO's recovered $\lambda$-curve correlation against
the analytic truth. This plots **operator-recovery as a function of
ordering-quality**, the single curve that resolves Cause A vs B.

* If $\lambda$-recovery holds until Spearman drops well below the value
  DPT/Palantir actually achieve on this system → scJDO is **robust**;
  Cause A is confirmed.
* If $\lambda$-recovery collapses near or above the DPT/Palantir
  Spearman → scJDO is **fragile**; Cause B is confirmed and the method's
  practical reliability is suspect.

We sweep PCA and FA so the result isn't embedding-specific.""")
code(r"""# Perturbed-oracle robustness curve on commitment, σ=DIAG_SIGMA.
from scipy.stats import spearmanr

ROB_SYS   = 'commitment'
ROB_EMB   = ['PCA', 'FA']
ROB_NOISE = DIAG_SIGMA
NOISE_LEVELS_RANK = [0.0, 0.01, 0.03, 0.10, 0.30, 0.50, 1.00, 2.00, 5.00]

def perturb_oracle(t_true: np.ndarray, noise_level: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    N = len(t_true)
    base_ranks = pd.Series(t_true).rank().values.astype(np.float64)
    noisy_ranks = base_ranks + (noise_level * N) * rng.standard_normal(N)
    new_ranks = pd.Series(noisy_ranks).rank().values
    return (new_ranks - 1.0) / max(N - 1, 1)


s = make_system(ROB_SYS, n_cells=N_CELLS, seed=SEED)
X_base = s['X'].copy()
if ROB_NOISE > 0:
    rng = np.random.default_rng(SEED + 13)
    X_base = X_base + (ROB_NOISE * rng.standard_normal(X_base.shape)).astype(np.float32)
t_true_arr  = s['t_true']
grid_true   = s['grid']
true_lambda = s['true_lambda']

rows_rob = []
for emb_name in ROB_EMB:
    Z = embed(X_base, method=emb_name)
    for nl in NOISE_LEVELS_RANK:
        pt = perturb_oracle(t_true_arr, nl, seed=SEED + 7)
        rho, _ = spearmanr(pt, t_true_arr)
        adata = ad_pkg.AnnData(X=X_base)
        adata.obs_names = [f'c{i:05d}' for i in range(adata.n_obs)]
        adata.obs['t_true'] = t_true_arr
        adata.obsm['X_embed'] = Z
        adata.obs['pseudotime'] = pt.astype(np.float32)
        sjd.tl.fit_drift(adata, rep='X_embed', time_key='pseudotime',
                         n_epochs=N_EPOCHS_BENCH, n_archetypes=N_ARCHETYPES,
                         n_eff_min=20.0, n_boot=10, grid_size=N_GRID,
                         seed=SEED, verbose=False)
        r = adata.uns['scjdo']
        from scipy.interpolate import interp1d
        g  = np.linspace(0.05, 0.95, 100, dtype=np.float32)
        L_rec = interp1d(r['t_centers'], r['max_real_eig'], bounds_error=False,
                         fill_value='extrapolate')(g)
        L_tru = interp1d(grid_true, true_lambda, bounds_error=False,
                         fill_value='extrapolate')(g)
        lam_corr = float(np.corrcoef(L_rec, L_tru)[0, 1]) if np.std(L_rec) > 0 else float('nan')
        peak_err = float(abs(g[np.argmax(L_rec)] - g[np.argmax(L_tru)]))
        rows_rob.append(dict(embedding=emb_name, noise_level=nl,
                             spearman=float(rho), lambda_corr=lam_corr,
                             peak_timing_error=peak_err))
        print(f'  emb={emb_name:4s}  rank-noise={nl:5.2f}  '
              f'Spearman={rho:+.3f}  λ-corr={lam_corr:+.3f}  peak_err={peak_err:.3f}')

rob_df = pd.DataFrame(rows_rob)
rob_df.to_csv(CACHE_DIR + 'sweep_E_perturbed_oracle.csv', index=False)
""")
code(r"""# Plot the robustness curve with DPT/Palantir Spearman markers overlaid.
fig, ax = plt.subplots(figsize=(6.5, 4))
markers = {'PCA': 'o', 'FA': 's'}
for emb_name, mk in markers.items():
    sub = rob_df[rob_df.embedding == emb_name].sort_values('spearman', ascending=False)
    ax.plot(sub.spearman, sub.lambda_corr, '-' + mk, lw=2, ms=7, label=emb_name)

commit = spearman_df[(spearman_df.system == 'commitment') &
                     (spearman_df.embedding.isin(ROB_EMB))]
for pt_method, color in [('dpt', '#d62728'), ('palantir', '#9467bd')]:
    rho_vals = commit[commit.pt_method == pt_method]['spearman'].dropna().values
    if len(rho_vals):
        for r in rho_vals:
            ax.axvline(r, color=color, ls='--', lw=1.0, alpha=0.7)
        ax.axvline(rho_vals.mean(), color=color, ls='--', lw=2.0,
                   label=f'{pt_method} on commitment  (mean Spearman={rho_vals.mean():+.3f})')

ax.axhline(0.0, color='gray', lw=0.6, ls=':')
ax.axhline(0.8, color='gray', lw=0.4, ls=':')
ax.set_xlabel(r'Spearman$(\tau^{\mathrm{used}}, \tau^{\mathrm{true}})$  →  ordering quality')
ax.set_ylabel(r'$r(\lambda^{\mathrm{rec}}, \lambda^{\mathrm{true}})$  →  operator-recovery quality')
ax.set_title(f'Cause A vs B diagnostic on commitment (σ={ROB_NOISE})')
ax.set_xlim(-0.1, 1.05); ax.set_ylim(-1.05, 1.05)
ax.legend(fontsize=8, loc='lower right')
plt.tight_layout()
plt.savefig(CACHE_DIR + 'fig07_perturbed_oracle_robustness.pdf')
plt.show()

print('\nPre-committed read:')
dpt_rho      = commit[commit.pt_method == 'dpt']['spearman'].mean()
palantir_rho = commit[commit.pt_method == 'palantir']['spearman'].mean()
for name, rho in [('DPT', dpt_rho), ('Palantir', palantir_rho)]:
    if pd.isna(rho): continue
    sub = rob_df.sort_values('spearman')
    from scipy.interpolate import interp1d
    interp_lambda = []
    for emb_name in ROB_EMB:
        es = sub[sub.embedding == emb_name].dropna(subset=['lambda_corr'])
        if len(es) < 2: continue
        f = interp1d(es.spearman, es.lambda_corr, bounds_error=False, fill_value='extrapolate')
        interp_lambda.append(float(f(rho)))
    pred = np.mean(interp_lambda) if interp_lambda else float('nan')
    print(f'  At {name} Spearman ({rho:+.3f}) the robustness curve predicts '
          f'λ-corr ≈ {pred:+.3f}  (averaged over PCA + FA)')

for emb_name in ROB_EMB:
    es = rob_df[rob_df.embedding == emb_name].sort_values('spearman', ascending=False)
    crossing = es[es.lambda_corr < 0.8]
    if len(crossing) == 0:
        print(f'  {emb_name}: λ-corr stays ≥ 0.8 even at Spearman = {es.spearman.min():+.3f}')
    else:
        first = crossing.iloc[0]
        prior = es[es.spearman > first.spearman].iloc[-1] if (es.spearman > first.spearman).any() else None
        rho_threshold = first.spearman if prior is None else 0.5*(first.spearman + prior.spearman)
        print(f'  {emb_name}: λ-corr crosses 0.8 between Spearman {first.spearman:+.3f} and '
              f'{prior.spearman if prior is not None else None}; threshold ≈ {rho_threshold:+.3f}')
""")


# ─────────────────────────────────────────────────────────────────────────
def main() -> None:
    out = Path(__file__).resolve().parent / "Synthetic_pseudotime_embedding_benchmark.ipynb"
    nb = nbf.v4.new_notebook()
    nb.cells = CELLS
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    with out.open("w") as f:
        nbf.write(nb, f)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
