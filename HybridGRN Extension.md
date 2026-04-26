# HybridGRN Extension for scQDiff

**Status: Experimental opt-in extension.** The default scQDiff workflow (latent Hybrid Drift → latent Jacobian → latent archetypes) is unchanged.

---

## Overview

The HybridGRN extension adds a gene-space GRN extraction path on top of the existing latent dynamics model. It is designed to be used after the standard scQDiff workflow is complete and validated.

The pipeline is:

```
Gene expression (G)
        │
        ▼  encoder E
Latent space (D)
        │
        ▼  DriftField f
Latent drift  J_z = ∂f/∂z   (D×D)
        │
        ▼  chain rule  J_x ≈ D_z · J_z · E_x
Gene-space operator  J_x   (G×n_tf)
        │
        ▼  SparseGRNRefiner
Sparse GRN  K_x   (n_tf×G, per pseudotime bin)
        │
        ▼  SVD archetypes
Regulatory archetypes  A_k   (rank × n_tf × G)
```

---

## Why this is an extension, not the default

The existing scQDiff design is already organised around a latent-space Hybrid Drift default. The data loader defaults to `X_pca`, the model warns that full Jacobians become risky once dimension gets large, and the present design philosophy is "estimate dynamics in a reduced state space first."

A hybrid GRN layer adds extra assumptions that are not part of the core model: an encoder/decoder map, a gene-space pullback, sparsity choices, TF-target constraints, and a GRN extraction objective. Those are valuable, but they create more failure modes than the current latent operator pipeline.

The HybridGRN path should only be promoted to default after it clears three tests:

| Criterion | Description |
|---|---|
| **Mathematical stability** | The pulled-back gene operator is consistent across seeds, nearby latent neighbourhoods, and reasonable decoder choices. Use `model.validate_consistency()`. |
| **Biological credibility** | Recovered regulators/modules outperform or at least match baselines (CellOracle, SCENIC) on known benchmark systems (Paul15, LARRY). |
| **Usability** | The added model does not make ordinary scQDiff workflows much harder to train or interpret. |

---

## Mathematical objects (kept strictly distinct)

| Symbol | Shape | Description |
|---|---|---|
| `J_z` | `(B, D, D)` | Latent Jacobian `∂f/∂z` from DriftField |
| `J_x` | `(B, G, n_tf)` | Induced gene-space operator from encoder/decoder pullback |
| `K_x` | `(T, n_tf, G)` | Sparse, prior-constrained GRN approximation to `J_x` |
| `A_k` | `(rank, n_tf, G)` | Archetypes extracted from `K_x` over pseudotime |

The chain rule identity is:

```
J_x(x, t) ≈ D_z · J_z(z, t) · E_x
```

where `D_z = ∂D/∂z` (decoder Jacobian, constant for linear decoder) and `E_x = ∂E/∂x` (encoder Jacobian).

---

## Module structure

```
scqdiff/
  models/
    representation.py     # PCARep, LDVAERep, VegaRep backends
    hybrid_grn.py         # HybridGRNConfig, HybridGRNModel, HybridGRNResult
  grn/
    pullback.py           # pullback_gene_operator, binned_pullback
    refine.py             # SparseGRNRefiner, loss terms
    priors.py             # TF masks, sign masks, ATAC/motif priors
    archetypes.py         # grn_modes, GRNArchetypeResult
    perturb.py            # knockout_score, control_energy_score
  io/
    anndata_hybrid.py     # tensors_from_anndata_hybrid, compute_bin_means
  pipeline/
    train_hybrid_grn.py   # train_hybrid_grn_from_anndata (staged)
  analysis/
    grn_scores.py         # regulator_centrality, branch_control_score
```

---

## Representation backends

The decoder choice is the most consequential design decision. Three backends are supported:

| Backend | Decoder | When to use |
|---|---|---|
| `PCARep` | Linear (PCA loadings) | Debugging, fast prototyping, when PCA is already computed |
| `LDVAERep` | Linear `W ∈ R^{G×D}` (no bias) | **Default.** Best balance of interpretability and capacity. |
| `VegaRep` | Masked linear (TF/pathway priors) | When you have strong module priors (e.g. regulon assignments) |

**Why LDVAERep is the default:** The linear decoder makes `D_z = W` constant, so the pullback `J_x = W J_z E_x` has no curvature terms and is numerically stable even for moderate G. The deep encoder still captures nonlinear structure in the latent space.

**Why not a generic nonlinear decoder for v1:** Standard scVI's latent space is explicitly less interpretable. Nonlinear decoders introduce curvature terms in the pullback that require second-order autograd and are numerically unstable for large G.

---

## Quickstart

```python
import warnings
import torch
from scqdiff.models.drift import DriftField, DriftConfig
from scqdiff.models.representation import RepresentationConfig, LDVAERep
from scqdiff.models.hybrid_grn import HybridGRNConfig, HybridGRNModel
from scqdiff.grn.refine import GRNRefinerConfig
from scqdiff.pipeline.train_hybrid_grn import (
    HybridGRNTrainConfig,
    train_hybrid_grn_from_anndata,
)

# --- Option A: Full pipeline from AnnData ---
model, result = train_hybrid_grn_from_anndata(
    adata,
    cfg=HybridGRNConfig(
        drift_cfg=DriftConfig(dim=50),
        rep_cfg=RepresentationConfig(backend="ldvae", n_latent=50, n_genes=2000),
        refiner_cfg=GRNRefinerConfig(lambda_sparse=1e-3, n_steps=200),
        grn_rank=10,
        n_time_bins=20,
        pullback_mode="tf_restricted",
    ),
    tf_names=my_tf_list,
    pseudotime_key="pseudotime",
    verbose=True,
)

# Access results
print(result.Kx.shape)          # (20, n_tf, 2000)
print(result.archetypes.archetypes.shape)  # (10, n_tf, 2000)

# --- Option B: Manual pipeline (recommended for power users) ---
# Step 1: pretrain representation
rep = LDVAERep(RepresentationConfig(backend="ldvae", n_latent=50, n_genes=2000))
# ... train rep on x_gene ...

# Step 2: train drift on latent z
drift = DriftField(DriftConfig(dim=50))
# ... train drift on z = rep.encode(x_gene) ...

# Step 3: build HybridGRN and run extraction
with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # suppress experimental warning if desired
    model = HybridGRNModel(
        cfg, rep=rep, drift=drift,
        tf_index=tf_index,
        tf_mask=tf_mask,
    )

result = model.run(X=x_gene, T=pseudotime, verbose=True)
```

---

## GRN extraction details

### Pullback modes

| Mode | Description | When to use |
|---|---|---|
| `"linear"` | Constant Jacobians (PCARep/LDVAERep exact) | Always for linear backends |
| `"autograd"` | Full autograd Jacobian for encoder | Small G (< 500) |
| `"projected"` | Random-projection approximation | Large G, memory-constrained |
| `"tf_restricted"` | Only TF columns of J_x | **Recommended default.** G/n_tf memory reduction. |

### Sparse GRN refinement

The refiner solves:

```
L_grn = ||K_x - J_x||_F²
      + λ_sparse  · ||K_x||_1
      + λ_prior   · Ω_prior(K_x)
      + λ_local   · ||K_x x_tf - dx_obs||²
      + λ_temporal· ||K_t - K_{t-1}||_F²
      + λ_stability · Σ relu(λ_max(K_x) - clip)
```

Key design choice: `K_x` has shape `(T, n_tf, G)` (TF→gene), not `(T, G, G)`. This is more identifiable (n_tf << G), biologically legible, and reduces parameters by G/n_tf.

### Archetype decomposition

```
K_x[t] ≈ K_mean + Σ_k  c_k(t) · A_k
```

where `A_k ∈ R^{n_tf × G}` are the archetypes and `c_k(t)` are their time-dependent activations. SVD is used (not NMF) to preserve signed regulation.

---

## Validation checklist before biological interpretation

Before trusting GRN conclusions, run:

```python
# 1. Mathematical stability
stability = model.validate_consistency(x_gene, pseudotime, n_seeds=3)
print(stability)
# {"mean_cosine_similarity": 0.95, "std_cosine_similarity": 0.02, "is_stable": True}

# 2. Check top regulators are not wildly unstable across seeds
# (run train_hybrid_grn_from_anndata with seed=0,1,2 and compare top TFs)

# 3. Check archetype variance explained
from scqdiff.grn.archetypes import archetype_summary
summaries = archetype_summary(result.archetypes, tf_names, gene_names)
for s in summaries:
    print(f"Archetype {s['rank']}: {s['variance_explained']:.3f} var, "
          f"top TFs: {s['top_tfs']}")
```

---

## Promotion criteria

The HybridGRN path should only be considered for default status after:

1. **Mathematical stability:** `validate_consistency()` returns `mean_cosine_similarity > 0.9` on at least two real datasets.
2. **Biological credibility:** Top TFs from `K_x` archetypes match known regulators on Paul15 (haematopoiesis) or LARRY (lineage tracing) benchmarks.
3. **Usability:** Training time increase over standard scQDiff is less than 3× for typical datasets (N=5000, G=2000).
