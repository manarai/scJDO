# scIDiff: Learning Single-Cell Regulatory Dynamics with Hybrid Drift Fields

**scIDiff** (single-cell inverse Diffusion) is a computational framework for learning continuous-time cellular dynamics directly from single-cell data. By combining **score-based diffusion models**, **Neural ODE residual learning**, and **RNA velocity biological priors**, scIDiff infers stochastic drift fields that describe how cells evolve through transcriptional state space.

The framework supports two complementary approaches:

1. **Hybrid Drift Field** (default) — Combines score networks, Neural ODE corrections, and velocity guidance for control analysis and general trajectory modeling
2. **Schrödinger Bridge** (optional) — Optimal transport between distributions for aging and rejuvenation analysis

---

## Key Features

-  **Hybrid drift field** combining score-based diffusion + Neural ODE + RNA velocity
-  **Schrödinger Bridge** for optimal transport between cell states (e.g., young ↔ old)
-  **RNA velocity integration** as biological reference drift
-  **Temporal Jacobian computation** for gene regulatory network inference
-  **Archetype decomposition** revealing regulatory modules
-  **Forward and reverse dynamics** for irreversibility quantification
-  **Flexible configuration** for different biological questions

---

## Mathematical Foundations

### Hybrid Drift Field Model (Default)

scIDiff models cellular dynamics as a controlled stochastic process:

$$
dX_t = f(X_t,t)\,dt + \sqrt{2\beta}\,dW_t
$$

where the **drift field** is decomposed as:

$$
f(x,t) = \underbrace{\beta \cdot \text{score}_\theta(x,t)}_{\text{diffusion component}} + \underbrace{\text{residual}_\theta(x,t)}_{\text{Neural ODE component}} + \underbrace{b(x,t)}_{\text{velocity prior}}
$$

**Components**:
- **Score network** $\text{score}_\theta(x,t)$ — Learns denoising score function $\nabla_x \log \rho_t(x)$ for generative modeling
- **Residual network** $\text{residual}_\theta(x,t)$ — Neural ODE-like correction for dynamics learning
- **Velocity prior** $b(x,t)$ — RNA velocity as biological reference (optional)

This **hybrid architecture** provides:
- **Generative capability** (score network)
- **Dynamics learning** (residual network)
- **Biological grounding** (velocity prior)
- **Explicit Jacobian** for control analysis

### RNA Velocity as Reference Drift

When RNA velocity is available, it provides a biologically grounded baseline:

$$
b(x,t) = \lambda \cdot g(t) \cdot w(x) \cdot \hat v(x)
$$

where:
- $\hat v(x)$ — RNA velocity interpolated via soft k-NN
- $w(x) = \text{conf}(x)^p$ — Confidence-based gating
- $g(t)$ — Time-dependent schedule (peaks at $t=0.5$ for "mid" mode)
- $\lambda$ — Magnitude scaling

**Velocity interpolation** (soft k-NN):
$$
\hat v(x) = \sum_{i=1}^k w_i \cdot v_i
\quad \text{where} \quad
w_i = \frac{\exp(-d_i/\tau)}{\sum_j \exp(-d_j/\tau)}
$$

### Training Objective

The model is trained by minimizing:

$$
\mathcal{L} = \mathcal{L}_{\text{DSM}} + \alpha_1 \mathcal{L}_{\text{control}} + \alpha_2 \mathcal{L}_{\text{FP}} + \alpha_3 \mathcal{L}_{\text{smooth}}
$$

where:
- $\mathcal{L}_{\text{DSM}}$ — Denoising score matching loss
- $\mathcal{L}_{\text{control}} = \mathbb{E}[\|u_\theta\|^2]$ — Control energy regularization
- $\mathcal{L}_{\text{FP}}$ — Fokker-Planck residual loss
- $\mathcal{L}_{\text{smooth}}$ — Laplacian smoothness regularization

---

## Schrödinger Bridge for Aging Analysis (NEW)

For applications with **natural endpoint distributions** (e.g., young → old cells), scIDiff now includes a **Schrödinger Bridge** module that learns **coupled forward and backward drifts** for optimal transport.

### When to Use Schrödinger Bridge

**Use Schrödinger Bridge for**:
- Aging and rejuvenation modeling
- Development and dedifferentiation
- Perturbation response with clear endpoints
- Optimal transport between conditions

**Use Hybrid Drift Field for**:
- Control factor discovery (Yamanaka factors, etc.)
- General trajectory modeling
- Jacobian-based regulatory analysis
- Flexible dynamics without endpoint constraints

### Mathematical Formulation

Given two distributions $\rho_0$ (source, e.g., young) and $\rho_1$ (target, e.g., old), the Schrödinger Bridge learns:

**Forward drift** (aging: $\rho_0 \to \rho_1$):
$$
dX_t = b_f(X_t, t)\,dt + \sqrt{2\beta}\,dW_t
$$

**Backward drift** (rejuvenation: $\rho_1 \to \rho_0$):
$$
dX_t = b_b(X_t, t)\,dt + \sqrt{2\beta}\,d\bar{W}_t
$$

These drifts are **coupled** through entropic optimal transport, ensuring:
- ✅ Forward process reaches $\rho_1$ from $\rho_0$
- ✅ Backward process reaches $\rho_0$ from $\rho_1$
- ✅ Minimal deviation from reference (Brownian motion)
- ✅ Consistency between forward and backward

### Training Algorithm

The Schrödinger Bridge is trained iteratively:

1. **Compute OT plan** between $\rho_0$ and $\rho_1$ using Sinkhorn algorithm
2. **Train forward drift** using score matching with OT guidance
3. **Train backward drift** using score matching with OT guidance
4. **Repeat** until convergence

---

## Installation

```bash
# Create conda environment
conda create -n scidiff python=3.10
conda activate scidiff

# Clone repository
git clone https://github.com/yourusername/scIDiff.git
cd scIDiff

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

**Dependencies**:
- Python ≥ 3.10
- PyTorch ≥ 2.0
- AnnData, scanpy
- numpy, scipy, matplotlib

---

## Quick Start

### 1. Hybrid Drift Field (General Use)

**Command Line**:
```bash
python -m scqdiff.pipeline.train_from_anndata \
    --h5ad your_data.h5ad \
    --use-velocity-prior \
    --normalize-velocity \
    --vel-layer velocity \
    --ptime-key pseudotime \
    --epochs 200 \
    --out-prefix my_model
```

**Python API**:
```python
import torch
import anndata as ad
from scqdiff.io.anndata import tensors_from_anndata
from scqdiff.models.drift import DriftField, DriftConfig

# Load data
adata = ad.read_h5ad("your_data.h5ad")
X, V, T = tensors_from_anndata(
    adata,
    vel_layer="velocity",
    pseudotime_key="pseudotime"
)

# Configure model with velocity prior
cfg = DriftConfig(
    dim=X.shape[1],
    beta=0.1,
    use_velocity_prior=True,
    vel_scale=1.0,
    vel_k=32,
    vel_time_mode="mid"
)

# Create and train model
model = DriftField(cfg, X_ref=X, V_ref=V)

# Compute drift and Jacobian
drift = model(X, T)
J = model.jacobian(X[:10], T[:10])
```

### 2. Schrödinger Bridge (Aging/Transport)

**Command Line**:
```bash
python -m scqdiff.pipeline.train_bridge \
    --h5ad aging_data.h5ad \
    --source-key age_group \
    --source-value young \
    --target-value old \
    --n-iterations 10 \
    --out-prefix aging_bridge
```

**Python API**:
```python
from scqdiff.models.schrodinger_bridge import SchrodingerBridge, SchrodingerBridgeConfig
from scqdiff.pipeline.train_bridge import train_bridge_from_anndata

# Load aging data
adata = sc.read_h5ad('aging_data.h5ad')

# Train bridge
cfg = SchrodingerBridgeConfig(
    dim=adata.n_vars,
    hidden=256,
    depth=4,
    beta=0.1,
    epsilon=0.1  # OT regularization
)

model, history = train_bridge_from_anndata(
    adata,
    source_key='age_group',
    source_value='young',
    target_value='old',
    cfg=cfg,
    n_iterations=10
)

# Predict aging trajectory
young_cell = torch.tensor(adata[0].X, dtype=torch.float32)
aging_trajectory = model.forward_integrate(young_cell, steps=100)

# Predict rejuvenation trajectory (TRUE REVERSE!)
old_cell = torch.tensor(adata[-1].X, dtype=torch.float32)
rejuvenation_trajectory = model.backward_integrate(old_cell, steps=100)

# Identify rejuvenation targets
gene_changes = (rejuvenation_trajectory[:, -1] - rejuvenation_trajectory[:, 0]).abs().mean(dim=0)
top_genes = torch.argsort(gene_changes, descending=True)[:20]
```

---

## Temporal Jacobians and Control Analysis

The temporal Jacobian captures local gene-gene regulatory influence:

$$
J(t) = \frac{\partial f}{\partial x}(t)
$$

where $J_{ij}(t) = \frac{\partial f_i}{\partial x_j}$ quantifies how gene $j$ influences the rate of change of gene $i$ at time $t$.

**Compute Jacobian**:
```python
# Using hybrid drift field
J = model.jacobian(X, T)  # (B, D, D)

# Eigenvalue decomposition for control analysis
eigenvalues, eigenvectors = torch.linalg.eig(J)

# Unstable eigenmodes (positive eigenvalues) are control directions
control_modes = eigenvectors[:, eigenvalues.real > 0]
```

**Archetype decomposition** via SVD:
$$
J(t) = U \Sigma V^\top
$$

where:
- **$U$** — Regulatory archetypes (co-regulated gene modules)
- **$V$** — Communication archetypes (influential cellular states)
- **$\Sigma$** — Archetype strengths

---

## Model Comparison

| Feature | Hybrid Drift Field | Schrödinger Bridge |
|---------|-------------------|-------------------|
| **Architecture** | Score + ODE + Velocity | Forward + Backward drifts |
| **Endpoint constraints** | No | Yes (guaranteed) |
| **Optimal transport** | No | Yes (entropic) |
| **Jacobian computation** | ✅ Explicit | ✅ Explicit |
| **Velocity integration** | ✅ Advanced | ⚠️ Can be added |
| **Generative sampling** | ✅ Yes | ⚠️ Limited |
| **Best for** | Control analysis, general trajectories | Aging, development, transport |
| **Training** | Score matching | Iterative Sinkhorn + score matching |
| **Reverse process** | Approximate (via score) | ✅ True backward drift |

**Recommendation**: 
- Use **Hybrid Drift Field** for control factor discovery and general analysis
- Use **Schrödinger Bridge** for aging/development with clear endpoints

---

## Advanced Usage

### Hyperparameter Tuning (Hybrid Model)

**Velocity scaling** (`--vel-scale`, $\lambda$):
- Default: 1.0 for normalized velocities
- Increase (2.0-5.0) for stronger velocity influence
- Decrease (0.1-0.5) if struggling to match endpoints

**Number of neighbors** (`--vel-k`, $k$):
- Default: 32 works well for most datasets
- Increase (64-128) for smoother fields
- Decrease (16-24) for more localized influence

**Time schedule** (`--vel-time-mode`):
- `"mid"` (default): $g(t) = 4t(1-t)$ — emphasize velocity during transitions
- `"flat"`: $g(t) = 1$ — constant velocity contribution

**Confidence gating** (`--vel-conf-power`, $p$):
- Default: 1.0 uses confidence linearly
- Increase (1.5-2.0) to suppress unreliable velocities
- Set to 0.0 to disable gating

### Hyperparameter Tuning (Schrödinger Bridge)

**Entropic regularization** (`--epsilon`):
- Default: 0.1
- Increase (0.5-1.0) for smoother OT plans
- Decrease (0.01-0.05) for sharper transport

**Number of iterations** (`--n-iterations`):
- Default: 10
- Increase (20-50) for better convergence
- Monitor OT cost to check convergence

**Sinkhorn iterations** (`--sinkhorn-max-iter`):
- Default: 100
- Increase (200-500) if OT plan doesn't converge

### Fate-Conditioned Archetype Analysis

```bash
python -m scqdiff.pipeline.train_from_anndata \
    --h5ad your_data.h5ad \
    --use-velocity-prior \
    --fate-index 0 \
    --nbins 10 \
    --rank 3 \
    --out-prefix my_model
```

This computes fate-conditioned Jacobians and extracts regulatory archetypes.

---

## Applications

### 1. Control Factor Discovery (Use Hybrid Model)

**Goal**: Identify Yamanaka-like factors that control cell fate

```python
# Train hybrid model
model = DriftField(cfg, X_ref=X, V_ref=V)

# Compute Jacobian
J = model.jacobian(X, T)

# Find unstable eigenmodes (control directions)
eigenvalues, eigenvectors = torch.linalg.eig(J)
control_factors = eigenvectors[:, eigenvalues.real > 0]

# Map back to genes
top_genes = identify_genes_from_eigenvectors(control_factors, adata.var_names)
```

**Why hybrid model**: Explicit Jacobian, local geometry analysis, not constrained by endpoints

### 2. Aging Analysis (Use Schrödinger Bridge)

**Goal**: Model aging process and identify rejuvenation targets

```python
# Train Schrödinger Bridge
sb = SchrodingerBridge(cfg, X_young, X_old)
sb.train_bridge(n_iterations=10)

# Predict aging
aging_path = sb.forward_integrate(young_cell)

# Predict rejuvenation (TRUE REVERSE!)
rejuvenation_path = sb.backward_integrate(old_cell)

# Identify key genes for rejuvenation
gene_importance = (rejuvenation_path[:, -1] - rejuvenation_path[:, 0]).abs().mean(dim=0)
rejuvenation_targets = torch.argsort(gene_importance, descending=True)[:20]
```

**Why Schrödinger Bridge**: Guaranteed endpoints, true reverse process, optimal transport

### 3. Drug Perturbation Analysis (Use Hybrid Model)

**Goal**: Identify how drugs alter regulatory networks

```python
# Train on control and treated
model_control = train_model(adata_control, use_velocity=True)
model_treated = train_model(adata_treated, use_velocity=True)

# Compare Jacobians
J_control = model_control.jacobian(X, T)
J_treated = model_treated.jacobian(X, T)
delta_J = J_treated - J_control  # Perturbation-induced changes

# Identify altered regulatory modules
altered_genes = identify_altered_modules(delta_J)
```

### 4. Development Modeling (Use Either)

**Hybrid model** — For general developmental trajectories:
```python
model = DriftField(cfg, X_ref=X, V_ref=V)
trajectories = model.integrate(X_progenitor, t_span)
```

**Schrödinger Bridge** — For transport between developmental stages:
```python
sb = SchrodingerBridge(cfg, X_progenitor, X_differentiated)
development_path = sb.forward_integrate(X_progenitor)
dedifferentiation_path = sb.backward_integrate(X_differentiated)
```

---

## What scIDiff Learns

| Layer | Description | Output | Model |
|-------|-------------|--------|-------|
| Regulatory drift | Deterministic transcriptional change | $f(x,t)$ | Both |
| Reverse drift | Reprogramming direction | $f_{\text{rev}}(x,t)$ | SB (true), Hybrid (approx) |
| Temporal Jacobian | Gene-gene regulatory influence | $J(t)=\tfrac{\partial f}{\partial x}$ | Both |
| Regulatory archetypes | Low-rank regulatory modes | $f(x,t)\approx\sum_k a_k(t)A_kx$ | Both |
| Optimal transport | Most probable paths | OT plan $P$ | SB only |
| Control directions | Unstable eigenmodes | $v: \lambda(J) > 0$ | Both |

**Biological Insights**:
- **Attractors** — Zeros of $f(x,t)$ (stable cell states)
- **Differentiation paths** — Flow lines of $f(x,t)$
- **Barriers** — Regions where $|f(x,t)|$ is large
- **Control factors** — Unstable eigenmodes of $J(t)$
- **Rejuvenation targets** — Genes with large changes in backward bridge

---

## Documentation

- **[RNA_VELOCITY_GUIDE.md](RNA_VELOCITY_GUIDE.md)** — Comprehensive guide to velocity integration
- **[QUICKSTART_VELOCITY.md](QUICKSTART_VELOCITY.md)** — Quick start guide
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** — Technical implementation details
- **[CHANGELOG_VELOCITY.md](CHANGELOG_VELOCITY.md)** — Version history
- **[SB_IMPLEMENTATION_GUIDE.md](SB_IMPLEMENTATION_GUIDE.md)** — Schrödinger Bridge implementation guide
- **[SB_QUICK_START.md](SB_QUICK_START.md)** — Quick start for Schrödinger Bridge

---

## Example Workflows

### Workflow 1: Basic Training with Velocity

```bash
# Prepare data with scVelo
# Ensure adata has .layers['velocity'] and .obs['pseudotime']

python -m scqdiff.pipeline.train_from_anndata \
    --h5ad data.h5ad \
    --use-velocity-prior \
    --normalize-velocity \
    --epochs 200 \
    --out-prefix my_model
```

### Workflow 2: Aging Analysis with Schrödinger Bridge

```bash
# Train bridge between young and old cells
python -m scqdiff.pipeline.train_bridge \
    --h5ad aging_data.h5ad \
    --source-key age_group \
    --source-value young \
    --target-value old \
    --n-iterations 10 \
    --out-prefix aging_bridge

# Analyze results
python analyze_aging_bridge.py --model aging_bridge.pt
```

### Workflow 3: Control Factor Discovery

```python
from scqdiff.models.drift import DriftField, DriftConfig
from scqdiff.archetypes.decompose import jacobian_modes

# Train model
cfg = DriftConfig(dim=X.shape[1], use_velocity_prior=True)
model = DriftField(cfg, X_ref=X, V_ref=V)

# Compute Jacobian
J = model.jacobian(X, T)

# Extract control factors
modes = jacobian_modes(J, n_modes=10)

# Identify genes
control_genes = identify_top_genes(modes, adata.var_names)
print(f"Top control genes: {control_genes}")
```

### Workflow 4: Trajectory Simulation

```python
from scqdiff.simulate.trajectories import euler_integrate

# Simulate forward trajectories
x0 = X[:10]  # Starting cells
trajectory = euler_integrate(
    model, x0,
    t0=0.0, t1=1.0,
    steps=100,
    stochastic=True
)

# Visualize
import matplotlib.pyplot as plt
for i in range(10):
    plt.plot(trajectory[i, :, 0], trajectory[i, :, 1], alpha=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Simulated Trajectories')
plt.show()
```

---

## Architecture Details

### Hybrid Drift Field Implementation

```python
class DriftField(nn.Module):
    def __init__(self, cfg, X_ref=None, V_ref=None):
        # Score network (diffusion component)
        self.score = MLPScore(cfg.dim, hidden=cfg.hidden, depth=cfg.depth)
        
        # Residual network (Neural ODE component)
        self.residual = ResidualNet(cfg.dim, hidden=cfg.hidden//2, depth=cfg.depth-1)
        
        # Velocity prior (biological component)
        if cfg.use_velocity_prior and V_ref is not None:
            self.vel = KNNVelocity(X_ref, V_ref, k=cfg.vel_k, tau=cfg.vel_tau)
    
    def forward(self, x, t):
        # Compute drift components
        u = self.cfg.beta * self.score(x, t) + self.residual(x, t)
        
        # Add velocity prior if available
        if hasattr(self, 'vel'):
            v, conf = self.vel(x)
            gate = conf.pow(self.cfg.vel_conf_power)
            g = self._g(t)  # Time schedule
            b = self.cfg.vel_scale * g * gate * v
            u = u + b
        
        return u
```

### Schrödinger Bridge Implementation

```python
class SchrodingerBridge(nn.Module):
    def __init__(self, cfg, X_0, X_1):
        # Forward drift network (source → target)
        self.forward_net = MLPScore(cfg.dim, hidden=cfg.hidden, depth=cfg.depth)
        
        # Backward drift network (target → source)
        self.backward_net = MLPScore(cfg.dim, hidden=cfg.hidden, depth=cfg.depth)
        
        # Store endpoint distributions
        self.X_0 = X_0  # Source (e.g., young)
        self.X_1 = X_1  # Target (e.g., old)
    
    def train_bridge(self, n_iterations=10):
        for iter in range(n_iterations):
            # 1. Compute OT plan
            self.compute_ot_plan()
            
            # 2. Train forward drift
            self.train_forward()
            
            # 3. Train backward drift
            self.train_backward()
```

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use scIDiff in your research, please cite:

```bibtex
@article{scidiff2024,
  title={scIDiff: Learning Single-Cell Regulatory Dynamics with Hybrid Drift Fields},
  author={Terooatea, Tommy},
  journal={bioRxiv},
  year={2024}
}
```

---

## Key References

**Hybrid Drift Field**:
- Song et al. (2021) "Score-Based Generative Modeling through SDEs" *ICLR*
- Chen et al. (2018) "Neural Ordinary Differential Equations" *NeurIPS*
- La Manno et al. (2018) "RNA velocity of single cells" *Nature*

**Schrödinger Bridge**:
- Chen et al. (2021) "Likelihood Training of Schrödinger Bridge using Forward-Backward SDEs Theory" *arXiv*
- Cuturi (2013) "Sinkhorn Distances: Lightspeed Computation of Optimal Transport" *NIPS*
- De Bortoli et al. (2021) "Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling" *NeurIPS*

**Optimal Transport**:
- Peyré & Cuturi (2019) "Computational Optimal Transport" *Foundations and Trends in Machine Learning*

---

## Contact

For questions, issues, or collaborations:
- **GitHub Issues:** https://github.com/yourusername/scIDiff/issues
- **Email:** tommy.terooatea@byu.edu

---

**scIDiff** — *Combining mathematical rigor with biological reality for interpretable single-cell dynamics*
