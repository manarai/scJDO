# scIDiff: Schrödinger Bridge Learning of Single-Cell Regulatory Dynamics

**scIDiff** (single-cell inverse Diffusion) is a computational framework for learning
continuous-time, causal cellular dynamics directly from single-cell data.
By framing development, differentiation, perturbation, and reprogramming as a
**Schrödinger Bridge inverse-diffusion problem**, scIDiff infers stochastic drift fields
that describe how cells evolve through transcriptional state space—and how these
processes could, in principle, be reversed.

Unlike conventional optimal-transport or score-based approaches that infer purely
energy-minimal trajectories, scIDiff supports **RNA velocity as a biological reference
drift**. When available, RNA velocity defines a biologically grounded baseline dynamics,
and scIDiff computes a Schrödinger Bridge that **minimally perturbs** this velocity-driven
process to satisfy global distributional constraints.

The result is a hybrid model in which **local transcriptional kinetics guide the flow**,
while **Schrödinger-Bridge regularization guarantees probabilistic consistency,
time-symmetry, and interpretability**.

From the learned dynamics, scIDiff computes **temporal Jacobian tensors** that capture
how gene–gene and cell–cell influences evolve through time. Decomposition of these
Jacobians reveals regulatory and communication archetypes, enabling principled analysis
of irreversibility, commitment points, and reprogramming potential.

---

## Mathematical Foundations

Given observed cell states sampled at different times or conditions, scIDiff models a
controlled stochastic process:

$$
dX_t = \big( b(X_t,t) + u_\theta(X_t,t) \big)\,dt + \sqrt{2\beta}\,dW_t
$$

where:

- $X_t \in \mathbb{R}^d$ — cell state (gene expression or latent embedding)
- $b(x,t)$ — biological reference drift (RNA velocity, when available)
- $u_\theta(x,t)$ — learned Schrödinger-Bridge correction drift
- $\beta$ — diffusion constant

The full drift is defined as:

$$
f(x,t) = b(x,t) + u_\theta(x,t)
$$

### Schrödinger Bridge Objective

The Schrödinger Bridge identifies, among all stochastic processes transporting the
empirical initial distribution $\rho_0$ to the terminal distribution $\rho_1$, the one
that deviates **minimally (in control energy)** from the biological reference drift
$b(x,t)$:

$$
\min_{u_\theta}
\mathbb{E}\left[
\int_0^1 \frac{1}{2\beta}\|u_\theta(X_t,t)\|^2\,dt
\right]
\quad \text{s.t.} \quad
X_0 \sim \rho_0, \; X_1 \sim \rho_1
$$

---

## RNA Velocity as a Reference Drift

RNA velocity provides an experimentally grounded estimate of a cell's instantaneous
transcriptional change. Rather than enforcing velocity as a hard constraint, scIDiff
treats RNA velocity as a **reference drift** defining default biological motion in state
space.

The reference drift is defined as:

$$
b(x,t) = \lambda \cdot g(t) \cdot w(x) \cdot \hat v(x)
$$

where:

- $\hat v(x)$ — RNA velocity vector field interpolated from observed cells via soft k-nearest neighbors
- $w(x) = \text{conf}(x)^p$ — velocity confidence raised to power $p$ (default: $p=1$)
- $g(t)$ — time-dependent gating function
- $\lambda$ — velocity magnitude scaling parameter (default: $\lambda=1$)

### Time-Dependent Gating

The gating function $g(t)$ controls when velocity guidance is strongest:

$$
g(t) = \begin{cases}
4t(1-t) & \text{if mode="mid" (default)} \\
1 & \text{if mode="flat"}
\end{cases}
$$

The **"mid" schedule** emphasizes velocity during transitions (peaks at $t=0.5$) while
giving the model freedom at boundaries to match endpoint distributions. This aligns with
the biological intuition that velocity is most reliable during active state transitions.

### Velocity Interpolation

RNA velocity is available only at discrete cell locations. To evaluate $\hat v(x)$ at any
point in state space, scIDiff uses **soft k-nearest neighbors interpolation**:

$$
\hat v(x) = \sum_{i=1}^k w_i \cdot v_i
\quad \text{where} \quad
w_i = \frac{\exp(-d_i/\tau)}{\sum_j \exp(-d_j/\tau)}
$$

where $d_i$ is the distance to the $i$-th nearest reference cell, $v_i$ is its velocity,
and $\tau$ is a temperature parameter controlling smoothness.

The Schrödinger Bridge learns a correction drift $u_\theta(x,t)$ that minimally adjusts
this velocity-driven process to match observed start and end populations.

---

## Implementation Details

### Learned Correction Drift

The learned correction is decomposed as:

$$
u_\theta(x,t) = \gamma \cdot \text{score}(x,t) + \text{residual}(x,t)
$$

where:
- $\gamma$ — score weighting parameter (distinct from diffusion constant $\beta$)
- $\text{score}(x,t)$ — neural network estimating the score function $\nabla_x \log \rho_t(x)$
- $\text{residual}(x,t)$ — neural network learning additional corrections

Optional graph Laplacian smoothing can be applied to the learned correction:

$$
u_\theta(x,t) \leftarrow u_\theta(x,t) - \lambda_L (x \mathbf{L}^\top)
$$

where $\mathbf{L}$ is the graph Laplacian and $\lambda_L$ is the smoothing strength.

### Training Objective

The model is trained by minimizing:

$$
\mathcal{L} = \mathcal{L}_{\text{DSM}} + \alpha_1 \mathcal{L}_{\text{control}} + \alpha_2 \mathcal{L}_{\text{FP}} + \alpha_3 \mathcal{L}_{\text{smooth}}
$$

where:
- $\mathcal{L}_{\text{DSM}}$ — denoising score matching loss
- $\mathcal{L}_{\text{control}} = \mathbb{E}[\|u_\theta\|^2]$ — control energy regularization
- $\mathcal{L}_{\text{FP}}$ — Fokker-Planck residual loss
- $\mathcal{L}_{\text{smooth}}$ — Laplacian smoothness regularization

---

## Forward and Reverse Dynamics

**Forward process**

$$
dX_t = f(X_t,t)\,dt + \sqrt{2\beta}\,dW_t
$$

**Reverse process**

$$
dX_t =
\big[f(X_t,t) - 2\beta\nabla_x\log\rho_t(X_t)\big]dt
+ \sqrt{2\beta}\,d\bar W_t
$$

where the reverse drift is given by:

$$
f_{\mathrm{rev}}(x,t) = f(x,t) - 2\beta\nabla_x\log\rho_t(x)
$$

The reverse dynamics assume a sufficiently smooth intermediate density $\rho_t(x)$, as standard in Schrödinger Bridge theory.

Forward–reverse asymmetry provides a quantitative measure of biological irreversibility.

---

## Temporal Jacobians and Archetypes

The temporal Jacobian tensor is computed from the full drift:

$$
J(t) = \frac{\partial f}{\partial x}(t)
$$

where $J_{ij}(t) = \frac{\partial f_i}{\partial x_j}$ quantifies how feature $j$ influences
the rate of change of feature $i$ at time $t$ (when $x$ represents gene expression, this captures gene-gene regulatory influence; in latent spaces, $J(t)$ captures effective regulatory structure).

Low-rank decomposition via SVD reveals regulatory and communication archetypes:

$$
J(t) = U \Sigma V^\top
$$

where:
- **$U$** — regulatory archetypes (co-regulated gene modules)
- **$V$** — communication archetypes (influential cellular states)
- **$\Sigma$** — singular values (archetype strengths)

---

## Quantifying Irreversibility

Entropy production is quantified as:

$$
\dot S(t) =
\mathbb{E}\left[
\frac{\|f(X_t,t)-f_{\mathrm{rev}}(X_t,t)\|^2}{2\beta}
\right]
$$

High entropy production indicates irreversible cell-fate decisions and commitment points.

---

## What scIDiff Learns

| Layer                  | Description                                      | Output                            |
| ---------------------- | ------------------------------------------------ | --------------------------------- |
| Regulatory drift       | Deterministic direction of transcriptional change | $f(x,t)$                          |
| Reverse drift          | Reprogramming direction opposing natural evolution | $f_{\text{rev}}(x,t)$             |
| Irreversibility field  | Entropic asymmetry between forward and reverse   | $\Delta f = f - f_{\text{rev}}$   |
| Temporal Jacobian      | Local causal gene-to-gene influence              | $J(t)=\tfrac{\partial f}{\partial x}$ |
| Regulatory archetypes  | Low-rank modes of temporal regulation            | $f(x,t)\approx\sum_k a_k(t)A_kx$ |
| (Optional) Communication | Dynamic ligand–receptor signaling between cells    | $W_{ij}(t)$                       |

**Biological Insight:** The learned drift $f(x,t)$ captures:

- Attractors as zeros of $f(x,t)$ (fixed points)
- Differentiation paths as flow lines of $f(x,t)$
- Barriers as regions where $|f(x,t)|$ is large (rapid transitions)

---

## Extension: scIDiff-Comm (Cell–Cell Communication)

With CellPhoneDB ligand–receptor priors, scIDiff models time-evolving communication graphs as coupled stochastic processes:

$$
dX_t^{(i)} = f_{\text{intra}}(X_t^{(i)}, t)dt + \sum_j W_{ij}(t)\,\phi(X_t^{(j)} - X_t^{(i)})dt + \sqrt{2\beta}dW_t^{(i)}
$$

where:
- $W_{ij}(t)$ — communication strength between cells *i* and *j*
- $\phi(\Delta x)$ — interaction kernel (e.g., linear or gated)

**Result:** Communication archetypes (e.g., inflammatory relay, exhaustion/resolution) that co-evolve with regulatory drift.

---

## Installation

```bash
# Create conda environment
conda create -n scidiff python=3.10
conda activate scidiff

# Clone repository
git clone https://github.com/manarai/scIDiff_V2.git
cd scIDiff_V2

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

---

## Quick Start

### Command Line Interface

Train scIDiff with RNA velocity as biological prior:

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

**Key arguments:**
- `--use-velocity-prior` — Enable RNA velocity integration
- `--normalize-velocity` — Normalize velocity to unit length (recommended)
- `--vel-scale 1.0` — Velocity magnitude scaling ($\lambda$, tune if needed)
- `--vel-k 32` — Number of neighbors for interpolation
- `--vel-time-mode mid` — Time schedule ("mid" or "flat")
- `--beta 0.1` — Diffusion constant ($\beta$)

### Python API

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

# Normalize velocity (recommended)
if V is not None:
    V = V / (V.norm(dim=1, keepdim=True) + 1e-8)

# Configure model with velocity prior
cfg = DriftConfig(
    dim=X.shape[1],
    beta=0.1,              # β: diffusion constant
    use_velocity_prior=True,
    vel_scale=1.0,         # λ: velocity magnitude scaling
    vel_k=32,              # k: number of neighbors
    vel_tau=1.0,           # τ: temperature parameter
    vel_conf_power=1.0,    # p: confidence exponent
    vel_time_mode="mid"    # g(t): time schedule
)

# Create model
model = DriftField(cfg, X_ref=X, V_ref=V)

# Compute drift
drift = model(X, T)

# Compute Jacobian
J = model.jacobian(X[:10], T[:10])
```

### Training Without Velocity

scIDiff is fully backward compatible. To train without velocity:

```bash
python -m scqdiff.pipeline.train_from_anndata \
    --h5ad your_data.h5ad \
    --ptime-key pseudotime \
    --epochs 200
```

Or in Python:

```python
cfg = DriftConfig(dim=X.shape[1], beta=0.1)
model = DriftField(cfg)  # No velocity
```

---

## Advanced Usage

### Hyperparameter Tuning

**Velocity scaling** (`--vel-scale`, $\lambda$):
- Start with 1.0 for normalized velocities
- Increase (2.0-5.0) if velocity should have stronger influence
- Decrease (0.1-0.5) if model struggles to match endpoints

**Number of neighbors** (`--vel-k`, $k$):
- Default: 32 works well for most datasets
- Increase (64-128) for smoother velocity fields
- Decrease (16-24) for more localized influence

**Time schedule** (`--vel-time-mode`):
- `"mid"` (default): Emphasize velocity during transitions, $g(t) = 4t(1-t)$
- `"flat"`: Constant velocity contribution, $g(t) = 1$

**Confidence gating** (`--vel-conf-power`, $p$):
- Default: 1.0 uses confidence linearly
- Increase (1.5-2.0) to more strongly suppress unreliable velocities
- Set to 0.0 to disable confidence gating

**Diffusion constant** (`--beta`, $\beta$):
- Controls noise level in the stochastic process
- Default: 0.1 works well for most datasets
- Increase for more stochastic trajectories

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

## Documentation

- **[RNA_VELOCITY_GUIDE.md](RNA_VELOCITY_GUIDE.md)** — Comprehensive guide to velocity integration
- **[QUICKSTART_VELOCITY.md](QUICKSTART_VELOCITY.md)** — Quick start guide
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** — Technical implementation details
- **[CHANGELOG_VELOCITY.md](CHANGELOG_VELOCITY.md)** — Version history and changes

---

## 🧮 Core Features

- ✅ **Schrödinger Bridge learning** with biologically grounded reference drift
- ✅ **RNA velocity integration** via soft k-NN interpolation
- ✅ **Time-dependent gating** for trajectory-aware velocity guidance
- ✅ **Confidence-based weighting** to handle unreliable velocity estimates
- ✅ **Temporal Jacobian computation** for gene regulatory network inference
- ✅ **Archetype decomposition** revealing regulatory and communication modules
- ✅ **Forward and reverse dynamics** for irreversibility quantification
- ✅ **Graph Laplacian smoothing** for spatial coherence
- ✅ **Flexible configuration** with extensive hyperparameter control
- ✅ **Backward compatible** — works with or without velocity

---

## 🔬 Applications

scIDiff is particularly valuable for:

- **Drug Response Studies** — Detecting altered splicing patterns indicating network rewiring
- **Differentiation Modeling** — Ensuring trajectories follow biologically feasible paths
- **Perturbation Analysis** — Identifying how interventions alter transcriptional dynamics
- **Reprogramming** — Understanding barriers to cell fate conversion and identifying key factors
- **Development** — Modeling temporal progression through developmental stages
- **Disease Progression** — Characterizing pathological state transitions

---

## 🧪 Example Workflows

### 1. Basic Training with Velocity

```bash
# Prepare data with scVelo or similar
# Ensure adata has .layers['velocity'] and .obs['pseudotime']

python -m scqdiff.pipeline.train_from_anndata \
    --h5ad data.h5ad \
    --use-velocity-prior \
    --normalize-velocity \
    --epochs 200
```

### 2. Drug Perturbation Analysis

```python
# Train on control and treated conditions
model_control = train_model(adata_control, use_velocity=True)
model_treated = train_model(adata_treated, use_velocity=True)

# Compare Jacobians to identify altered regulatory networks
J_control = model_control.jacobian(X, T)
J_treated = model_treated.jacobian(X, T)
delta_J = J_treated - J_control  # Perturbation-induced changes
```

### 3. Trajectory Simulation

```python
from scqdiff.simulate.integrate import integrate_sde

# Simulate forward trajectories
t_span = torch.linspace(0, 1, 100)
x0 = X[:10]  # Starting cells
trajectories = integrate_sde(model, x0, t_span)

# Visualize
import matplotlib.pyplot as plt
for traj in trajectories:
    plt.plot(traj[:, 0], traj[:, 1], alpha=0.5)
plt.show()
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
  title={scIDiff: Schrödinger Bridge Learning of Single-Cell Regulatory Dynamics},
  author={[Tommy Terooatea]},
  journal={[Journal]},
  year={2024}
}
```

---

## Acknowledgments

This implementation integrates RNA velocity as a biological prior following best practices
from the Schrödinger Bridge and optimal transport literature. The design prioritizes
biological interpretability while maintaining mathematical rigor and computational efficiency.

**Key References:**
- La Manno et al. (2018) "RNA velocity of single cells" *Nature*
- Chen et al. (2021) "Likelihood Training of Schrödinger Bridge using Forward-Backward SDEs Theory"
- Cuturi (2013) "Sinkhorn Distances: Lightspeed Computation of Optimal Transport"

---

## 📧 Contact

For questions, issues, or collaborations:
- **GitHub Issues:** https://github.com/manarai/scIDiff_V2/issues
- **Email:** tommy.terooatea@byu.edu

---

**scIDiff** — *Bridging the gap between mathematical optimality and biological reality*
