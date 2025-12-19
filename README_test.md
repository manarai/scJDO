# 🧬 scIDiff: Schrödinger Bridge Learning of Single-Cell Regulatory Dynamics

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

## 🌌 Mathematical Foundations

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

### Schrödinger Bridge Objective

The Schrödinger Bridge identifies, among all stochastic processes transporting the
empirical initial distribution $\rho_0$ to the terminal distribution $\rho_1$, the one
that deviates **minimally** from the reference dynamics:

$$
\min_{u_\theta}
\mathbb{E}\left[
\int_0^1 \frac{1}{2\beta}\|u_\theta(X_t,t)\|^2\,dt
\right]
\quad \text{s.t.} \quad
X_0 \sim \rho_0, \; X_1 \sim \rho_1
$$

---

## 🧬 RNA Velocity as a Reference Drift

RNA velocity provides an experimentally grounded estimate of a cell’s instantaneous
transcriptional change. Rather than enforcing velocity as a hard constraint, scIDiff
treats RNA velocity as a **reference drift** defining default biological motion in state
space.

The reference drift is defined as:

$$
b(x,t) = g(t)\, w(x)\, \hat v(x)
$$

where:

- $\hat v(x)$ — RNA velocity vector field interpolated from observed cells
- $w(x) \in [0,1]$ — velocity confidence or reliability weight
- $g(t)$ — time-dependent gating function (strongest mid-trajectory, zero at endpoints)

The Schrödinger Bridge learns a correction drift $u_\theta(x,t)$ that minimally adjusts
this velocity-driven process to match observed start and end populations.

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

Forward–reverse asymmetry provides a quantitative measure of biological irreversibility.

---

## Temporal Jacobians and Archetypes

The temporal Jacobian tensor is computed from the full drift:

$$
J(t) = \frac{\partial f}{\partial x}(t)
$$

Low-rank decomposition of $J(t)$ reveals regulatory and communication archetypes.

---

## ⚖️ Quantifying Irreversibility

Entropy production is quantified as:

$$
\dot S(t) =
\mathbb{E}\left[
\frac{\|f(X_t,t)-f_{\mathrm{rev}}(X_t,t)\|^2}{2\beta}
\right]
$$

---

## Installation

```bash
conda create -n scidiff python=3.10
conda activate scidiff
git clone https://github.com/manarai/scIDiff.git
cd scIDiff
pip install -r requirements.txt
pip install -e .
```

---

## Quick Start

```python
import scidiff
import scanpy as sc

adata = sc.read_h5ad("your_data.h5ad")

model = scidiff.SchrodingerBridge(
    adata,
    time_key="pseudotime",
    velocity_layer="velocity"  # optional
)

model.train(n_epochs=1000)
```
