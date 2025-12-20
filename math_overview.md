
# Mathematical Foundations of scIDiff

**scIDiff** (*single-cell Inverse Diffusion*) learns a **time-dependent drift field**
that describes how cellular gene-expression states evolve over continuous time or
pseudotime. It is grounded in the **Schrödinger Bridge formulation of optimal transport**,
linking score-based generative modeling, Fokker–Planck dynamics, and **RNA velocity as a biological prior**.

A key innovation of **scIDiff** is the integration of **RNA velocity** as a *biological reference drift*,
enabling the model to learn trajectories that are not only mathematically optimal but also
**biologically plausible**.

---

## 1. Conceptual overview

Given single-cell data at different developmental or perturbation states,
we seek a continuous stochastic process

$$
X_t \in \mathbb{R}^d
$$

whose probability distribution $\rho_t(x)$ evolves smoothly from an initial (naïve) state
$\rho_0$ to a terminal (perturbed) state $\rho_1$.

scIDiff models cell-state dynamics as:

$$
dX_t = f(X_t,t)\,dt + \sqrt{2\beta}\,dW_t,
$$

where:

- $f(x,t)$ is the **total drift field**
- $dW_t$ is Brownian noise
- $\beta$ controls stochasticity

The corresponding density evolution follows the **Fokker–Planck equation**:

$$
\partial_t\rho_t
=
-\nabla\!\cdot(\rho_t f)
+
\beta\,\Delta\rho_t.
$$

---

## 2. RNA velocity as biological reference drift

### 2.1 Biological motivation

RNA velocity provides an experimentally grounded estimate of a cell’s instantaneous
transcriptional change derived from unspliced and spliced mRNA ratios.
Rather than enforcing velocity as a hard constraint, scIDiff treats RNA velocity as a
**reference drift** that defines a default biological direction of motion in state space.

### 2.2 Reference drift formulation

$$
b(x,t)
=
\lambda \cdot g(t) \cdot w(x) \cdot \hat{v}(x)
$$

---

## 3. Schrödinger Bridge with velocity prior

$$
f(x,t) = b(x,t) + u_\theta(x,t)
$$

---

## 4. Learning the correction drift

$$
u_\theta(x,t)
=
\gamma\,s_\theta(x,t)
+
v_\theta(x,t)
$$

---

## 5. Simulation

$$
dX_t = f(X_t,t)\,dt + \sqrt{2\beta}\,dW_t
$$

---

*BYU Genomics & Bioinformatics Core — scIDiff Development Team*
