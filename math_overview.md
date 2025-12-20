# scIDiff — Mathematical Overview

scIDiff (*single-cell Inverse Diffusion*) formulates cell-state transitions as a
stochastic control problem under the **Schrödinger Bridge** framework, with
**RNA velocity incorporated as a reference drift**.

---

## 1. State space and dynamics

Let $X_t \in \mathbb{R}^d$ denote the cell state at (pseudo)time $t \in [0,1]$.
Cell dynamics are modeled by the stochastic differential equation

$$
dX_t = f(X_t,t)\,dt + \sqrt{2\beta}\,dW_t,
$$

where:

- $f : \mathbb{R}^d \times [0,1] \to \mathbb{R}^d$ is the drift field  
- $\beta > 0$ is the diffusion coefficient  
- $W_t$ is standard Brownian motion  

---

## 2. Density evolution

Let $\rho_t(x)$ denote the probability density of $X_t$.
The density evolves according to the **Fokker–Planck equation**

$$
\partial_t \rho_t
=
-\nabla \cdot (\rho_t f)
+
\beta\,\Delta \rho_t,
$$

with boundary conditions

$$
\rho_{t=0} = \rho_0,
\qquad
\rho_{t=1} = \rho_1,
$$

where $\rho_0$ and $\rho_1$ are empirical distributions estimated from data.

---

## 3. Reference drift from RNA velocity

A time-dependent reference drift $b(x,t)$ is constructed from RNA velocity:

$$
b(x,t)
=
\lambda\, g(t)\, w(x)\, \hat{v}(x),
$$

where:

- $\hat{v}(x)$ is an interpolated RNA velocity field  
- $w(x)$ is a confidence weight  
- $g(t)$ is a temporal gating function  
- $\lambda$ is a global scaling parameter  

The reference drift defines a baseline stochastic process.

---

## 4. Drift decomposition

The total drift is decomposed as

$$
f(x,t) = b(x,t) + u_\theta(x,t),
$$

where $u_\theta(x,t)$ is a learnable correction drift parameterized by a neural network.

---

## 5. Schrödinger Bridge objective

scIDiff seeks the stochastic process that transports $\rho_0$ to $\rho_1$
while minimizing deviation from the reference drift $b(x,t)$.

Equivalently, it solves the control problem

$$
\min_{u_\theta}
\;
\mathbb{E}\!\int_0^1
\|u_\theta(X_t,t)\|^2\,dt,
$$

subject to the controlled Fokker–Planck equation

$$
\partial_t \rho_t
=
-\nabla \cdot \big(\rho_t (b + u_\theta)\big)
+
\beta\,\Delta \rho_t,
$$

and endpoint constraints

$$
X_0 \sim \rho_0,
\qquad
X_1 \sim \rho_1.
$$

---

## 6. Learned dynamics

The resulting drift

$$
f(x,t) = b(x,t) + u_\theta(x,t)
$$

defines a full generative dynamical system that is:

- consistent with observed endpoint distributions  
- minimally perturbed from the reference process  
- stochastic and time-continuous  

---

## 7. Summary

scIDiff formulates trajectory inference as a **Schrödinger Bridge with a structured
reference drift**. RNA velocity defines the reference process, and the learned
correction $u_\theta$ accounts for global distributional constraints.

---

*BYU Genomics & Bioinformatics Core — scIDiff Development Team*
