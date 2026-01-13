# Mathematical Foundations of scIDiff

**scIDiff** (“single-cell Inference of Differential operators”) learns a *time-dependent drift field*
that governs how cellular states move through latent space.  
Unlike trajectory methods, scIDiff focuses on the **local Jacobian operators** of this drift,
which encode stability, fragility, and fate-controlling regulatory modes.

RNA velocity and Schrödinger-Bridge constraints may be used to guide drift learning,
but all operator and archetype analysis is performed on the learned dynamical system itself.

---

## 1. Conceptual overview

We model a cell as a point

$$
X_t \in \mathbb{R}^d
$$

evolving over pseudotime $t \in [0,1]$.  
The stochastic dynamics are

$$
dX_t = f(X_t,t)\,dt + \sqrt{2\beta}\,dW_t,
$$

where

- $f(x,t)$ is the **regulatory drift field**  
- $\beta$ controls stochasticity  
- $W_t$ is Brownian motion  

The drift $f(x,t)$ represents the instantaneous regulatory program driving cell-state change.

---

## 2. Population-level dynamics

Let $\rho_t(x)$ denote the probability density of $X_t$.  
It evolves according to the Fokker–Planck equation

$$
\partial_t \rho_t
= -\nabla\!\cdot(\rho_t f) + \beta\,\Delta \rho_t.
$$

scIDiff learns $f(x,t)$ so that this stochastic flow explains the observed distribution of single cells across pseudotime.

---

## 3. Local regulatory operators (Jacobians)

The central object in scIDiff is the **Jacobian of the drift**

$$
J(x,t) = \frac{\partial f(x,t)}{\partial x}.
$$

This governs how small perturbations evolve:

$$
d(\delta X_t) = J(X_t,t)\,\delta X_t\,dt + \sqrt{2\beta}\,dW_t.
$$

Eigenvalues of $J(x,t)$ determine local regulatory behavior:

| Eigenvalue | Meaning |
|-----------|--------|
| $\lambda < 0$ | Stable, buffered state |
| $\lambda \approx 0$ | Fragile or plastic |
| $\lambda > 0$ | Unstable, fate-controlling direction |

Thus Jacobians directly quantify **stability, sensitivity, and commitment**.

---

## 4. Temporal Jacobian tensor

Sampling Jacobians along pseudotime gives

$$
J(t_1),\,J(t_2),\,\ldots,\,J(t_T).
$$

These form the **temporal Jacobian tensor**

$$
\mathcal{J} \in \mathbb{R}^{d \times d \times T},
$$

which captures how regulatory sensitivities change over time.

---

## 5. Operator archetypes

scIDiff factorizes this tensor as

$$
\mathcal{J} \;\approx\; \sum_{k=1}^{K} A_k \, c_k(t),
$$

where

- $A_k \in \mathbb{R}^{d\times d}$ are **operator archetypes**  
- $c_k(t)$ are their time-dependent activations  

Each $A_k$ is a reusable **regulatory control module**.  
The functions $c_k(t)$ encode when each module is active.

---

## 6. Control-relevant modes

Eigenvectors of an archetype $A_k$ with positive eigenvalues are **unstable control modes**:
perturbations along these directions grow and drive fate transitions.

When mapped into gene space, these modes identify **control-relevant transcriptional programs**
whose modulation has large dynamical impact.

---

## 7. RNA velocity as an optional drift prior

RNA velocity can guide learning through a reference field

$$
b(x,t) = \lambda\,g(t)\,w(x)\,\hat v(x),
$$

where

- $\hat v(x)$ is RNA velocity  
- $w(x)$ is a confidence weight  
- $g(t)$ is a temporal gate  
- $\lambda$ controls strength  

The learned drift is

$$
f(x,t) = b(x,t) + u_\theta(x,t).
$$

Velocity improves directional alignment but does not alter Jacobian or archetype analysis.

---

## 8. Endpoint-constrained dynamics (optional Schrödinger Bridge)

If start and end populations $\rho_0$ and $\rho_1$ are known, scIDiff can enforce them by minimizing

$$
\min_u \; \mathbb{E}\!\int_0^1 \|u(X_t,t)\|^2\,dt
$$

subject to

$$
\partial_t \rho_t
= -\nabla\!\cdot\!\bigl(\rho_t (b+u)\bigr) + \beta\,\Delta \rho_t,
$$

$$
X_0 \sim \rho_0,
\qquad
X_1 \sim \rho_1.
$$

This modifies only the drift field; **Jacobian operators and archetypes are computed the same way.**

---

## 9. Interpretation

scIDiff treats cell fate as a **dynamical control problem**.
Instead of clustering states or tracing paths, it identifies

- where the system is **stable**  
- where it is **fragile**  
- and which **unstable regulatory modes** decide fate.

These are encoded in the time-resolved Jacobian operators and their archetypes.
