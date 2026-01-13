# scIDiff — Mathematical Overview  
*Time-resolved regulatory operators for single-cell dynamics*

scIDiff (**s**ingle-**c**ell **I**nference of **Diff**erential operators) is an operator-centric framework for learning the **time-dependent dynamical rules** that govern cellular state transitions. Rather than only reconstructing trajectories, scIDiff infers the **local Jacobian operators** of a learned drift field and uses their temporal structure to reveal conserved regulatory programs (“operator archetypes”) and decision-sensitive control modes.

**Note.** RNA velocity and Schrödinger Bridge constraints can be used as **optional priors** when learning the drift field, but all downstream operator and archetype analysis is independent of these choices.

---

## 1. Stochastic model of cell-state dynamics

Let  
\[
X(t) \in \mathbb{R}^d
\]
denote the latent cellular state at pseudotime \( t \in [0,1] \).

Cell dynamics are modeled by the stochastic differential equation
\[
dX(t) = f(X(t), t)\,dt + \sqrt{2\beta}\, dW(t),
\]
where

- \( f(x,t) \) is the **time-dependent drift field**,  
- \( \beta > 0 \) controls stochasticity,  
- \( W(t) \) is standard Brownian motion.

The drift \( f(x,t) \) represents the instantaneous **regulatory program** driving cellular motion in latent state space.

---

## 2. Density evolution

Let \( \rho(t,x) \) denote the probability density of \( X(t) \).  
It evolves according to the **Fokker–Planck equation**
\[
\frac{\partial \rho}{\partial t}
= - \nabla \cdot (\rho f) + \beta \Delta \rho .
\]

scIDiff learns \( f(x,t) \) so that this stochastic flow explains the observed single-cell distribution over pseudotime.

---

## 3. Local regulatory operators (Jacobians)

The central object in scIDiff is the **time-resolved Jacobian operator**
\[
J(x,t) = \nabla_x f(x,t).
\]

This governs linearized perturbation dynamics
\[
d(\delta X) = J(X,t)\,\delta X\,dt + \sqrt{2\beta}\, dW(t).
\]

The eigenvalues of \( J(x,t) \) define regulatory behavior:

| Eigenvalue sign | Interpretation |
|----------------|----------------|
| Negative       | Stable (buffered states) |
| Near zero      | Plastic (fragile states) |
| Positive       | Unstable (fate-deciding control directions) |

These operators quantify **stability, sensitivity, and commitment** directly.

---

## 4. Temporal Jacobian tensor

Evaluating Jacobians along pseudotime gives
\[
J(t_1),\, J(t_2),\, \ldots,\, J(t_T).
\]

Stacking them forms the **temporal Jacobian tensor**
\[
\mathcal{T} \in \mathbb{R}^{T \times d \times d},
\]
which summarizes how regulatory sensitivity evolves during differentiation.

---

## 5. Operator archetypes

scIDiff factorizes the temporal tensor as
\[
\mathcal{T}(t) \approx \sum_{k=1}^{K} a_k(t)\, A_k ,
\]
where

- \( A_k \in \mathbb{R}^{d \times d} \) are **operator archetypes**,  
- \( a_k(t) \) are their time-dependent activations.

Sequential and concurrent archetype activation defines a conserved **regulatory grammar of cell fate**.

---

## 6. Control-relevant modes

Eigenvectors of \( A_k \) with **positive eigenvalues** are **unstable modes**: perturbations along these directions are amplified.

When projected into gene space, these modes identify **control-relevant transcriptional programs**.

---

## 7. RNA velocity as an optional drift prior

RNA velocity can guide drift learning via a reference field
\[
b(x,t) = \lambda\, g(t)\, w(x)\, \hat v(x),
\]
where

- \( \hat v(x) \) is RNA velocity,  
- \( w(x) \) is a confidence weight,  
- \( g(t) \) is a temporal gate,  
- \( \lambda \) controls strength.

The learned drift is
\[
f(x,t) = b(x,t) + u_\theta(x,t).
\]

RNA velocity improves **directional alignment** but does not alter Jacobian or archetype analysis.

---

## 8. Endpoint-constrained dynamics (Schrödinger Bridge, optional)

If start and end populations are known, scIDiff enforces them by minimizing control energy
\[
\min_{u} \; \mathbb{E}\!\left[\int \|u(X,t)\|^2 \, dt \right]
\]
subject to
\[
\frac{\partial \rho}{\partial t}
= -\nabla \cdot (\rho (b+u)) + \beta \Delta \rho,
\qquad
X(0)\sim\rho_0,\;\; X(1)\sim\rho_1 .
\]

This modifies only the **drift field**; Jacobians and archetypes are computed in exactly the same way.
