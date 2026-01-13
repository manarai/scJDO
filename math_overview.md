# scIDiff — Mathematical Framework  
*Time-resolved regulatory operators for single-cell dynamics*

scIDiff (**s**ingle-**c**ell **I**nference of **Diff**erential operators) is an operator-centric framework for learning the **time-dependent dynamical rules** that govern cellular state transitions. Instead of reconstructing only trajectories, scIDiff infers the **local Jacobian operators** of a learned drift field and uses their temporal organization to reveal conserved regulatory programs (“operator archetypes”) and fate-controlling instability modes.

RNA velocity and Schrödinger Bridge constraints may be used as **optional priors** for learning the drift field, but all downstream operator and archetype analysis is independent of these choices.

---

## 1. Stochastic model of cellular dynamics

Let  
\[
X(t)\in\mathbb{R}^d
\]
denote the latent cellular state at pseudotime \(t\in[0,1]\). Cell dynamics are modeled as a controlled stochastic differential equation

\[
dX(t)=f(X(t),t)\,dt+\sqrt{2\beta}\,dW(t),
\]

where  

- \(f(x,t)\) is the **time-dependent regulatory drift**,  
- \(\beta>0\) controls intrinsic stochasticity,  
- \(W(t)\) is standard Brownian motion.

The drift field \(f(x,t)\) represents the instantaneous **regulatory program** driving cells through latent state space.

---

## 2. Population-level evolution

Let \(\rho(t,x)\) denote the probability density of \(X(t)\). It evolves according to the Fokker–Planck equation

\[
\frac{\partial \rho}{\partial t}
= -\nabla\cdot(\rho f) + \beta\,\Delta\rho.
\]

scIDiff learns \(f(x,t)\) so that this stochastic flow reproduces the observed distribution of single cells over pseudotime.

---

## 3. Local regulatory operators (Jacobians)

The central object in scIDiff is the **time-resolved Jacobian operator**

\[
J(x,t)=\nabla_x f(x,t)\in\mathbb{R}^{d\times d}.
\]

This governs linearized perturbation dynamics around a cell state:

\[
d(\delta X)=J(X,t)\,\delta X\,dt+\sqrt{2\beta}\,dW(t).
\]

The eigenvalues of \(J(x,t)\) determine local regulatory behavior:

| Eigenvalue | Interpretation |
|------------|----------------|
| \(\lambda<0\) | Stable (buffered states) |
| \(\lambda\approx 0\) | Plastic or fragile states |
| \(\lambda>0\) | Unstable, fate-controlling directions |

Thus, Jacobians directly quantify **stability, sensitivity, and commitment**.

---

## 4. Temporal Jacobian tensor

Evaluating Jacobians along pseudotime gives

\[
J(t_1),J(t_2),\dots,J(t_T).
\]

Stacking them defines the **temporal Jacobian tensor**

\[
\mathcal T\in\mathbb{R}^{T\times d\times d},
\]

which captures how regulatory sensitivity evolves during differentiation, activation, or aging.

---

## 5. Operator archetypes

scIDiff factorizes the temporal tensor as

\[
\mathcal T(t)\approx\sum_{k=1}^K a_k(t)\,A_k,
\]

where  

- \(A_k\in\mathbb{R}^{d\times d}\) are **operator archetypes**,  
- \(a_k(t)\) are their time-dependent activations.

These archetypes represent reusable **regulatory control modules**. Their sequential and overlapping activation defines a conserved **regulatory grammar** of cell fate.

---

## 6. Control-relevant modes

Eigenvectors of an archetype \(A_k\) with positive eigenvalues are **unstable control modes**: perturbations along these directions grow and determine fate transitions.

When projected into gene space, these modes identify **control-relevant transcriptional programs** whose modulation produces large dynamical effects.

---

## 7. RNA velocity as an optional drift prior

RNA velocity can guide drift learning via a reference field

\[
b(x,t)=\lambda\,g(t)\,w(x)\,\hat v(x),
\]

where  

- \(\hat v(x)\) is RNA velocity,  
- \(w(x)\) is a confidence weight,  
- \(g(t)\) is a temporal gate,  
- \(\lambda\) controls strength.

The learned drift is

\[
f(x,t)=b(x,t)+u_\theta(x,t).
\]

Velocity improves directional alignment but does not alter Jacobian- or archetype-based analysis.

---

## 8. Endpoint-constrained dynamics (Schrödinger Bridge, optional)

If start and end populations \(\rho_0,\rho_1\) are known, scIDiff may enforce them by minimizing control energy

\[
\min_u\;\mathbb{E}\!\left[\int\|u(X,t)\|^2dt\right]
\]

subject to

\[
\frac{\partial\rho}{\partial t}
=-\nabla\cdot\bigl(\rho(b+u)\bigr)+\beta\Delta\rho,\qquad
X(0)\sim\rho_0,\;\;X(1)\sim\rho_1.
\]

This modifies only the drift field; **Jacobian operators and archetypes are computed in the same way.**

---

## Conceptual summary

scIDiff treats single-cell systems as **time-dependent dynamical operators**, not merely trajectories.  
Cell fate is governed by a small set of **unstable regulatory modes** whose activation is encoded in the evolving Jacobian tensor.  
Learning these operators reveals the **control logic of cell identity, differentiation, and fragility**.
