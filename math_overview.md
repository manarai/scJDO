# scIDiff — Mathematical Overview
*Time-resolved regulatory operators for single-cell dynamics*

scIDiff (**s**ingle-**c**ell **I**nference of **Diff**erential operators) is an operator-centric framework for learning the **time-dependent dynamical rules** that govern cellular state transitions. Rather than only reconstructing trajectories, scIDiff infers the **local Jacobian operators** of a learned drift field and uses their temporal structure to reveal conserved regulatory programs (“operator archetypes”) and decision-sensitive control modes.

RNA velocity and Schrödinger Bridge constraints can be used as **optional priors** when learning the drift field, but all downstream operator and archetype analysis is independent of these choices.

---

## 1. Stochastic model of cell-state dynamics

Let \(X_t \in \mathbb{R}^d\) denote the latent cellular state at pseudotime \(t \in [0,1]\).
Cell dynamics are modeled as a stochastic differential equation

\[
dX_t = f(X_t,t)\,dt + \sqrt{2\beta}\,dW_t,
\]

where  

- \(f : \mathbb{R}^d \times [0,1] \to \mathbb{R}^d\) is a time-dependent drift field,  
- \(\beta > 0\) controls stochasticity,  
- \(W_t\) is standard Brownian motion.

The drift \(f(x,t)\) represents the **instantaneous regulatory program** driving cellular motion in latent state space.

---

## 2. Density evolution

Let \(\rho_t(x)\) be the probability density of \(X_t\). It evolves according to the Fokker–Planck equation

\[
\partial_t \rho_t
=
-\nabla \cdot (\rho_t f)
+
\beta \Delta \rho_t.
\]

scIDiff learns \(f(x,t)\) so that this stochastic flow explains the observed single-cell distribution over pseudotime.

---

## 3. Local regulatory operators (Jacobians)

The central object in scIDiff is the **time-resolved Jacobian operator**

\[
J(x,t) = \nabla_x f(x,t).
\]

This operator governs the linearized dynamics of perturbations:

\[
d(\delta X_t)
=
J(X_t,t)\,\delta X_t\,dt
+
\sqrt{2\beta}\,dW_t.
\]

The eigenvalues and eigenvectors of \(J(x,t)\) define:

- **Stable modes** (negative eigenvalues): buffered, committed programs  
- **Weakly stable modes** (near-zero eigenvalues): plastic directions  
- **Unstable modes** (positive eigenvalues): fate-deciding control directions  

These operators quantify **stability, sensitivity, and commitment** directly.

---

## 4. Temporal Jacobian tensor

Evaluating Jacobians along inferred trajectories or pseudotime windows yields a sequence

\[
\{J(t_1), J(t_2), \dots, J(t_T)\}.
\]

Stacking them forms the **temporal Jacobian tensor**

\[
\mathcal{T} \in \mathbb{R}^{T \times d \times d}.
\]

This tensor summarizes how regulatory sensitivity evolves during differentiation.

---

## 5. Operator archetypes

To obtain an interpretable representation, scIDiff performs a low-rank tensor factorization

\[
\mathcal{T}(t)
\;\approx\;
\sum_{k=1}^{K} a_k(t)\,A_k,
\]

where  

- \(A_k \in \mathbb{R}^{d \times d}\) are **operator archetypes** (reusable regulatory programs),  
- \(a_k(t)\) are their time-varying activation profiles.

Archetype coordination patterns—**sequential handoffs** and **concurrent activation**—constitute a conserved regulatory grammar of cell fate decisions.

---

## 6. Control-relevant modes

Eigenvectors of archetypal operators \(A_k\) with positive eigenvalues define **unstable modes**—directions where perturbations are amplified.  
When projected to gene space, these modes identify **control-relevant transcriptional programs** that disproportionately influence fate outcomes.

---

## 7. RNA velocity as an optional drift prior

RNA velocity can be incorporated as a **reference drift** \(b(x,t)\) to guide learning of \(f\):

\[
b(x,t)
=
\lambda\, g(t)\, w(x)\, \hat v(x),
\]

where  

- \(\hat v(x)\) is the interpolated RNA velocity field,  
- \(w(x)\) is a confidence weight,  
- \(g(t)\) is a temporal gate,  
- \(\lambda\) controls its influence.

The learned drift is

\[
f(x,t) = b(x,t) + u_\theta(x,t),
\]

where \(u_\theta\) is a neural correction.  
RNA velocity improves directional alignment but does **not** alter downstream Jacobian or archetype analysis.

---

## 8. Endpoint-constrained dynamics via Schrödinger Bridge (optional)

For problems with known start and end populations, scIDiff can enforce endpoint constraints via a Schrödinger Bridge:

\[
\min_{u_\theta}
\;
\mathbb{E}\!\int_0^1 \|u_\theta(X_t,t)\|^2\,dt
\]

subject to

\[
\partial_t \rho_t
=
-\nabla\!\cdot(\rho_t (b+u_\theta))
+
\beta \Delta \rho_t,
\quad
X_0\!\sim\!\rho_0,\;
X_1\!\sim\!\rho_1.
\]

This modifies only the inferred drift \(f\); Jacobians, archetypes, and operator-defined states are computed exactly as in the unconstrained case.

---

## 9. Summary

scIDiff is an **operator-centric** framework that infers time-resolved regulatory dynamics from single-cell data. By learning a drift field, computing its Jacobian, and decomposing the resulting temporal tensor into archetypes, scIDiff exposes a conserved grammar of cell fate decisions and identifies control-relevant unstable modes for perturbation and reprogramming.

RNA velocity and Schrödinger Bridge provide optional guidance and constraints—but the core object of scIDiff is the **time-varying regulatory operator**.
