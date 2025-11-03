# Mathematical Foundations of scQDiff

**scQDiff** (“single-cell Quantum Diffusion”) learns a *time-dependent drift field*
that describes how cellular gene-expression states evolve over continuous time or
pseudotime. It is grounded in the **Schrödinger Bridge formulation of optimal transport**,  
linking score-based generative modeling, Fokker–Planck dynamics, and biological regulatory interpretation.

---

## 1. Conceptual overview

Given single-cell data at different developmental or perturbation states,
we seek a continuous process $X_t \in \mathbb{R}^d$ whose probability
distribution $\rho_t(x)$ evolves smoothly from an initial (naïve) state
$\rho_0$ to a terminal (perturbed) state $\rho_1$.

scQDiff estimates a **drift field**
$$
u(x,t) \approx \frac{d\,\mathbb{E}[X_t]}{dt},
$$
which defines the deterministic part of the cell-state dynamics:

$$
u(x,t) \approx \frac{d\,\mathbb{E}[X_t]}{dt},
$$

where $dW_t$ is Brownian noise and $\beta$ controls stochasticity.
The associated density $\rho_t$ satisfies the **Fokker–Planck equation**

$$
\partial_t\rho_t = -\nabla\!\cdot(\rho_t u) + \beta\,\Delta\rho_t.
$$

---

## 2. Schrödinger Bridge formulation

We wish to find the *least-energy* control $u_t$
that steers the diffusion from $\rho_0$ to $\rho_1$:

$$
\begin{aligned}
\min_{u_t}\;&
\mathbb{E}\!\int_0^1 \|u_t(X_t)\|^2\,dt, \\
\text{s.t. }&
\partial_t\rho_t = -\nabla\!\cdot(\rho_t u_t) + \beta\,\Delta\rho_t,\\
& \rho_{t=0} = \rho_0,\quad \rho_{t=1} = \rho_1.
\end{aligned}
$$

This is the **Schrödinger Bridge problem**—a stochastic, entropy-regularized
variant of optimal transport. Its solution yields the most likely
“flow of probability” between the observed cellular distributions.

In practice, $u_t$ is represented by a neural network $u_\theta(x,t)$,
and the optimal-transport constraints are imposed through learning losses.

---

## 3. Learning the drift field

### 3.1 Score-based approximation
For small diffusion $\beta$, the optimal drift relates to the
**score function** $s(x,t) = \nabla_x \log\rho_t(x)$:

$$
u(x,t) \approx \beta\,s(x,t).
$$

scQDiff learns $s_\theta(x,t)$ via **denoising score matching (DSM)**:
$$
\mathcal{L}_{\text{DSM}} =
\mathbb{E}_{x,t,\epsilon}\big\|
s_\theta(x+\sigma\epsilon,t)
+ \frac{\epsilon}{\sigma^2}
\big\|^2.
$$

### 3.2 Residual control term
A flexible residual $v_\theta(x,t)$ corrects the score-based drift:
$$
u_\theta(x,t) = \beta\,s_\theta(x,t) + v_\theta(x,t).
$$

### 3.3 Additional regularizers
- **Control energy:** $\|u_\theta\|^2$
- **Fokker–Planck residual:**
  $$
  \mathcal{L}_{\text{FP}}
  = \big\|
  \partial_t\rho_t + \nabla\!\cdot(\rho_t u_\theta)
  - \beta\,\Delta\rho_t
  \big\|^2
  $$
  (approximated from mini-batch divergence)
- **Graph Laplacian smoothness:**
  $$
  \mathcal{L}_{\text{lap}}
  = u_\theta(x,t)^{\!\top}\!L\,u_\theta(x,t),
  $$
  where $L$ is a kNN or regulatory-prior Laplacian.

The total training objective:
$$
\mathcal{L}_{\text{total}}
= \mathcal{L}_{\text{DSM}}
+ \lambda_{\text{FP}}\mathcal{L}_{\text{FP}}
+ \lambda_{\text{lap}}\mathcal{L}_{\text{lap}}
+ \lambda_E\|u_\theta\|^2.
$$

---

## 4. Temporal Jacobian and regulatory logic

Once $u_\theta$ is trained, we can compute its **Jacobian**
with respect to expression coordinates:

$$
J(x,t) = \frac{\partial u_\theta(x,t)}{\partial x}.
$$

Intuitively:
- $J_{ij}(x,t) > 0$ → *gene i up-regulates gene j* at time $t$
- $J_{ij}(x,t) < 0$ → *inhibition*

Sampling $J(x,t)$ along pseudotime yields a **tensor**
$$
\mathcal{J} \in \mathbb{R}^{d\times d\times T},
$$
representing the *time-varying regulatory landscape*.

---

## 5. Archetype decomposition

To extract interpretable programs, scQDiff factorizes the Jacobian tensor as

$$
\boxed{
\mathcal{J} \;\approx\;
\sum_{r=1}^{R}
a_r \otimes b_r \otimes c_r,
}
$$

where
- $a_r \in \mathbb{R}^d$: source-gene loading (regulators)  
- $b_r \in \mathbb{R}^d$: target-gene loading (effected genes)  
- $c_r(t) \in \mathbb{R}$: temporal activation of archetype $r$

Each triplet $(a_r,b_r,c_r)$ defines a **regulatory archetype**—
an interpretable gene-to-gene program that turns on and off along
the cellular trajectory.

When **CellRank** fate probabilities $P_{\text{fate}}$
are available, the Jacobians are **weighted** by these probabilities to
produce *fate-conditioned* archetypes.

---

## 6. Simulation of trajectories

Once $u_\theta(x,t)$ is learned, scQDiff can *simulate* cell-state evolution:

$$
\frac{dX_t}{dt} = u_\theta(X_t,t)
\quad\text{(deterministic)}, \qquad
dX_t = u_\theta\,dt + \sqrt{2\beta}\,dW_t
\quad\text{(stochastic)}.
$$

- **Forward integration** $t:0\!\to\!1$: naïve → perturbed trajectory  
- **Reverse integration** $t:1\!\to\!0$: reprogramming path  
- **Archetype modulation:**
  $$
  u'(x,t) = u_\theta(x,t) + \lambda_r\,(A_r x)\,c_r(t),
  $$
  allowing virtual perturbation of a specific regulatory program.

These simulations generate synthetic datasets or visualize how altering a
particular archetype reshapes the fate landscape.

---

## 7. Relation to classical methods

| Framework | Focus | scQDiff connection |
|------------|--------|-------------------|
| Optimal Transport (OT) | Cost-efficient mass transport | SB is entropy-regularized OT |
| Score-based Diffusion | Generative modeling via ∇log ρ | scQDiff learns time-conditional scores |
| scVelo | RNA velocity ODE | scQDiff generalizes it to continuous stochastic drift |
| CellRank | Fate probability graphs | scQDiff weights Jacobians by fate probabilities |

---

## 8. References

- Schrödinger, E. (1931). *Über die Umkehrung der Naturgesetze.*  
- Léonard, C. (2013). *A survey of the Schrödinger problem and some of its connections with optimal transport.*  
- De Bortoli et al. (2021). *Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling.*  
- Bergen et al. (2020). *scVelo: dynamical modeling of RNA velocity.*  
- Lange et al. (2022). *CellRank: directed single-cell fate mapping.*

---

*Prepared by the BYU Genomics & Bioinformatics Core — scQDiff development team.*
