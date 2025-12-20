# scIDiff — Minimal Mathematical Core

**scIDiff** (*single-cell Inverse Diffusion*) models cellular state transitions as a
stochastic process guided by **RNA velocity** and refined via the
**Schrödinger Bridge** formulation of optimal transport.

---

## State dynamics

Cell states evolve according to the stochastic differential equation

$$
dX_t = f(X_t,t)\,dt + \sqrt{2\beta}\,dW_t,
$$

where $X_t \in \mathbb{R}^d$ is the cell state, $f(x,t)$ is the drift field,
and $\beta$ controls stochasticity.

The corresponding density $\rho_t(x)$ satisfies the Fokker–Planck equation

$$
\partial_t\rho_t
=
-\nabla\!\cdot(\rho_t f)
+
\beta\,\Delta\rho_t.
$$

---

## RNA velocity as reference drift

RNA velocity defines a biologically grounded **reference drift**

$$
b(x,t)
=
\lambda \cdot g(t) \cdot w(x) \cdot \hat{v}(x),
$$

where $\hat{v}(x)$ is interpolated RNA velocity, $w(x)$ is a confidence weight,
$g(t)$ is a temporal gating function, and $\lambda$ scales velocity magnitude.

---

## Schrödinger Bridge formulation

The total drift is decomposed as

$$
f(x,t) = b(x,t) + u_\theta(x,t),
$$

where $u_\theta(x,t)$ is a learned correction drift.

The Schrödinger Bridge objective learns the **minimal correction**
required to transport the initial distribution $\rho_0$ to the terminal
distribution $\rho_1$:

$$
\min_{u_\theta}
\;
\mathbb{E}\!\int_0^1 \|u_\theta(X_t,t)\|^2\,dt,
\quad
\text{s.t. } X_0 \sim \rho_0,\; X_1 \sim \rho_1.
$$

---

## Key interpretation

- RNA velocity provides **local biological directionality**
- Schrödinger Bridge enforces **global distributional consistency**
- $u_\theta$ learns the **minimal deviation** from velocity needed to satisfy fate constraints

**scIDiff unifies biological kinetics and probabilistic optimal transport
in a single dynamical system.**

---

*BYU Genomics & Bioinformatics Core — scIDiff Development Team*
