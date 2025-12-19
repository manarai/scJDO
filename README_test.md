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
(Full README content exactly as provided above)
