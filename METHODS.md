# 🔬 METHODS: Training and Analysis in scIDiff

## Model Architecture
- Neural network parameterization of u_θ(x,t)
- Time-conditioned drift model
- Density estimation via score matching or particle methods

## Training Procedure
1. Sample trajectories from reference drift
2. Optimize control cost subject to endpoint constraints
3. Enforce marginal consistency via SB loss

## Jacobian Computation

Temporal Jacobian:
J(t) = ∂f / ∂x

Computed via automatic differentiation of the learned drift.

## Archetype Decomposition

At each time t:
J(t) = U Σ Vᵀ

- U: regulatory gene modules
- V: communication or signaling modes

## Outputs
- Temporal GRNs
- Forward/reverse archetypes
- Entropy production curves

## Practical Notes
- Works with or without RNA velocity
- Compatible with Scanpy / scVelo pipelines
- Supports latent embeddings or gene space
