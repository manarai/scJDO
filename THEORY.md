# 🌌 THEORY: Schrödinger Bridge Dynamics in scIDiff

## Controlled Stochastic Dynamics

scIDiff models cellular dynamics as:

dX_t = (b(X_t,t) + u_θ(X_t,t)) dt + √(2β) dW_t

where:
- b(x,t): biological reference drift (RNA velocity)
- u_θ(x,t): learned control drift
- β: diffusion strength

## Schrödinger Bridge Objective

Among all stochastic processes transporting ρ₀ → ρ₁, scIDiff finds the one minimizing:

E[ ∫₀¹ ||u_θ(X_t,t)||² / (2β) dt ]

This yields the most probable dynamics consistent with biological priors.

## RNA Velocity as Prior

RNA velocity defines a soft reference drift:

b(x,t) = g(t) · w(x) · v̂(x)

allowing local transcriptional kinetics to guide global dynamics.

## Forward vs Reverse Dynamics

Forward drift:
f(x,t) = b(x,t) + u_θ(x,t)

Reverse drift:
f_rev = f − 2β ∇ₓ log ρ_t

The asymmetry quantifies irreversibility and reprogramming cost.

## Entropy Production

Entropy production rate:

Ṡ(t) = E[ ||f − f_rev||² / (2β) ]

High values indicate committed or exhausted states.
