scIDiff — Mathematical Overview
Time-resolved regulatory operators for single-cell dynamics
scIDiff (single-cell Inference of Differential operators) is an operator-centric framework for learning the time-dependent dynamical rules that govern cellular state transitions. Rather than only reconstructing trajectories, scIDiff infers the local Jacobian operators of a learned drift field and uses their temporal structure to reveal conserved regulatory programs (“operator archetypes”) and decision-sensitive control modes.
RNA velocity and Schrödinger Bridge constraints can be used as optional priors when learning the drift field, but all downstream operator and archetype analysis is independent of these choices.
1. Stochastic model of cell-state dynamics
Let X(t) ∈ R^d denote the latent cellular state at pseudotime t ∈ [0, 1].
Cell dynamics are modeled as the stochastic differential equation
dX(t) = f(X(t), t) dt + sqrt(2β) dW(t)
where
f(x,t) is the time-dependent drift field
β > 0 controls stochasticity
W(t) is Brownian motion
The drift f(x,t) represents the instantaneous regulatory program driving cellular motion in latent state space.
2. Density evolution
Let ρ(t, x) denote the probability density of X(t).
It evolves according to the Fokker–Planck equation
∂ρ/∂t = − ∇·(ρ f) + β Δρ
scIDiff learns f(x,t) so that this stochastic flow explains the observed single-cell distribution over pseudotime.
3. Local regulatory operators (Jacobians)
The central object in scIDiff is the time-resolved Jacobian operator
J(x,t) = ∇x f(x,t)
This governs the linearized perturbation dynamics
d(δX) = J(X,t) δX dt + sqrt(2β) dW(t)
The eigenvalues of J(x,t) define:
Negative → stable (buffered states)
Near-zero → plastic (fragile states)
Positive → unstable (fate-deciding control directions)
These operators quantify stability, sensitivity, and commitment directly.
4. Temporal Jacobian tensor
Evaluating Jacobians along pseudotime gives
J(t1), J(t2), …, J(tT)
Stacking them forms the temporal Jacobian tensor
T ∈ R^(T × d × d)
which summarizes how regulatory sensitivity evolves during differentiation.
5. Operator archetypes
scIDiff factorizes the temporal tensor as
T(t) ≈ Σ_{k=1..K} a_k(t) A_k
where
A_k are operator archetypes (d × d matrices)
a_k(t) are their time-dependent activations
Sequential and concurrent archetype activation defines a conserved regulatory grammar of cell fate.
6. Control-relevant modes
Eigenvectors of A_k with positive eigenvalues are unstable modes: perturbations along these directions are amplified.
When projected into gene space, these modes identify control-relevant transcriptional programs.
7. RNA velocity as an optional drift prior
RNA velocity can guide drift learning via a reference field
b(x,t) = λ g(t) w(x) v̂(x)
where
v̂(x) is RNA velocity
w(x) is a confidence weight
g(t) is a temporal gate
λ controls strength
The learned drift is
f(x,t) = b(x,t) + uθ(x,t)
RNA velocity improves directional alignment but does not alter Jacobian or archetype analysis.
8. Endpoint-constrained dynamics (Schrödinger Bridge, optional)
If start and end populations are known, scIDiff enforces them by minimizing control energy
min_u E ∫ ||u(X,t)||² dt
subject to
∂ρ/∂t = −∇·(ρ(b+u)) + β Δρ
X(0) ~ ρ0 , X(1) ~ ρ1
This changes only the drift field; Jacobians and archetypes are computed the same way.
9. Summary
scIDiff infers time-resolved regulatory operators from single-cell data.
By learning a drift field, computing its Jacobian, and decomposing the resulting temporal tensor into archetypes, scIDiff reveals a conserved grammar of fate decisions and identifies control-relevant unstable modes.
RNA velocity and Schrödinger Bridge provide optional guidance, but the core object of scIDiff is the time-varying regulatory operator.
