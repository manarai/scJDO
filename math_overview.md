scIDiff — Mathematical Overview

Time-resolved regulatory operators for single-cell dynamics

scIDiff (single-cell Inference of Differential operators) is an operator-centric framework for learning the time-dependent dynamical rules that govern cellular state transitions. Rather than only reconstructing trajectories, scIDiff infers the local Jacobian operators of a learned drift field and uses their temporal structure to reveal conserved regulatory programs (“operator archetypes”) and decision-sensitive control modes.

Note: RNA velocity and Schrödinger Bridge constraints can be used as optional priors when learning the drift field, but all downstream operator and archetype analysis is independent of these choices.
1. Stochastic model of cell-state dynamics

Let 
X
(
t
)
∈
R
d
X(t)∈R 
d
  denote the latent cellular state at pseudotime 
t
∈
[
0
,
1
]
t∈[0,1].

Cell dynamics are modeled as the stochastic differential equation

d
X
(
t
)
=
f
(
X
(
t
)
,
t
)
 
d
t
+
2
β
 
d
W
(
t
)
dX(t)=f(X(t),t)dt+ 
2β
​	
 dW(t)
where

f
(
x
,
t
)
f(x,t) is the time-dependent drift field,
β
>
0
β>0 controls stochasticity,
W
(
t
)
W(t) is a standard Brownian motion.
The drift 
f
(
x
,
t
)
f(x,t) represents the instantaneous regulatory program driving cellular motion in latent state space.

2. Density evolution

Let 
ρ
(
t
,
x
)
ρ(t,x) denote the probability density of 
X
(
t
)
X(t).
It evolves according to the Fokker–Planck equation

∂
ρ
∂
t
=
−
∇
⋅
(
ρ
f
)
+
β
Δ
ρ
.
∂t
∂ρ
​	
 =−∇⋅(ρf)+βΔρ.
scIDiff learns 
f
(
x
,
t
)
f(x,t) so that this stochastic flow explains the observed single-cell distribution over pseudotime.

3. Local regulatory operators (Jacobians)

The central object in scIDiff is the time-resolved Jacobian operator

J
(
x
,
t
)
=
∇
x
f
(
x
,
t
)
.
J(x,t)=∇ 
x
​	
 f(x,t).
This governs the linearized perturbation dynamics

d
(
δ
X
)
=
J
(
X
,
t
)
 
δ
X
 
d
t
+
2
β
 
d
W
(
t
)
.
d(δX)=J(X,t)δXdt+ 
2β
​	
 dW(t).
The eigenvalues of 
J
(
x
,
t
)
J(x,t) define:

Eigenvalue sign	Interpretation
Negative	stable (buffered states)
Near-zero	plastic (fragile states)
Positive	unstable (fate-deciding control directions)
These operators quantify stability, sensitivity, and commitment directly.

4. Temporal Jacobian tensor

Evaluating Jacobians along pseudotime gives

J
(
t
1
)
,
  
J
(
t
2
)
,
  
…
,
  
J
(
t
T
)
.
J(t 
1
​	
 ),J(t 
2
​	
 ),…,J(t 
T
​	
 ).
Stacking them forms the temporal Jacobian tensor

T
∈
R
T
×
d
×
d
,
T∈R 
T×d×d
 ,
which summarizes how regulatory sensitivity evolves during differentiation.

5. Operator archetypes

scIDiff factorizes the temporal tensor as

T
(
t
)
≈
∑
k
=
1
K
a
k
(
t
)
 
A
k
,
T(t)≈ 
k=1
∑
K
​	
 a 
k
​	
 (t)A 
k
​	
 ,
where

A
k
A 
k
​	
  are operator archetypes (
d
×
d
d×d matrices),
a
k
(
t
)
a 
k
​	
 (t) are their time-dependent activations.
Sequential and concurrent archetype activation defines a conserved regulatory grammar of cell fate.

6. Control-relevant modes

Eigenvectors of 
A
k
A 
k
​	
  with positive eigenvalues are unstable modes: perturbations along these directions are amplified.

When projected into gene space, these modes identify control-relevant transcriptional programs.

7. RNA velocity as an optional drift prior

RNA velocity can guide drift learning via a reference field

b
(
x
,
t
)
=
λ
 
g
(
t
)
 
w
(
x
)
 
v
^
(
x
)
,
b(x,t)=λg(t)w(x) 
v
^
 (x),
where

v
^
(
x
)
v
^
 (x) is RNA velocity,
w
(
x
)
w(x) is a confidence weight,
g
(
t
)
g(t) is a temporal gate,
λ
λ controls strength.
The learned drift is

f
(
x
,
t
)
=
b
(
x
,
t
)
+
u
θ
(
x
,
t
)
.
f(x,t)=b(x,t)+u 
θ
​	
 (x,t).
RNA velocity improves directional alignment but does not alter Jacobian or archetype analysis.

8. Endpoint-constrained dynamics (Schrödinger Bridge, optional)

If start and end populations are known, scIDiff enforces them by minimizing control energy

min
⁡
u
  
E
[
∫
∥
u
(
X
,
t
)
∥
2
 
d
t
]
u
min
​	
 E[∫∥u(X,t)∥ 
2
 dt]
subject to

∂
ρ
∂
t
=
−
∇
⋅
(
ρ
 
(
b
+
u
)
)
+
β
Δ
ρ
,
X
(
0
)
∼
ρ
0
,
  
X
(
1
)
∼
ρ
1
.
∂t
∂ρ
​	
 =−∇⋅(ρ(b+u))+βΔρ,X(0)∼ρ 
0
​	
 ,X(1)∼ρ 
1
​	
 .
This changes only the drift field; Jacobians and archetypes are computed the same way.
