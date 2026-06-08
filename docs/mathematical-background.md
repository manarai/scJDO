# Mathematical background

scJDO models a cell as a point in latent space evolving over pseudotime. Its central object is not only the cell trajectory, but the local differential operator governing how perturbations grow or decay around that trajectory.

## Stochastic dynamics

A cell state is represented as

$$
X_t \in \mathbb{R}^d,
$$

with pseudotime $t \in [0,1]$. scJDO uses stochastic dynamics of the form

$$
dX_t = f(X_t,t)\,dt + \sqrt{2\beta}\,dW_t,
$$

where $f(x,t)$ is the regulatory drift field, $\beta$ controls stochasticity, and $W_t$ is Brownian motion.

## Local regulatory operators

The key object is the Jacobian of the drift,

$$
J(x,t) = \frac{\partial f(x,t)}{\partial x}.
$$

This operator describes the local evolution of small perturbations. Positive real eigenvalues indicate directions where perturbations can grow, negative real eigenvalues indicate stable or buffered directions, and near-zero values indicate fragile or plastic regimes.

| Eigenvalue regime | Interpretation |
|---|---|
| $\lambda < 0$ | Stable, buffered state |
| $\lambda \approx 0$ | Fragile or plastic state |
| $\lambda > 0$ | Locally unstable, fate-controlling direction |

## Temporal Jacobian tensor and archetypes

scJDO aggregates per-cell Jacobians across pseudotime using **adaptive
Gaussian kernel windowing**:

$$
\bar J(\tau;h) \;=\; \frac{\sum_i w_i(\tau)\,J_i}{\sum_i w_i(\tau)},
\qquad w_i(\tau) \;=\; \exp\!\Big(-\tfrac{(\tau_i-\tau)^2}{2 h^2}\Big),
$$

with bandwidth $h$ selected automatically by maximising

$$
S(h) \;=\; R(h)\cdot C(h)\cdot L(h)
$$

— bootstrap reproducibility $R$ (mean pairwise Pearson correlation of
$\lambda_{\max}$ curves across cell-resamples), peak contrast
$C(h)=\max_\tau\lambda_{\max}-\mathrm{median}_\tau\lambda_{\max}$, and peak
localisation $L(h)=\max_\tau\lambda_{\max}/\mathrm{FWHM}$ — subject to an
effective-sample-size floor $n_\mathrm{eff}(\tau)=(\sum_i w_i)^2/\sum_i w_i^2 \ge n_\mathrm{min}$.
The kernel grid is a continuous temporal resolution, not a count of
independent observations, and downstream curves carry pointwise bootstrap
uncertainty. An opt-in locally-adaptive bandwidth $h(\tau)$ (k-th nearest
cell in pseudotime) is available for trajectories with non-uniform density.
The pre-v0.4 fixed-window scheme remains available via
`windowing='fixed'`.

The resulting tensor $\mathcal{J}\in\mathbb{R}^{T\times D\times D}$ is then
approximated as a sum of reusable operator archetypes with time-dependent
activations:

$$
\mathcal{J} \approx \sum_{k=1}^{K} A_k c_k(t),
$$

where $A_k$ is an operator archetype (signed) and $c_k(t)\ge 0$ is its
temporal activation (semi-NMF). The non-negative activation profiles
identify when each regulatory mode is active.

## Endpoint-constrained dynamics

When source and target populations are defined, scJDO can use a Schrödinger Bridge formulation. The bridge seeks a stochastic process that transports the source distribution to the target distribution while remaining close to a reference process. scJDO then applies the same Jacobian and archetype analysis to the learned forward and backward dynamics.
