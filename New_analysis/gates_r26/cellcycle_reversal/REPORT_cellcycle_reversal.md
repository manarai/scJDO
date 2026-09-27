# REPORT — Cell-cycle reversal calibration

**Finding**: vanilla scJDO with the Gate-0c-selected default `vel_scale = 0` does NOT encode pseudotime direction as tangential drift on synthetic cyclic data. Reversing pseudotime does not reverse the inferred rotation — because the vanilla DSM-only training produces near-zero tangential drift in either direction.

**Prereg**: `PREREG_cellcycle_reversal.md` (frozen `78b1256`, before compute).
**Compute**: ~20 min (forward 3 × ~160 s + reverse 3 × ~245 s + figure).

## Numbers

Angular drift = per-cell drift dotted with the true circle-tangent direction (`(−sin φ, cos φ)` in latent 2-D, embedded via the fixed random orthogonal projection into the 30-D ambient space):

| Arm | Mean angular drift (3-seed) | Spread across seeds |
|:---:|:---:|:---:|
| forward (`τ = φ / 2π`) | **−0.00017** | 2.3e−5 |
| reverse (`τ = 1 − φ / 2π`) | **−0.00017** | 1.5e−5 |
| |fwd + rev| / |fwd| | **2.03** (should be ≪ 1 for antisymmetric reversal) | — |

Per-seed values (forward, reverse):
- seed 0: (−0.00020, −0.00019)
- seed 1: (−0.00015, −0.00015)
- seed 2: (−0.00016, −0.00018)

Per-cell drift magnitude was NOT tiny — `|v|` median ≈ 2.68 in both arms. The drift is large but essentially all NON-tangential (radial or transverse to the cycle).

## Why the demo fails on vanilla scJDO

`fit_drift` with `vel_scale = 0` trains DriftField via denoising score matching (DSM) on the snapshot density at each pseudotime bin. The DSM objective learns the SCORE function `∇ log p(x, t)` — a density gradient, not a temporal flow. On a stationary uniform-on-circle distribution:
- The score is nearly zero everywhere on the circle (density is uniform along the ring).
- What the trained drift models is the RADIAL restoring force pulling cells back onto the ring after Gaussian perturbation, not the ROTATION around it.
- Reversing pseudotime relabels bins but does not change what the DSM objective learns from the (identical, stationary) density.

This is a substantive scJDO calibration statement: the fitted field does not respect pseudotime as a directional flow on cyclic geometries when the underlying distribution is stationary. Any "rotation" a downstream user infers from Jacobian analysis of a cyclic trajectory needs a velocity supervision (e.g. `vel_scale > 0` with a tangential V_ref, or the Gate 2 redo velocity-matching loss) — it is not produced by the default DSM training.

## Figure

`cellcycle_reversal.pdf` and `cellcycle_reversal.png`: 2 panels of the latent (cos φ, sin φ) scatter for the forward and reverse arms, coloured by per-cell tangent-projected drift on a signed diverging colourmap. Overlay quiver arrows at 60 phases. In both panels the colour is essentially neutral (drift ≈ 0) and the quiver arrows do not consistently point in either direction. Titles include the mean angular drift and 3-seed spread.

## What the paper can say from this calibration

- scJDO's default DSM training extracts density-based operators (Jacobian of the score); it does NOT extract direction-of-flow from pseudotime alone on cyclic substrates.
- Any figure that positions scJDO as recovering "the direction of the cell cycle" from pseudotime without external velocity supervision is not supported.
- Correctly recovering direction on cyclic trajectories requires supervision (external velocities, oriented boundary conditions, or explicit velocity-matching loss like Gate 2 redo's L arm), not the default settings.

## Deliverables

- Prereg: `PREREG_cellcycle_reversal.md` (`78b1256`).
- Runner: `run_cellcycle_reversal.py` (single deterministic entry).
- This report.
- `cellcycle_reversal.pdf` + `cellcycle_reversal.png` — the 2-panel figure.
- `summary.json` — machine-readable numbers.
- Log: `run.log`.

## Downstream note

Per the Gate 3 downstream chain, item 3 (Gate 0d — sign-preserving one-to-one consensus with fold-wise refits, shuffled-time null, precision-matrix baseline) is next.
