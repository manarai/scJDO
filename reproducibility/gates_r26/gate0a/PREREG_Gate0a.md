# PREREG — Gate 0a: bias-plumbing diagnostic

**Status**: pre-registered before compute. No fits have been executed yet against this protocol; all thresholds and decisions below are frozen.

**Base**: `scjdo-v56-submission-20260922-r25` @ `cbfd591`.

## Question

r24 Task A+B showed A1 (`vel_scale=2.0`, `bias_strength=1.5`) and B1 (`vel_scale=2.0`, `bias_strength=0.0`) produced identical aggregate reproducibility numbers (both 0.437 eigvec-cos / 0.292 top-15 Jaccard within-config). r25 trace showed `_biased_velocity(bias=0)` == `_pseudotime_velocity` (max diff 0) and `_biased_velocity(bias=1.5)` differs (mean per-cell cos 0.84). But r25 did not diff what `DriftField.forward` actually consumes at run-time or what the trained weights end up as.

**Gate 0a resolves the ambiguity**: is the bias parameter delivering different `v_hat` into `DriftField.forward`, and does that produce different trained weights?

## Prereg design

- **Substrate**: marrow Ery branch cells (1,151 cells, r24 pinned pipeline: MAGIC-imputed Palantir 1.4.4 preprocessing).
- **Configurations**: A1 (`vel_scale=2.0`, `bias_strength=1.5`) and B1 (`vel_scale=2.0`, `bias_strength=0.0`).
- **Seeds**: 3 seeds (0, 1, 2). Every fit reported.
- **Instrumentation**: monkey-patch `DriftField.forward` at fit time to capture the `v_hat` tensor argument on the first forward call of the last training epoch (guaranteed to be after any input caching). Save `v_hat.detach().cpu().numpy()` per seed per config.
- **Weight diff**: after each fit, save `model.state_dict()` and diff parameter-by-parameter between A1 seed s and B1 seed s.
- **Fit knobs** (fixed): `depth=4`, `n_epochs=5000`, `n_archetypes=5`, `grid_size=200`, everything else at Fig 3 pipeline defaults.

## Frozen predictions (three exhaustive cases, one will match)

| Case | v_hat identical | weights identical | Interpretation | Downstream action |
|:---:|:---:|:---:|---|---|
| **Case 1** | YES | YES | Bias never reaches the model at all. Bug. | Fix in-code, add regression test, CHANGELOG entry for v0.3.0. Retract every manuscript sentence about "terminal-state guidance". |
| **Case 2** | NO | YES | v_hat differs but training dynamics converge to bit-identical weights. Explainable only by unusual optimizer path. | Investigate why. Provisionally treat bias as functionally inert. |
| **Case 3** | NO | NO | Bias reaches model and produces different weights, but seed dominates basin selection at the top-K display level (as r24 empirically observed at seeds 1/2 giving basin-identical top-5 despite different weights). | Document that bias is real but non-load-bearing for basin selection on this branch. Not a bug. Manuscript "terminal-state guidance" claim needs softening to reflect what bias empirically accomplishes. |

## Success threshold (frozen)

- Report the case that actually matches, verbatim, with the two specific numerics (max |Δv_hat| per seed; max |Δparam| per seed).
- Fail the gate if the instrumentation itself fails (v_hat capture returns None; state_dict save errors).

## Success does not mean "any of the three cases wins"

The gate is designed to distinguish which of Cases 1-3 is true. Any of the three is a valid outcome; the gate is a **diagnosis**, not a filter. The gate PASSES when we know which case we're in.

## Stop rules

- Case 1: fix the bug (r-tag; NOT counted toward Gates 1-2 timeline). After fix, all Gate 0a/b/c/1/2 outputs must be regenerated on the fixed code.
- Case 2 or Case 3: proceed to Gate 0b immediately. No manuscript edits based on Gate 0a alone.

## Report deliverable

`REPORT_Gate0a.md` covering:
- Case determination (1/2/3) with numerics.
- If Case 1: pointer to the fix commit + regression test.
- If Case 2 or 3: table of per-seed max |Δv_hat| and max |Δparam|.
- Explicit named lookup for the r24 A1/B1 aggregate-metric equivalence: was it a coincidence of basin-selection patterns (Case 3) or bit-identical fits (Cases 1 or 2)?
