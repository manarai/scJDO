"""
G-9 — Full ladder re-run on the best per-τ configuration.

RUNS ONLY IF G-7 criterion 3 (signal-subspace cos with the analytic drift
in the pre-saddle regime) materially improves over the current
uniform-batching baseline of -0.44 to -0.58.

Expected reruns (on the winning batch_mode='per_tau', window=<best>):
  - signal-subspace cos test (per-cell distribution, 3 seeds).
  - crossing detection on lifted truth AND trained field, 4 bandwidths.
  - scale-stability bandwidth selection.
  - Paul15 all + per-branch, with the new sampler.
  - Multiome branches, if the fitted-drift matches the manuscript's
    fitting protocol (Palantir pseudotime + FA embedding).
  - Circular-shift permutation null on any observed crossings.

Kept as a template — not invoked unless G_7 evaluate script emits a
`crit3_improved: True` flag.
"""

from __future__ import annotations

# Deliberately empty scaffold. Populate after inspecting G-5/G-6 results.
if __name__ == "__main__":
    print("G-9 is a conditional rerun; do not invoke directly.")
    print("Populate this file only after G-7 crit 3 improvement is confirmed.")
