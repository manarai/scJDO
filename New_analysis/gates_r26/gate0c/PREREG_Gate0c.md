# PREREG — Gate 0c: consensus archetypes (cNMF-style)

**Status**: pre-registered before compute. Frozen against `9d4e5ab` (Gate 0b report).

## Question

Which of `vel_scale ∈ {0.0, 2.0}` gives more stable consensus archetype components across seeds? The default for Gates 1–2 will be the winner. Consensus components (recurring in ≥ 80% of runs) become the ONLY archetype output used downstream.

## Design (frozen)

- **Substrate**: marrow Ery branch cells (r24 pinned pipeline, 1,151 cells, X_fa 30-D).
- **Configurations**: two arms —
  - Arm V0: `vel_scale = 0.0`, `bias_strength = 1.5`, `hidden = 256`, `depth = 4`, `n_epochs = 5000`, `n_archetypes = 5`, `grid_size = 200`
  - Arm V2: `vel_scale = 2.0`, `bias_strength = 1.5`, everything else the same
- **Replicates**: R = 10 seeds per arm (seeds 0..9). Each replicate is a full `fit_drift_branches` fit.
- **Consensus clustering**: for each arm separately —
  1. Extract patterns from all R replicates → 50 patterns (10 × 5), each (30, 30).
  2. Flatten each pattern to 900-D vector, sign-invariant normalise (unit L2, then flip sign so leading element is positive).
  3. Compute pairwise |cosine| between all 50 patterns → 50×50 similarity matrix.
  4. Agglomerative clustering (single linkage on 1 − |cosine|) into K_cluster = 5 target clusters (matching the per-replicate K = 5).
  5. For each cluster: count the number of distinct replicates it draws from (call this the recurrence R_cluster).
  6. A cluster **passes consensus** if R_cluster ≥ 8 (i.e. ≥ 80 % of the 10 replicates contributed at least one pattern to it).
  7. Consensus centroid = mean of all patterns in a passing cluster, renormalised.
- **Metric per arm**:
  - **n_consensus_clusters** = number of clusters passing the 80 % rule.
  - **mean intra-cluster cosine** = mean |cos| within each passing cluster, averaged over passing clusters.
- **Default choice (frozen)**: whichever arm has HIGHER `n_consensus_clusters`. If tie, whichever has HIGHER mean intra-cluster cosine.

## Frozen success threshold

- Report `n_consensus_clusters` and mean intra-cluster cosine per arm.
- Pick default per the deterministic rule above (higher n_consensus; tiebreak by cosine).
- Consensus components used downstream = the passing clusters of the chosen arm.

## Stop rules

- If **both arms have `n_consensus_clusters = 0`** → the archetype output is not stable enough for downstream use in any configuration. Report; do not stop the gate chain (Gates 1/2 will use single-fit outputs with seed-diagnostic caveats). This outcome is a scientific finding, not a runtime error.
- If Gate 0c produces stable consensus → Gates 1/2 use those consensus components as the pre-declared scJDO features.

## Reported quantities

- Per-arm: n_consensus_clusters, per-cluster recurrence R_cluster, per-cluster intra-cluster mean |cos|, list of consensus centroids saved as npz.
- Chosen default arm.
- Sanity: how many of the 5 within-replicate archetypes are within the same passing cluster (should be ≤ 1 for a well-decomposed problem — otherwise the semi-NMF is returning duplicate patterns).

## Expected wall clock

10 seeds × 2 arms × ~4 min per fit = ~80 min compute. Consensus clustering: <1 min.

## Not part of the gate

- Whether these consensus components carry biological meaning (Gates 1/2).
- Alternative K values. K = 5 matches the Fig 3 pipeline; changing K would open a separate identifiability question.
