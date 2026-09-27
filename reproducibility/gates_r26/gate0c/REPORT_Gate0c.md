# REPORT — Gate 0c: consensus archetypes (cNMF-style)

**Prereg**: `PREREG_Gate0c.md` (frozen `dbd67e4`, before compute).
**Compute**: ~88 min wall (Arm V0 46.3 min for 10 seeds; Arm V2 42.5 min for 10 seeds; clustering < 1 s per arm).

## Verdict — DEFAULT = V0 (vel_scale = 0.0); 1 consensus mode per arm; 4 archetypes per fit are seed-specific.

Per the frozen rule (higher `n_consensus_clusters`, tiebreak by higher mean intra-cluster |cos|):

| Arm | vel_scale | n_consensus_clusters (R ≥ 8/10) | mean intra |cos| |
|:---:|:---:|:---:|:---:|
| V0  | 0.0 | **1** | **0.856** |
| V2  | 2.0 | 1 | 0.844 |

Tie on `n_consensus_clusters`; tiebreak on intra-cluster cosine → **V0 chosen**.

## Per-arm cluster breakdown

Both arms produce a single passing cluster of the same shape:
- 1 cluster with recurrence R = 10 (every seed contributes at least one pattern to it), containing all patterns not otherwise assigned.
- 4 singleton clusters (each a single pattern from a single seed).

**Arm V0** (vel_scale = 0.0):
```
label   R_cluster   n_members   intra_cos
  1        10         46          0.856
  2         1          1          1.000
  3         1          1          1.000
  4         1          1          1.000
  5         1          1          1.000
```

**Arm V2** (vel_scale = 2.0):
```
label   R_cluster   n_members   intra_cos
  1        10         46          0.844
  2         1          1          1.000
  3         1          1          1.000
  4         1          1          1.000
  5         1          1          1.000
```

## Sanity flag (pre-declared)

The PREREG asked to check "how many of the 5 within-replicate archetypes are within the same passing cluster (should be ≤ 1 for a well-decomposed problem — otherwise the semi-NMF is returning duplicate patterns)."

**Both arms fail this sanity check: all 10 replicates contribute multiple patterns to the passing cluster** (`dup_within_rep_per_passing = [10]` for both). Within each fit, ~4-5 of the K = 5 archetypes are near-duplicates of one another; only 0-1 pattern per fit is distinct enough to fall out as a singleton.

## Interpretation

The scJDO Jacobian-archetype decomposition, in this configuration, does not yield 5 independently reproducible modes. It yields **one** mode that recurs across seeds with intra-cluster |cos| ≈ 0.85, plus a residue that varies per seed. The prior arm (V2) and the no-prior arm (V0) both collapse to this same pattern; the velocity prior does not add or subtract robustness at the archetype-decomposition level.

**What this constrains for downstream claims:**
- The "5 archetypes" reported in Fig 3-style panels are not 5 stable objects. There is one stable direction (the consensus mode) plus 4 seed-dependent decompositional slack.
- Any manuscript claim that ties biological meaning to specific individual archetypes beyond the consensus mode is not defensible from a single fit; only the consensus mode is defensible across seeds.
- The velocity prior (vel_scale=2 vs 0) does not change decomposition stability in this substrate.

**What this does not say:**
- Whether the single consensus mode itself carries fate-predictive or perturbation-response signal (Gates 1 and 2 test this).
- Whether the collapse to K_eff = 1 is a property of K=5 being too high for this branch's rank, or a property of the semi-NMF procedure, or both. Not part of the frozen gate.

## Downstream action per prereg

- **Default arm for Gates 1-2: V0 (vel_scale = 0.0)**.
- **Consensus archetype feature for Gates 1-2: the single passing cluster centroid** (saved to `consensus_centroids_V0.npz`, position of `passing_mask=True` entry). The other K-1 = 4 archetypes are seed-specific noise and are NOT part of the pre-declared scJDO feature bundle.
- Per-cell scJDO features for Gate 1 shrink to: **Re(λ_max) per cell**, **leading-Jacobian-direction projection**, and **1 consensus-archetype activation** (not 5).
- No manuscript edits based on Gate 0c alone.

## Deliverables

- Prereg: `PREREG_Gate0c.md` (`dbd67e4`)
- Runner: `run_gate0c.py`
- Machine-readable summary: `gate0c_summary.json`
- Consensus centroids: `consensus_centroids_V0.npz`, `consensus_centroids_V2.npz`
- Runtime log: `run_gate0c.log`

**Next**: Gate 1 (LARRY early fate prediction with the reduced scJDO feature bundle).
