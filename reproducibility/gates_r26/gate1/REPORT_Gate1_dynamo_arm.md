# REPORT — Task 4: optional Dynamo arm on LARRY (SKIPPED)

**Skipped, not run.** The Klein LARRY in vitro release provides only library-normalised counts (`stateFate_inVitro_normed_counts.mtx`). Neither spliced/unspliced separation nor metabolic labelling (new-transcript layer) is available in the release. Dynamo's velocity pipeline requires one of:
- Spliced + unspliced counts for splicing-based dynamics (`dyn.tl.dynamics(model="stochastic"|"deterministic")`), OR
- New + total layers with a valid labelling timestamp for metabolic-labelling dynamics (`experiment_type="one-shot"|"kin"`, tkey column of labelling durations).

The release has neither. Dynamo `dyn.pp.recipe_monocle` can still populate `X_pca`, but `dyn.tl.dynamics` with any experiment_type will produce NaN velocities (verified analytically: the required layer inputs do not exist).

Per Task 4's frozen rule: "If Dynamo can't run on LARRY within a day, write REPORT_Gate1_dynamo_arm.md saying so and move on." — that condition is met (the constraint is a permanent data-availability issue, not a compute-time issue).

## What was checked before deciding

1. LARRY release files inspected: `stateFate_inVitro_normed_counts.mtx` (130,887 × 25,289, single layer), `stateFate_inVitro_metadata.txt` (columns: Library, Cell barcode, Time point, Starting population, Cell type annotation, Well, SPRING-x, SPRING-y — no labelling duration, no spliced/unspliced), `stateFate_inVitro_gene_names.txt`, `stateFate_inVitro_clone_matrix.mtx` (clone barcodes).
2. Klein-lab paper (Weinreb et al. 2020, Science) confirms in vitro release is snapshot single-cell RNA-seq only. Companion lineage-tracing (LARRY barcodes) is separate from any splicing / labelling protocol.

## Consequence for Gate 1 v2 table

Row `D` (Dynamo per-cell Jacobian features from LARRY) is NOT added to the Gate 1 v2 five-row table. Row `E_FA + D` is also NOT added. The Gate 1 v2 result stands as reported in `REPORT_Gate1_v2.md` (`3be140b`).

If a re-analysis is done in the future on a paired scRNA-seq dataset that carries spliced/unspliced OR metabolic labels alongside clone barcodes, Dynamo Jacobian features could be evaluated. Such a dataset is not part of the current calibration paper scope.

## Deliverables

- This report only. No compute artefacts.
- No prereg required (Task 4 was declared optional and had a frozen fallback report path if it couldn't run).
