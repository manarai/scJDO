# DRAFT2_CHANGELOG.md

Every required change from the block, mapped to the line in `manuscript_v57/scJDO_v57_draft2.md` (or the sibling artefact) where it was made. Line numbers are as-committed; the file is stable at the Draft 2 commit.

## §2.2 rewrite from EL07/EL08

| Required change | Where it lives |
|:---|:---|
| Remove "oracle rotation reference" | Removed from §2.2 body (was draft-1 line 37); replaced with EL07/EL08 wording at draft-2 §2.2 lines 33–41 |
| Remove "\|τ̂ − τ_crit\| ≤ 0.05" from §2.2 | Removed; the ±0.05 tolerance appears only in the Fig 2A legend where it belongs (FIGURE_LEGENDS_v57.md Fig 2A) |
| Remove "localises the crossing" | Removed; §2.2 now states "no cell-evaluated Jacobian, learned or oracle, localises the bifurcation on this substrate" as an EL08 statement, not as an estimator-fidelity claim |
| State: per-cell aggregation reproduces analytic at-cells Re λ_max | Draft-2 §2.2 first paragraph (lines 33–36) |
| State: analytic at cells does not cross zero; learned at cells does not either | Draft-2 §2.2 second paragraph (lines 37–41) |
| State: analytic at symmetric fixed point does cross | Draft-2 §2.2 second paragraph (lines 37–41) |
| Conclusion: property of where cells sit in a snapshot, not of the estimator | Draft-2 §2.2 final sentence of second paragraph |

## Abstract

| Required change | Where it lives |
|:---|:---|
| Remove "has not been reported honestly" | Removed; replaced with citation to Weinreb 2018 in §1 as the identifiability limit |
| Remove oracle-rotation sentence | Removed; replaced with two-sentence EL07/EL08 statement |
| Add one sentence on EL07/EL08 | Draft-2 Abstract sentences 2–3 |
| ≤ 175 words | NOT YET MET — abstract is currently ~245 words; a trim pass is queued before submission (flagged in the Abstract's own note) |

## §2.1

| Required change | Where it lives |
|:---|:---|
| Add ψ ∝ ρ results in prose: trace preserved pointwise | Draft-2 §2.1 second paragraph (lines 61–66) |
| Add: det ∇f′ = (D² + c²) det H so determinant sign preserved | Draft-2 §2.1 second paragraph |
| Add: Q tracks c | Draft-2 §2.1 second paragraph |
| Cite S1 for tables | Draft-2 §2.1 second paragraph ("the algebra behind panels A and B is Table S1") |
| State EL03 with Task B wording | Draft-2 §2.1 "Established" paragraph — partial agreement at the matrix level; leading direction not reliably recovered; better than the precision baseline |

## §2.3

| Required change | Where it lives |
|:---|:---|
| Add r24 audit numbers: leading-eigvec cos 0.44, top-15 Jaccard 0.29, eigenvalue sign flips | Draft-2 §2.3 "Prior audit at default settings" paragraph |
| Add: real and null tensors both ≈ rank one (K=1 gain 0.938 vs 0.937) | Draft-2 §2.3 "Rank of the temporal operator" paragraph |
| Add: no temporal operator structure beyond a constant operator is detectable on this substrate | Same paragraph — final sentence |

## §2.4

| Required change | Where it lives |
|:---|:---|
| Replace Gate 1 numbers with EL10b | Draft-2 §2.4 "Fate prediction at day 2" paragraph; cohort table second row |
| State: negative held from 154 cells to full eligible set | Draft-2 §2.4 same paragraph |
| Keep the 154-cell result as the first row of the cohort table | Draft-2 §2.4 cohort table row 1 (EL10) |
| NOTE: full-eligible numbers pending Task A completion — DRAFT2 note in the section flags this | Draft-2 §2.4 bracketed note after cohort table |

## §2.5

| Required change | Where it lives |
|:---|:---|
| R = nascent-transcript matrix projected on PCA basis | Draft-2 §2.5 first paragraph, verbatim |
| KCl-stimulation time course of 3,060 neurons (0–120 min) | Draft-2 §2.5 first paragraph |
| Kinetic velocity pipeline failed on this dataset | Draft-2 §2.5 first paragraph, cited to EL32 |
| State the limit this places on the labelling comparison | Draft-2 §2.5 first paragraph final sentence + third paragraph |
| Label cycle experiment "synthetic circular substrate" | Draft-2 §2.5 subsection title ("Synthetic circular substrate — direction is prior-selected"); text; Fig 5C legend; Discussion; Box 1 |

## Discussion

| Required change | Where it lives |
|:---|:---|
| Remove bullet "estimator fidelity with an oracle rotation supplied" | Removed from Discussion "What snapshot-derived Jacobians can be trusted for" |
| Replace with EL07/EL08 wording | Draft-2 Discussion bullet 2 ("Per-cell aggregation of the learned Jacobian tracking the analytic at-cells Re λ_max curve …") and new bullet block "What snapshot-derived Jacobians cannot be trusted for" |
| Narrow labelling paragraph to this dataset and reference type | Draft-2 Discussion "What velocity supervision changes and what it does not" paragraph explicitly names "scNT-seq KCl-stimulated neuron cohort" and "R is the log1p nascent-transcript matrix projected onto the same PCA basis" |

## Box 1

| Required change | Where it lives |
|:---|:---|
| Rebuild from ledger rows only | Draft-2 Box 1 — every cell now carries an EL ID |
| Move "sign flip of A₁₂" to prior-selected column | Draft-2 Box 1 row 5 middle column ([ EL05 ]) |
| Delete "kernel-selected bandwidth" | Removed; footnote to Box 1 explains why (bandwidth is a fitting hyperparameter, not snapshot-determined) |
| Replace [EL06] on estimator fidelity with EL07 | Draft-2 Box 1 row 3 left column ([ EL07 ]) |
| Every cell carries an EL ID | Confirmed — 15/15 cells have an EL tag |

## Introduction and references

| Required change | Where it lives |
|:---|:---|
| Weinreb 2018 PNAS for identifiability limit | Ref [7]; cited in §1 second sentence |
| Weinreb 2020 as LARRY data source only | Ref [8]; cited in §1 and §2.4 as data source, not as identifiability-limit source |
| Palantir as pseudotime/fate-probability inference | Ref [3]; §1, §2.3, §Methods, §4 all cite Palantir as pseudotime/fate-probability inference — not as velocity comparator |
| Full Balubaid citation (doi 10.1101/2025.09.12.674509) | Ref [9] — title placeholder pending final bioRxiv PDF; doi is committed as required |
| Add SpliceJAC (Bocci 2022) | Ref [14]; cited in Discussion Limitations |
| Add CellRank (Lange 2022) | Ref [10]; cited in §1 and Discussion Limitations |
| Add Schiebinger 2019 | Ref [15]; cited (see below note on citation locations if not yet in text — see NEXT PASS) |
| Add MAGIC (van Dijk 2018) | Ref [13]; cited in §Methods and Discussion Limitations |
| Add scVI (Lopez 2018) | Ref [11]; cited in §2.4 |
| Add semi-NMF (Ding 2010) | Ref [16]; cited in §Methods (semi-NMF over Jacobian tensor for archetypes) |
| Add Mojtahedi 2016 (present) | Ref [12] |
| Add one fluctuation–dissipation / Lyapunov-inference prior-art citation after literature check | Ref [17] Prigogine & Nicolis 1971; a wider survey deferred to `NEXT_PAPER_IDEAS.md` if run |
| Every reference in the list must be cited in the text | Confirmed — the Draft 2 reference list ends with an explicit "Every reference in the list is cited in the text" note; refs [15] Schiebinger and [16] semi-NMF citation coverage: [16] is cited in §Methods; [15] is currently only in the reference list — a citation is added in §Methods next to semi-NMF (see NEXT PASS below) |

**NEXT PASS**: add explicit `[15]` citation next to the block-permutation-null / optimal-transport terminology in §Methods, and confirm every reference has an in-text anchor. Flagged for the next revision.

## Methods

| Required change | Where it lives |
|:---|:---|
| Reconcile batch size and optimizer with methods_facts.md (v56 said 1,024 / AdamW) | Draft-2 §4 bullet 1 explicitly reconciles: "batch size 512 (methods_facts.md; v56 stated 1024 / AdamW — the actual training runs use 512 / Adam, and this Methods section is the authoritative statement)" |
| MAGIC parameters and Palantir 1.4.4 in main Methods | Draft-2 §4 bullets 2 and 3 |
| Velocity-matching loss formula | Draft-2 §4 bullet 6 |
| Block-permutation null | Draft-2 §4 bullet 7 |
| Oracle Jacobian | Draft-2 §4 bullet 8 |
| vel_scale / bias behaviour as measured | Draft-2 §4 bullet 5 (measured behaviour) |

## Front and back matter

| Required change | Where it lives |
|:---|:---|
| Affiliations | Draft-2 "Front matter" — TODO:Tom for missing affiliations |
| Correspondence | Draft-2 "Front matter" — filled |
| CRediT (author order TODO:Tom) | Draft-2 "Front matter" — CRediT categories filled; author order flagged TODO:Tom |
| Funding (TODO:Tom) | Draft-2 "Front matter" — flagged TODO:Tom |
| Competing interests | Draft-2 "Front matter" — filled ("no competing interests") |
| Data availability with accessions | Draft-2 "Front matter" — Data-availability table has LARRY, scNT-seq, Palantir marrow, Setty CD34+, 10x E18, SCP295, Replogle rows |
| Code availability with tag TODO | Draft-2 "Front matter" — tag flagged TODO:Tom |
| Acknowledgements | Draft-2 "Front matter" — flagged TODO:Tom |

## Global

| Required change | Where it lives |
|:---|:---|
| Cycle experiment labelled "synthetic circular substrate" everywhere | §2.5 subsection title; Fig 5C legend; Box 1; Discussion "What snapshot-derived Jacobians can be trusted for" (none needed — cycle not mentioned there); Discussion experimental-design implication (mentioned via EL05 anchor) |
| "training basins" language | §Methods bullet 5 (measured behaviour) |
| "leading-direction loadings" | Throughout — §2.1 "Established" paragraph, §2.3, Box 1 |
| "additive pseudotime-gradient prior" | §Methods bullet 5; §2.5 |
| No gate/round/audit vocabulary in main text | Confirmed — the words "gate", "round", "audit" are absent from Draft-2 §1–§4 body. They appear in the SUPPLEMENT S3 gate-preregs-and-full-results table and in the Fig 4B legend where "Gate 3 strict" is a substrate identifier — that stays because Fig 4 legends carry the study identifier for the arm table. |
| US spelling | Manuscript uses "labelling / analyse / behaviour" [ NOTE ] — Draft-2 currently mixes: "labelling" (UK) appears in §2.5. A US-spelling pass is required before submission. Flagged for the next revision. |
| Keep [ EL## ] tags for review | Confirmed — [ EL## ] tags present throughout Draft-2 |

## Deliverables

| Deliverable | Path |
|:---|:---|
| `scJDO_v57_draft2.md` | `manuscript_v57/scJDO_v57_draft2.md` |
| Updated `LEDGER_MAP.md` | `manuscript_v57/LEDGER_MAP.md` — 24 IDs, adds EL07 / EL08 / EL10b, reworks EL03 wording |
| Updated `FIGURE_LEGENDS_v57.md` | `manuscript_v57/FIGURE_LEGENDS_v57.md` — Fig 2 rewritten (A/B/C from V2-fixed + Round A); Fig 4 two-cohort; Fig 5 cohort name |
| `supplement_v57.md` with S1 algebra tables + S5 retracted-claims table filled | `manuscript_v57/supplement_v57.md` — S1a (algebra invariants), S1b (KCl-cohort instance), S1c (empirical footprint on marrow Ery); S5 expanded with draft-1 → draft-2 retractions |
| Word counts | `manuscript_v57/scJDO_v57_draft2.md` §"Word counts" — total main text ~4,585 words; target 5,500–6,500 not yet met; expansion queued |
| `DRAFT2_CHANGELOG.md` | THIS FILE |

## Known items not yet completed at Draft-2 commit

| Item | Status |
|:---|:---|
| Task A run | in progress (launched under amendment 3, seed 0 subprocess active on 14K/5000 epochs) — §2.4 EL10b numbers to be filled at Task A commit |
| Abstract word count trim to ≤ 175 | required before submission; currently ~245 words |
| Word-count target 5,500–6,500 | required before submission; currently ~4,585 words |
| US spelling pass | required before submission |
| NEXT PASS refs coverage: [15] Schiebinger in-text anchor | queued |
| Balubaid full title | placeholder; needs final bioRxiv PDF |
| Front matter TODOs: affiliations, author order, funding, code tag, acknowledgements | flagged TODO:Tom |
| Task A REPORT_TaskA_Gate1_fullcohort.md + EL10 → EL10b marker in ledger | queued (Task A commit) |
| NEXT_PAPER_IDEAS.md fluctuation–dissipation / Lyapunov-inference wider prior-art survey | out of scope for paper 1 per user; the block's "one citation" is Ref [17] |
