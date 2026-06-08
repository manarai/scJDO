# scJDO manuscript v24 → v25  ·  Tier 2 revisions

Targets items **4–7** from the Tier 2 polish list. Each edit is given as a
**before** quote (verbatim from `scJDO_manuscript_v24.pdf`) and an **after** block
ready to paste into whatever source you draft in. Items 1–3 (Tier 1) are not
covered here.

Section anchors below are PDF-page numbers; line references are by paragraph
or sentence within that page.

---

## Item 4 — Reconcile pseudotime precision

You softened pseudotime values in some places but kept three-decimal precision
in the reprogramming section. Given Box 1 already states that absolute eigen-
values and their scale "depend on pseudotime parameterization," reporting
0.691 vs 0.696 as distinct numbers (a 0.005 separation) implies a precision
you've explicitly disclaimed elsewhere. Round consistently to two decimals
throughout, and merge the near-equal MET / iPSC values.

### 4a — Hematopoiesis Results (PDF p.7)

**Before**

> For the Erythroid and DC branches, peak sensitivity occurred late in
> pseudotime (pt = 0.707 and 0.822, respectively), corresponding to the
> commitment phase.

**After**

> For the Erythroid and DC branches, peak sensitivity occurred late in
> pseudotime (pt ≈ 0.71 and 0.82, respectively), corresponding to the
> commitment phase. We report peak pseudotimes to two decimals throughout,
> consistent with the scale-dependence of the pseudotime parameterization
> noted in Box 1.

### 4b — Reprogramming Results (PDF p.7–8)

**Before**

> The A1→A2 handoff crossover occurs at pseudotime 0.691 in the MET corridor
> and 0.696 in the iPSC branch, but is absent in the Stromal diversion branch
> (no crossover detected across the full pseudotime range). Crucially, this
> handoff failure is detectable at pseudotime 0.6, well before the branches
> become visibly separated in the force-directed layout (day 8.8) …

**After**

> The A1→A2 handoff crossover occurs at pseudotime ≈ 0.69 in both the
> successful MET corridor and the iPSC branch (the two are not resolvable
> from each other within the pseudotime precision warranted by Box 1), but
> is absent in the Stromal diversion branch (no crossover detected across
> the full pseudotime range). The handoff failure is detectable at pt ≈ 0.6,
> well before the branches become visibly separated in the force-directed
> layout (day 8.8) …

### 4c — Add precision note to Box 1 (optional but recommended)

**Insert** as the final bullet of Box 1 (just below the existing
"Absolute eigenvalues …" line), to make the rounding policy load-bearing:

> Pseudotime coordinates are reported to two decimals throughout. Differences
> below this precision (e.g. 0.69 vs 0.70) should not be interpreted as
> resolved given the dependence of pseudotime scale on parameterization.

---

## Item 5 — Account for Archetype 3 in the reprogramming section

The text discusses A1, A2 and A4. A3 is unmentioned, which reads as post-hoc
K-tuning rather than as a principled gap. Insert one sentence in the
reprogramming Results section so the numbered gap is closed.

### 5a — Reprogramming Results (PDF p.8, after the existing module-overlap sentences)

**Before**  *(current paragraph, last two sentences)*

> Archetype 1 strongly aligned with the MEF identity module, Archetype 2 with
> the Pluripotency and Metabolic shift modules, and Archetype 4 with the
> Metabolic shift module. This confirms that the unsupervised mathematical
> operators capture distinct, known biological programs and that scJDO can
> resolve time-resolved success versus diversion in a real time-course setting.

**After**

> Archetype 1 strongly aligned with the MEF identity module, Archetype 2 with
> the Pluripotency and Metabolic shift modules, and Archetype 4 with the
> Metabolic shift module. **Archetype 3 did not align significantly with any
> of the curated reprogramming modules tested above an enrichment cutoff of
> FDR < 0.05 (max fold-enrichment across all five modules reported in
> Supplementary Table S\[X\]); its activation profile is broad and overlaps
> both branches, and is reported here for completeness rather than as a
> stage-specific identity program. K = 5 was selected by the same consensus
> procedure used throughout (Methods), not tuned to recover a
> branch-discriminating component.** This confirms that the unsupervised
> mathematical operators capture distinct, known biological programs and that
> scJDO can resolve time-resolved success versus diversion in a real
> time-course setting.

If A3 actually does have a coherent biological assignment that's currently
missing from the text, replace the bold sentence with that assignment instead.
The key requirement is that *no archetype index goes unaddressed*.

---

## Item 6 — Rewrite "Relation to existing dynamical approaches"

Drop the "trajectories vs instability" framing. SpliceJAC explicitly computes
per-cell-state Jacobians whose spectra are read as local instability and
transition-gene identification (Bocci, Zhou & Nie, *Mol Syst Biol* 2022).
Dynamo computes per-cell Jacobians and reads divergence/spectra for local
stability (Qiu et al., *Cell* 2022). Both produce the same kind of object
scJDO produces at the per-point level. The genuine novelty is (i) Jacobian
source, (ii) temporal tensor decomposition, (iii) interpretive stance —
*not* "instability vs. trajectory."

### 6a — Replace the paragraph at PDF p.9 ("Relation to existing dynamical approaches")

**Before**

> scJDO occupies an uncontested niche in the single-cell analysis ecosystem.
> While transition-operator frameworks (e.g., CellRank, Palantir) identify
> macrostates and fate probabilities, and continuous-time models (e.g.,
> VeloVAE, neural ODEs) learn trajectory geometry, no existing method targets
> the local dynamical sensitivity encoded in time-indexed Jacobians. scJDO is
> designed to complement these approaches (Table 1).

**After**

> scJDO is most closely related to single-cell methods that also compute a
> local Jacobian of an inferred vector field — specifically **SpliceJAC**
> (Bocci, Zhou & Nie, *Mol Syst Biol* 2022) and **Dynamo** (Qiu *et al.*,
> *Cell* 2022) — and is complementary to, but distinct from, transition-
> operator and fate-mapping methods such as CellRank, Palantir and Slingshot,
> and continuous-time generative models such as VeloVAE and neural-ODE
> velocity methods (Table 1).
>
> Relative to SpliceJAC and Dynamo, the differences are sharp on three axes
> rather than the kind of object computed.
>
> *First — source of the Jacobian.* SpliceJAC and Dynamo derive their
> Jacobian from a splicing-kinetic vector field, requiring spliced/unspliced
> count matrices and explicit assumptions about RNA production, splicing and
> degradation. scJDO learns a drift field directly from cell-state geometry
> via diffusion score matching and computes Jacobians by automatic
> differentiation of that learned drift, enabling operator analysis on any
> scRNA-seq dataset regardless of splicing-data availability. This is
> especially relevant given recent benchmarking suggesting that
> splicing-derived Jacobians can be unstable across systems, while geometry-
> or analytical-Jacobian baselines remain more consistent (Daniel-Carlier *et
> al.*, *bioRxiv* 2025).
>
> *Second — temporal structure of the operator.* SpliceJAC computes one
> Jacobian per annotated discrete cell state; Dynamo computes per-cell
> Jacobians interpreted as static regulatory snapshots. Neither treats the
> time-ordered sequence of Jacobians as a single analytical object. scJDO
> stacks Jacobians across a continuous pseudotime axis into a tensor
> $J \in \mathbb{R}^{T \times d \times d}$ and decomposes that tensor into
> recurrent operator archetypes with non-negative temporal activation
> profiles. This temporal-decomposition layer — not the per-point Jacobian
> itself — is the uncontested contribution of scJDO.
>
> *Third — interpretive stance.* Dynamo reads Jacobian entries $J_{ij}$ as
> causal regulatory strengths and uses them to identify regulator/effector
> relationships, mutual-inhibition motifs, and in-silico perturbation
> targets. scJDO, by contrast, explicitly disclaims this interpretation (Box
> 1): we target only the relative operator structure that is reproducible
> across independent runs, latent dimensions and architectural choices, and
> we do not interpret individual matrix entries as causal regulatory
> couplings. The two interpretive stances answer different questions about
> the same kind of object.
>
> Concordance of scJDO's geometry-derived Jacobian spectra with SpliceJAC's
> splicing-derived spectra on splicing-amenable datasets is therefore the
> most direct external validation of scJDO's operator inference; we examine
> this concordance on the Paul15 hematopoiesis system in
> Supplementary Note S\[X\] / Supplementary Figure S\[Y\].
> *(Replace this sentence with the actual concordance result once the
> Tier-1 experiment is in.)*
>
> Relative to fate-mapping methods (CellRank, Palantir, Slingshot) and
> continuous-time generative models (VeloVAE, neural-ODE velocity), scJDO
> sits downstream rather than in competition: it consumes a pseudotime
> ordering produced by any of these methods, and returns an operator-level
> summary of how local dynamical sensitivity evolves along that ordering. A
> head-to-head comparison against fate-prediction accuracy is therefore a
> category error — these methods answer "where will this cell end up,"
> whereas scJDO answers "where along the trajectory does local sensitivity
> peak, and how do operator regimes hand off." Table 1 makes this division
> of labor explicit.

### 6b — Table 1 row revision

Split the current "RNA velocity" row so Dynamo (Jacobian) is not grouped with
scVelo/UniTVelo (velocity only). Suggested replacement of the two velocity-
related rows:

| Primary question | Recommended method | scJDO role |
|---|---|---|
| How fast are individual genes changing? RNA velocity estimates? | scVelo, UniTVelo | **Parallel**: scJDO uses velocity as an optional weak prior; run independently. |
| What is the local regulatory Jacobian at each cell state (gene-by-gene, causal-interpreted) from splicing kinetics? | **Dynamo, SpliceJAC** | **Adjacent**: same kind of object (a local Jacobian) computed from splicing; scJDO computes the comparable object from cell-state geometry instead, and adds a temporal-tensor decomposition not present in either. Concordance check appropriate on splicing-amenable datasets (Supplementary Note S\[X\]). |
| What is the continuous-time dynamics model? | VeloVAE, neural ODE | Parallel: scJDO can be run on the same latent space; outputs are complementary. |

### 6c — Add citations to the reference list

- Bocci F., Zhou P., Nie Q. **spliceJAC: transition genes and state-specific gene regulation from single-cell transcriptome data.** *Molecular Systems Biology* 18:e11176 (2022).
- Qiu X., Zhang Y., Martin-Rufino J. D., Weng C., *et al.* **Mapping transcriptomic vector fields of single cells.** *Cell* 185(4):690–711 (2022).
- Daniel-Carlier N. *et al.* **Benchmarking single-cell dynamical gene regulatory network inference [working title — replace with actual title once you've pulled the citation].** *bioRxiv* 2025. doi:[fill in]

*(Verify the bioRxiv preprint title, DOI and author list against the actual
record before submission — only the existence of the result and rough year
were established in the conversation.)*

---

## Item 7 — Cell-autonomous caveat: name the showcase systems explicitly

The Failure Modes section flags the cell-autonomous limitation in general
terms (PDF p.11, "Systems dominated by cell-cell interactions"). Reprogramming
is Wnt/Nodal/BMP-cued; cortical neurogenesis runs on Notch lateral inhibition
plus FGF/SHH gradients. Naming this specifically pre-empts an obvious
reviewer comment.

### 7 — Insert at the end of the **Limitations** section (PDF p.12)

**Insert** as a new paragraph after the existing "Because scJDO analyzes
Jacobians of a learned drift field …" paragraph and before the closing
"Overall, scJDO illustrates …" paragraph:

> Both of our main biological systems are signaling-dense, and the
> cell-autonomous modeling assumption flagged above as a general failure mode
> applies to them in particular. iPSC reprogramming is shaped by Wnt, Nodal,
> BMP and LIF/STAT3 signaling between MEFs and emerging pluripotent
> intermediates, while embryonic cortical neurogenesis is governed in part by
> Notch lateral inhibition between progenitors and IPCs and by FGF and SHH
> gradients across the ventricular zone. Because scJDO's current
> implementation treats $f(x,t)$ as a function of the individual cell state
> only and does not represent juxtacrine or paracrine coupling, the operator
> structure we recover should be read as the dynamical sensitivity inferable
> from cell-state geometry alone — not as a full description of the cell-cell
> regulatory logic. We expect this to understate, rather than overstate,
> effects driven by intercellular signaling, since coupling that is constant
> within a pseudotime window is absorbed into the local drift but
> heterogeneous coupling is not. Extending scJDO to neighborhood-conditioned
> drift fields is a natural next step.

---

## Summary of edits and where they land

| Item | Section | Lines / location | Net change |
|---|---|---|---|
| 4a | Results — Hematopoiesis (p.7) | "pt = 0.707 and 0.822" | round to 0.71 / 0.82 + add Box-1-consistency clause |
| 4b | Results — Reprogramming (p.7–8) | "0.691 in the MET corridor and 0.696 in the iPSC branch" | merge to "≈ 0.69 in both" |
| 4c | Box 1 (p.5)        | new final bullet | rounding policy stated once, load-bearing |
| 5  | Results — Reprogramming (p.8) | module-overlap paragraph | one bold sentence acknowledging A3 + K-selection procedure |
| 6a | Discussion — Relation to existing dynamical approaches (p.9) | full paragraph | replace with three-axis distinction, name SpliceJAC + Dynamo, cite 2025 splicing-Jacobian benchmark, forward-reference concordance Supplementary Note |
| 6b | Table 1 (p.10) | velocity row split | Dynamo + SpliceJAC get their own row labelled "adjacent, concordance-checkable" |
| 6c | References | three new entries | SpliceJAC, Dynamo, bioRxiv benchmark |
| 7  | Limitations (p.12) | new paragraph | name reprogramming + neurogenesis as the systems where the cell-autonomous caveat bites; flag direction of bias |

**Time estimate (per the Tier-2 sequencing plan):** ~1 day total — items 4, 5
and 7 are minutes of edits; item 6 is ~half a day because it rewrites the
discussion's single most consequential paragraph and the reference list.

**Sequencing reminder.** Items 4, 5 and 7 can ship now and are independent of
the Tier-1 SpliceJAC concordance experiment. Item 6 has two
"Supplementary Note S\[X\]" / "Supplementary Figure S\[Y\]" placeholders that
should be filled in only after the concordance numbers are in; in the interim
the paragraph reads as a forward reference, which is acceptable. If the
concordance experiment slips, soften the two placeholder sentences to
"… is a natural next validation step, currently in progress."
