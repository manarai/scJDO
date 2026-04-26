"""
scqdiff/grn/__init__.py
========================
HybridGRN extension — opt-in experimental module.

This package provides gene-space GRN extraction on top of the existing
scQDiff latent dynamics workflow.  It is NOT the default workflow.

See docs/hybrid_grn_extension.md for full documentation.

Quick imports
-------------
    from scqdiff.grn.pullback import pullback_gene_operator, binned_pullback
    from scqdiff.grn.refine import GRNRefinerConfig, SparseGRNRefiner
    from scqdiff.grn.archetypes import grn_modes, GRNArchetypeResult
    from scqdiff.grn.priors import identify_tfs, build_tf_mask
    from scqdiff.grn.perturb import knockout_score, batch_knockout_scores
"""
from scqdiff.grn.pullback import pullback_gene_operator, binned_pullback
from scqdiff.grn.refine import GRNRefinerConfig, SparseGRNRefiner
from scqdiff.grn.archetypes import grn_modes, GRNArchetypeResult
from scqdiff.grn.priors import identify_tfs, build_tf_mask
from scqdiff.grn.perturb import knockout_score, batch_knockout_scores

__all__ = [
    "pullback_gene_operator",
    "binned_pullback",
    "GRNRefinerConfig",
    "SparseGRNRefiner",
    "grn_modes",
    "GRNArchetypeResult",
    "identify_tfs",
    "build_tf_mask",
    "knockout_score",
    "batch_knockout_scores",
]
