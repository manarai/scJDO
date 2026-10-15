"""scJDO v2 — field-based local operators.

v2 estimates a Jacobian per cell from latent-space neighbourhoods instead of
pseudotime slices. v1 remains at :mod:`scjdo` and is unaffected.

Phase B (this module) builds the field machinery on top of v1's log-rate
derivatives of the drift. Phase C swaps in a scvi-tools count-likelihood
rate model and the Fisher ledger.

Public API (Phase B):

- :func:`scjdo.v2.embed.fit_embedding`         — Step 1
- :func:`scjdo.v2.drift.fit_drift_v2`          — Step 2
- :func:`scjdo.v2.field.build_operator_field`  — Steps 5
- :func:`scjdo.v2.opspace.operator_descriptors`— Step 7
- :func:`scjdo.v2.archetypes.fit_archetypes`   — Step 8
"""
from __future__ import annotations

__version__ = "0.4.0.dev0"

from scjdo.v2.embed import fit_embedding, EmbeddingResult
from scjdo.v2.drift import fit_drift_v2, DriftV2Result
from scjdo.v2.field import build_operator_field, OperatorFieldResult
from scjdo.v2.opspace import operator_descriptors, DescriptorResult
from scjdo.v2.archetypes import fit_archetypes, ArchetypeResult

__all__ = [
    "fit_embedding", "EmbeddingResult",
    "fit_drift_v2", "DriftV2Result",
    "build_operator_field", "OperatorFieldResult",
    "operator_descriptors", "DescriptorResult",
    "fit_archetypes", "ArchetypeResult",
]
