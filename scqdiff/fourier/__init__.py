"""
scQDiff: Learning Single-Cell Regulatory Dynamics with Hybrid Drift Fields.

Subpackages
-----------
scqdiff.models      — DriftField, SchrodingerBridge, MultiBandScoreNet
scqdiff.fourier     — Fourier-domain diffusion utilities
scqdiff.transport   — Optimal transport and spectral bridge
scqdiff.io          — AnnData I/O utilities
scqdiff.pipeline    — Training pipelines (CLI entry points)
scqdiff.atlas       — scOpAtlas operator regime analysis
"""

# Core models — keep existing import that was already there
from scqdiff.models.drift import DriftField, DriftConfig

# Add the rest
from scqdiff.models.schrodinger_bridge import SchrodingerBridge, SchrodingerBridgeConfig
from scqdiff.models.fourier_score_network import MultiBandScoreNet

__all__ = [
    "DriftField",
    "DriftConfig",
    "SchrodingerBridge",
    "SchrodingerBridgeConfig",
    "MultiBandScoreNet",
]
