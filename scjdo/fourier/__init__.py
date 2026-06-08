"""
scJDO: Learning Single-Cell Regulatory Dynamics with Hybrid Drift Fields.

Subpackages
-----------
scjdo.models      — DriftField, SchrodingerBridge, MultiBandScoreNet
scjdo.fourier     — Fourier-domain diffusion utilities
scjdo.transport   — Optimal transport and spectral bridge
scjdo.io          — AnnData I/O utilities
scjdo.pipeline    — Training pipelines (CLI entry points)
scjdo.atlas       — scOpAtlas operator regime analysis
"""

# Core models — keep existing import that was already there
from scjdo.models.drift import DriftField, DriftConfig

# Add the rest
from scjdo.models.schrodinger_bridge import SchrodingerBridge, SchrodingerBridgeConfig
from scjdo.models.fourier_score_network import MultiBandScoreNet

__all__ = [
    "DriftField",
    "DriftConfig",
    "SchrodingerBridge",
    "SchrodingerBridgeConfig",
    "MultiBandScoreNet",
]
