"""scQDiff: Operator-level analysis of single-cell dynamics."""

# Scanpy-style namespaces
from . import pp
from . import tl
from . import pl

# Core models (direct access)
from .models.drift import DriftField, DriftConfig
from .models.schrodinger_bridge import SchrodingerBridge, SchrodingerBridgeConfig

# IO
from .io.anndata import tensors_from_anndata

__version__ = "0.3.0"

__all__ = [
    "pp", "tl", "pl",
    "DriftField", "DriftConfig",
    "SchrodingerBridge", "SchrodingerBridgeConfig",
    "tensors_from_anndata",
]
