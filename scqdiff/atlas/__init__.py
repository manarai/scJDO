"""
scOpAtlas: Stable Operator Atlas for scidiff

This module provides operator-based state definition for single-cell data.
It defines cellular states by their local stability structure rather than
expression patterns alone.

Main components:
- OperatorMetrics: Compute eigenvalue-derived metrics
- OperatorRegimeClassifier: Classify cells into operator regimes
- StableOperatorAtlas: Main atlas object
"""

from .operator_metrics import OperatorMetrics
from .regime_classifier import OperatorRegimeClassifier
from .atlas_builder import StableOperatorAtlas

__all__ = [
    "OperatorMetrics",
    "OperatorRegimeClassifier",
    "StableOperatorAtlas",
]
