"""scQDiff: Learning single-cell regulatory dynamics with hybrid drift fields."""

# Core models
from .models.drift import DriftField, DriftConfig
from .models.schrodinger_bridge import SchrodingerBridge, SchrodingerBridgeConfig

# IO
from .io.anndata import tensors_from_anndata

# Fourier-domain utilities
from .fourier.transforms import dft, idft, pack_ri, unpack_ri
from .fourier.bands import split_bands, merge_bands
from .fourier.features import power_spectrum_features
from .fourier.losses_spectral import band_weighted_score_loss, spectral_smoothness_loss
from .fourier.kspace_samplers import KSpaceEulerMaruyama, KSpaceHeun

__version__ = "0.2.0"

__all__ = [
    # Models
    "DriftField",
    "DriftConfig",
    "SchrodingerBridge",
    "SchrodingerBridgeConfig",
    # IO
    "tensors_from_anndata",
    # Fourier
    "dft",
    "idft",
    "pack_ri",
    "unpack_ri",
    "split_bands",
    "merge_bands",
    "power_spectrum_features",
    "band_weighted_score_loss",
    "spectral_smoothness_loss",
    "KSpaceEulerMaruyama",
    "KSpaceHeun",
]
