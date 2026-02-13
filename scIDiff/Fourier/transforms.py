
"""
Fourier transforms and utilities for scIDiff-v2.
- Operate along the genes axis using torch.fft
- Provide helpers for real<->complex packing if needed
"""
from __future__ import annotations
from typing import Tuple

try:
    import torch
except Exception as e:  # pragma: no cover
    torch = None


def _require_torch():
    if torch is None:
        raise ImportError("PyTorch not available. Install torch to use Fourier modules.")


def dft(x, axis: int = -1):
    """Compute rFFT along genes axis. x: real tensor (B, G) or (..., G). Returns complex tensor.
    """
    _require_torch()
    return torch.fft.rfft(x, dim=axis)


def idft(x_hat, n: int | None = None, axis: int = -1):
    """Inverse rFFT along genes axis. x_hat: complex tensor. n can set original size.
    """
    _require_torch()
    return torch.fft.irfft(x_hat, n=n, dim=axis)


def mag_phase(x_hat):
    """Return (magnitude, phase) of complex spectrum."""
    _require_torch()
    return torch.abs(x_hat), torch.angle(x_hat)


def pack_ri(x_hat):
    """Pack complex tensor into real-valued last-dim=2 representation."""
    _require_torch()
    return torch.stack([x_hat.real, x_hat.imag], dim=-1)


def unpack_ri(x_hat_ri):
    """Unpack real-imag representation back to complex tensor."""
    _require_torch()
    return torch.view_as_complex(x_hat_ri)
