
from __future__ import annotations
try:
    import torch
except Exception as e:  # pragma: no cover
    torch = None

from ..fourier.transforms import dft, idft


def spectral_precondition_marginals(mu0: 'torch.Tensor', mu1: 'torch.Tensor', axis: int = -1, cutoff: float = 0.6):
    """Low-pass filter marginals by zeroing high-frequency coefficients above `cutoff`.
    Returns (mu0_s, mu1_s).
    """
    if torch is None:
        raise ImportError("PyTorch required for spectral_precondition_marginals")
    x0h = dft(mu0, axis=axis)
    x1h = dft(mu1, axis=axis)
    K = x0h.shape[-1]
    kc = int(K * cutoff)
    def lp(xh):
        out = xh.clone()
        out[..., kc:] = 0
        return out
    return idft(lp(x0h), n=mu0.shape[-1], axis=axis), idft(lp(x1h), n=mu1.shape[-1], axis=axis)
