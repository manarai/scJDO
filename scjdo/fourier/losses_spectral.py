
from __future__ import annotations
try:
    import torch
    import torch.nn.functional as F
except Exception as e:  # pragma: no cover
    torch = None
    F = None


def _require_torch():
    if torch is None:
        raise ImportError("PyTorch not available. Install torch to use spectral losses.")


def spectral_smoothness_loss(x_hat, weight: float = 1.0):
    """Finite-difference smoothness along frequency axis on magnitude."""
    _require_torch()
    mag = torch.sqrt(x_hat.real**2 + x_hat.imag**2 + 1e-8)
    diff = mag[..., 1:] - mag[..., :-1]
    return weight * (diff.pow(2).mean())


def band_weighted_score_loss(pred_hat, target_hat, band_weights=(1.0, 1.0, 0.5)):
    """L2 score loss with lighter penalty on high-frequency band."""
    _require_torch()
    K = pred_hat.shape[-1]
    k1 = int(0.2 * K)
    k2 = int(0.6 * K)
    w = torch.ones(K, device=pred_hat.device)
    w[:k1] = band_weights[0]
    w[k1:k2] = band_weights[1]
    w[k2:] = band_weights[2]
    diff = (pred_hat - target_hat)
    # magnitude error
    err = (diff.real**2 + diff.imag**2)
    return (w * err).mean()
