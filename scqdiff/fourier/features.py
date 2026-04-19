
from __future__ import annotations
from typing import Dict

try:
    import torch
except Exception as e:  # pragma: no cover
    torch = None


def _require_torch():
    if torch is None:
        raise ImportError("PyTorch not available. Install torch to use Fourier modules.")


def power_spectrum(x_hat) -> 'torch.Tensor':
    """Return power spectrum |X(k)|^2 along last dim."""
    _require_torch()
    return (x_hat.real ** 2 + x_hat.imag ** 2)


def power_spectrum_features(x_hat, splits=(0.2, 0.6)) -> Dict[str, 'torch.Tensor']:
    """Band energy features for conditioning."""
    _require_torch()
    ps = power_spectrum(x_hat)
    B, K = ps.shape[0], ps.shape[-1]
    k1 = int(K * splits[0])
    k2 = int(K * splits[1])
    feats = {
        'ps_low': ps[..., :k1].mean(dim=-1, keepdim=True),
        'ps_mid': ps[..., k1:k2].mean(dim=-1, keepdim=True),
        'ps_high': ps[..., k2:].mean(dim=-1, keepdim=True),
    }
    # Concatenate into a single feature vector
    feats['ps_all'] = torch.cat([feats['ps_low'], feats['ps_mid'], feats['ps_high']], dim=-1)
    return feats
