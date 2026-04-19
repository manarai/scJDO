
from __future__ import annotations
from typing import Tuple, Dict

try:
    import torch
except Exception as e:  # pragma: no cover
    torch = None


def _require_torch():
    if torch is None:
        raise ImportError("PyTorch not available. Install torch to use Fourier modules.")


def split_bands(x_hat, splits=(0.2, 0.6)) -> Dict[str, torch.Tensor]:
    """Split complex spectrum x_hat (B, K) into low/mid/high by frequency index fractions.
    splits: two fractions in (0,1) for boundaries.
    Returns dict with masked tensors (same shape; zeros elsewhere).
    """
    _require_torch()
    B = x_hat.shape[0]
    K = x_hat.shape[-1]
    k1 = int(K * splits[0])
    k2 = int(K * splits[1])
    low_mask = torch.zeros(K, dtype=torch.bool, device=x_hat.device)
    mid_mask = torch.zeros(K, dtype=torch.bool, device=x_hat.device)
    high_mask = torch.zeros(K, dtype=torch.bool, device=x_hat.device)
    low_mask[:k1] = True
    mid_mask[k1:k2] = True
    high_mask[k2:] = True
    zeros = torch.zeros_like(x_hat)
    return {
        'low': torch.where(low_mask, x_hat, zeros),
        'mid': torch.where(mid_mask, x_hat, zeros),
        'high': torch.where(high_mask, x_hat, zeros),
    }


def merge_bands(d: Dict[str, 'torch.Tensor']):
    """Merge band-split tensors by summation (assumes disjoint masks)."""
    _require_torch()
    return d['low'] + d['mid'] + d['high']
