
from __future__ import annotations
from typing import Callable, Dict

try:
    import torch
except Exception as e:  # pragma: no cover
    torch = None


def _require_torch():
    if torch is None:
        raise ImportError("PyTorch not available. Install torch to use k-space samplers.")


class KSpaceEulerMaruyama:
    """Euler–Maruyama stepper in Fourier space.
    Expects a score_fn that maps (x_hat_t, t, cond) -> score_hat.
    """
    def __init__(self, g_fn: Callable[[float], float]):
        self.g_fn = g_fn

    def step(self, x_hat_t, t, dt, score_fn: Callable, cond: Dict = None):
        _require_torch()
        cond = cond or {}
        g = self.g_fn(t)
        score_hat = score_fn(x_hat_t, t, cond)
        drift = - (g ** 2) * score_hat
        noise = (g * (dt ** 0.5)) * (torch.randn_like(x_hat_t.real) + 1j * torch.randn_like(x_hat_t.imag))
        return x_hat_t + drift * dt + noise


class KSpaceHeun:
    """Predictor–corrector (Heun) scheme in Fourier space."""
    def __init__(self, g_fn: Callable[[float], float]):
        self.g_fn = g_fn

    def step(self, x_hat_t, t, dt, score_fn: Callable, cond: Dict = None):
        _require_torch()
        cond = cond or {}
        g = self.g_fn(t)
        s1 = score_fn(x_hat_t, t, cond)
        drift1 = - (g ** 2) * s1
        x_pred = x_hat_t + drift1 * dt
        s2 = score_fn(x_pred, t + dt, cond)
        drift2 = - (g ** 2) * s2
        drift = 0.5 * (drift1 + drift2)
        noise = (g * (dt ** 0.5)) * (torch.randn_like(x_hat_t.real) + 1j * torch.randn_like(x_hat_t.imag))
        return x_hat_t + drift * dt + noise
