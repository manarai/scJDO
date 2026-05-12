
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
    """Euler–Maruyama stepper in Fourier space."""
    def __init__(self, g_fn: Callable[[float], float]):
        self.g_fn = g_fn

    def step(self, x_hat_t, t, dt, score_fn: Callable, cond: Dict = None):
        _require_torch()
        cond = cond or {}
        g = self.g_fn(t)
        score_hat = score_fn(x_hat_t, t, cond)
        drift = -(g ** 2) * score_hat
        step_size = abs(dt)
        noise = (g * (step_size ** 0.5)) * (
            torch.randn_like(x_hat_t.real) + 1j * torch.randn_like(x_hat_t.imag)
        )
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
        drift1 = -(g ** 2) * s1
        x_pred = x_hat_t + drift1 * dt
        s2 = score_fn(x_pred, t + dt, cond)
        drift2 = -(g ** 2) * s2
        drift = 0.5 * (drift1 + drift2)
        step_size = abs(dt)
        noise = (g * (step_size ** 0.5)) * (
            torch.randn_like(x_hat_t.real) + 1j * torch.randn_like(x_hat_t.imag)
        )
        return x_hat_t + drift * dt + noise


def make_fourier_score_fn(model):
    """
    Wrap a real-space score model so it works with Fourier-space samplers.

    model: callable like model(x_real, t, cond) -> score_real
    returns: score_fn(x_hat, t, cond) -> score_hat
    """
    def score_fn(x_hat, t, cond=None):
        device = next(model.parameters()).device

        # Fourier -> real, preserving the original gene dimension
        n = getattr(model, "gene_dim", None)
        x = torch.fft.irfft(x_hat, n=n, dim=-1)
        x = x.to(device)

        # real-space score
        score = model(x, t, cond or {})

        # real -> Fourier
        score_hat = torch.fft.rfft(score, dim=-1)
        return score_hat

    return score_fn
