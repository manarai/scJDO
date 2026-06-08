from __future__ import annotations

from typing import Optional, Callable

try:
    import torch
except Exception as e:  # pragma: no cover
    torch = None

from .kspace_samplers import KSpaceEulerMaruyama, KSpaceHeun, make_fourier_score_fn


def _require_torch():
    if torch is None:
        raise ImportError("PyTorch is required for Fourier-domain sampling.")


def sigma_schedule(t, sigma_min: float = 0.01, sigma_max: float = 1.0):
    """Shared VE-style noise schedule used by both training and sampling."""
    _require_torch()
    return sigma_min * (sigma_max / sigma_min) ** t


def sample(
    model,
    shape,
    steps: int = 100,
    device: str = "cpu",
    use_heun: bool = True,
    sigma_min: float = 0.01,
    sigma_max: float = 1.0,
    cond: Optional[dict] = None,
    init_scale: float = 1.0,
):
    """
    Sample from a Fourier-space reverse process using a real-space score model.

    Parameters
    ----------
    model:
        A real-space score model with signature model(x_real, t, cond) -> score_real.
        It is expected to expose ``gene_dim`` for exact iFFT reconstruction.
    shape:
        Tuple (batch_size, gene_dim).
    steps:
        Number of reverse-time integration steps.
    device:
        Device used for the latent trajectory.
    use_heun:
        If True, use KSpaceHeun; otherwise use KSpaceEulerMaruyama.
    sigma_min, sigma_max:
        Shared VE-style noise schedule parameters.
    cond:
        Optional conditioning dictionary passed to the model.
    init_scale:
        Initial Gaussian scale for the starting sample in real space.

    Returns
    -------
    torch.Tensor
        Final samples in real space, shape (batch_size, gene_dim).
    """
    _require_torch()

    if not isinstance(shape, (tuple, list)) or len(shape) != 2:
        raise ValueError("shape must be a (batch_size, gene_dim) tuple")

    batch_size, gene_dim = int(shape[0]), int(shape[1])
    if batch_size <= 0 or gene_dim <= 0:
        raise ValueError("shape entries must be positive")

    device = torch.device(device)
    model = model.to(device)
    model.eval()

    # Start from Gaussian noise in real space at the largest noise level.
    x = torch.randn(batch_size, gene_dim, device=device) * float(init_scale)
    x_hat = torch.fft.rfft(x, dim=-1)

    g_fn: Callable[[float], float] = lambda t: sigma_schedule(
        t, sigma_min=sigma_min, sigma_max=sigma_max
    )
    sampler = KSpaceHeun(g_fn) if use_heun else KSpaceEulerMaruyama(g_fn)
    score_fn = make_fourier_score_fn(model)

    dt = -1.0 / float(steps)
    t = 1.0

    with torch.no_grad():
        for _ in range(steps):
            x_hat = sampler.step(x_hat, t, dt, score_fn, cond=cond)
            t += dt

    # Return real-space samples.
    return torch.fft.irfft(x_hat, n=gene_dim, dim=-1)
