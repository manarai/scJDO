from __future__ import annotations
from typing import Dict, Optional

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = object

from ..Fourier.losses_spectral import spectral_smoothness_loss


def _require_torch():
    if torch is None:
        raise ImportError("PyTorch required for SpectralDiffusionTrainer.")


class SpectralDiffusionTrainer:
    """
    VE-style denoising score matching trainer with optional spectral regularization.

    Training objective:
        x_t = x_0 + sigma(t) * eps
        target score = -(x_t - x_0) / sigma(t)^2 = -eps / sigma(t)

    The model should return a real-space score estimate:
        score_theta(x_t, t, cond) -> (B, G)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer,
        device: str = "cpu",
        sigma_min: float = 0.01,
        sigma_max: float = 1.0,
        spectral_weight: float = 0.01,
        grad_clip: Optional[float] = 1.0,
    ):
        _require_torch()
        self.model = model.to(device)
        self.opt = optimizer
        self.device = device
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.spectral_weight = float(spectral_weight)
        self.grad_clip = grad_clip

    def sigma(self, t: "torch.Tensor") -> "torch.Tensor":
        """
        Exponential noise schedule:
            sigma(t) = sigma_min * (sigma_max / sigma_min) ** t
        """
        ratio = self.sigma_max / self.sigma_min
        return self.sigma_min * (ratio ** t)

    def train_step(self, x: "torch.Tensor", cond: Optional[Dict] = None) -> Dict[str, float]:
        _require_torch()

        if x.ndim != 2:
            raise ValueError(f"Expected x with shape (batch, genes), got {tuple(x.shape)}")

        x = x.to(self.device)
        if not torch.is_floating_point(x):
            x = x.float()

        B = x.shape[0]
        t = torch.rand(B, device=self.device, dtype=x.dtype)

        sigma_t = self.sigma(t).view(B, 1)
        eps = torch.randn_like(x)
        x_noisy = x + sigma_t * eps

        # Denoising score matching target in real space.
        target = -eps / sigma_t

        pred = self.model(x_noisy, t, cond or {})

        if pred.shape != target.shape:
            raise ValueError(f"Model output shape {tuple(pred.shape)} does not match target shape {tuple(target.shape)}")

        loss_dsm = torch.mean((pred - target) ** 2)

        loss_spec = torch.tensor(0.0, device=self.device, dtype=x.dtype)
        if self.spectral_weight > 0:
            pred_hat = torch.fft.rfft(pred, dim=-1)
            loss_spec = spectral_smoothness_loss(pred_hat, weight=self.spectral_weight)

        loss = loss_dsm + loss_spec

        self.opt.zero_grad(set_to_none=True)
        loss.backward()

        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.opt.step()

        return {
            "loss": float(loss.detach().cpu()),
            "dsm": float(loss_dsm.detach().cpu()),
            "spectral": float(loss_spec.detach().cpu()),
            "sigma_mean": float(sigma_t.mean().detach().cpu()),
        }

    @torch.no_grad()
    def fit(self, data_loader, epochs: int = 1, cond_getter=None):
        """
        Convenience loop for DataLoader training.
        cond_getter(batch) -> optional conditioning dict
        """
        history = []
        self.model.train()

        for _ in range(epochs):
            for batch in data_loader:
                x = batch[0] if isinstance(batch, (tuple, list)) else batch
                cond = cond_getter(batch) if cond_getter is not None else None
                stats = self.train_step(x, cond=cond)
                history.append(stats)

        return history
