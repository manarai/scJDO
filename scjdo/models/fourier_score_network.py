from __future__ import annotations
from typing import Dict, Optional

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = object


def _require_torch():
    if torch is None:
        raise ImportError("PyTorch required for spectral score models.")


def _sinusoidal_time_embedding(t: "torch.Tensor", dim: int) -> "torch.Tensor":
    """
    Standard sinusoidal embedding for diffusion time.
    t: (B,)
    returns: (B, dim)
    """
    _require_torch()
    if t.ndim != 1:
        t = t.view(-1)

    half = dim // 2
    if half == 0:
        return t[:, None]

    device = t.device
    dtype = t.dtype
    freqs = torch.exp(
        torch.arange(half, device=device, dtype=dtype)
        * (-torch.log(torch.tensor(10000.0, device=device, dtype=dtype)) / max(half - 1, 1))
    )
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros(t.shape[0], 1, device=device, dtype=dtype)], dim=-1)
    return emb


def _cond_to_tensor(
    cond: Optional[Dict],
    batch_size: int,
    device,
    dtype,
) -> Optional["torch.Tensor"]:
    """
    Flatten tensor-valued conditioning entries into a single feature matrix.
    Non-tensor entries are ignored.
    """
    _require_torch()
    if not cond:
        return None

    parts = []
    for _, v in sorted(cond.items(), key=lambda kv: kv[0]):
        if not torch.is_tensor(v):
            continue

        v = v.to(device=device, dtype=dtype)

        if v.ndim == 0:
            v = v.view(1, 1).expand(batch_size, 1)
        elif v.ndim == 1:
            if v.shape[0] == batch_size:
                v = v[:, None]
            else:
                continue
        else:
            if v.shape[0] != batch_size:
                continue
            v = v.reshape(batch_size, -1)

        parts.append(v)

    if not parts:
        return None
    return torch.cat(parts, dim=-1)


class MultiBandScoreNet(nn.Module):
    """
    Spectral score network that:
      1) transforms x -> Fourier domain
      2) processes low/mid/high bands independently
      3) reconstructs a real-space score

    Forward signature:
        score = model(x_noisy, t, cond=None)

    where:
        x_noisy: (B, G) real input
        t: scalar or (B,)
        cond: optional dict of tensor conditioning features
    """

    def __init__(
        self,
        gene_dim: int,
        hidden_dim: int = 512,
        band_splits=(0.2, 0.6),
        time_embed_dim: int = 64,
        cond_dim: int = 0,
    ):
        super().__init__()
        _require_torch()

        if gene_dim < 2:
            raise ValueError("gene_dim must be at least 2")

        self.gene_dim = int(gene_dim)
        self.band_splits = band_splits
        self.time_embed_dim = int(time_embed_dim)
        self.cond_dim = int(cond_dim)

        K = gene_dim // 2 + 1  # rFFT size
        self.K = K

        # Ensure non-empty bands whenever possible.
        if K >= 3:
            k1 = max(1, min(K - 2, int(K * band_splits[0])))
            k2 = max(k1 + 1, min(K - 1, int(K * band_splits[1])))
        else:
            k1 = 1
            k2 = max(2, K)

        self.k1, self.k2 = k1, k2

        def make_mlp(in_dim: int, out_dim: int):
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, out_dim),
            )

        band_ctx = self.time_embed_dim + self.cond_dim

        self.low = make_mlp(2 * self.k1 + band_ctx, 2 * self.k1)
        self.mid = make_mlp(2 * max(1, self.k2 - self.k1) + band_ctx, 2 * max(1, self.k2 - self.k1))
        self.high = make_mlp(2 * max(1, self.K - self.k2) + band_ctx, 2 * max(1, self.K - self.k2))

    def forward(self, x: "torch.Tensor", t, cond: Optional[Dict] = None) -> "torch.Tensor":
        """
        x: (B, G) real-space noisy input
        t: scalar float / scalar tensor / (B,)
        cond: optional dict of tensor features

        returns:
            score: (B, G) real-space score estimate
        """
        _require_torch()

        if x.ndim != 2:
            raise ValueError(f"Expected x with shape (batch, genes), got {tuple(x.shape)}")

        if not torch.is_floating_point(x):
            x = x.float()

        B, G = x.shape
        if G != self.gene_dim:
            raise ValueError(f"Expected gene_dim={self.gene_dim}, got input dim {G}")

        device = x.device
        dtype = x.dtype

        # Normalize t to shape (B,)
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=device, dtype=dtype)
        if t.ndim == 0:
            t = t.expand(B)
        elif t.ndim == 1 and t.shape[0] != B:
            raise ValueError(f"t must be scalar or shape (B,), got {tuple(t.shape)}")
        elif t.ndim > 1:
            t = t.reshape(B)

        t = t.to(device=device, dtype=dtype)
        t_emb = _sinusoidal_time_embedding(t, self.time_embed_dim).to(device=device, dtype=dtype)

        c = _cond_to_tensor(cond, B, device, dtype)
        if self.cond_dim > 0:
            if c is None:
                c = torch.zeros(B, self.cond_dim, device=device, dtype=dtype)
            elif c.shape[1] != self.cond_dim:
                raise ValueError(
                    f"cond_dim={self.cond_dim} but conditioning features have width {c.shape[1]}"
                )
        else:
            c = None

        # Fourier transform of noisy input
        x_hat = torch.fft.rfft(x, dim=-1)  # (B, K)
        ri = torch.stack([x_hat.real, x_hat.imag], dim=-1)  # (B, K, 2)

        ctx = t_emb if c is None else torch.cat([t_emb, c], dim=-1)

        # Low band
        low = ri[:, : self.k1, :].reshape(B, -1)
        low_out = self.low(torch.cat([low, ctx], dim=-1))
        low_out = low_out.view(B, self.k1, 2)

        # Mid band
        mid_width = max(1, self.k2 - self.k1)
        mid = ri[:, self.k1 : self.k2, :]
        if mid.shape[1] == 0:
            mid = torch.zeros(B, mid_width, 2, device=device, dtype=dtype)
        mid = mid.reshape(B, -1)
        mid_out = self.mid(torch.cat([mid, ctx], dim=-1))
        mid_out = mid_out.view(B, mid_width, 2)

        # High band
        high_width = max(1, self.K - self.k2)
        high = ri[:, self.k2 :, :]
        if high.shape[1] == 0:
            high = torch.zeros(B, high_width, 2, device=device, dtype=dtype)
        high = high.reshape(B, -1)
        high_out = self.high(torch.cat([high, ctx], dim=-1))
        high_out = high_out.view(B, high_width, 2)

        # Reassemble spectrum
        spec_ri = torch.cat([low_out, mid_out, high_out], dim=1)

        # If we padded a band because K was small, trim back to K
        if spec_ri.shape[1] > self.K:
            spec_ri = spec_ri[:, : self.K, :]
        elif spec_ri.shape[1] < self.K:
            pad = self.K - spec_ri.shape[1]
            spec_ri = torch.cat(
                [spec_ri, torch.zeros(B, pad, 2, device=device, dtype=dtype)],
                dim=1,
            )

        score_hat = torch.complex(spec_ri[..., 0], spec_ri[..., 1])
        score = torch.fft.irfft(score_hat, n=self.gene_dim, dim=-1)
        return score
