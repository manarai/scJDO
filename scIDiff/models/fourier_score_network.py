
from __future__ import annotations
from typing import Dict
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as e:  # pragma: no cover
    torch = None
    nn = object

class MultiBandScoreNet(nn.Module):
    """
    Simple multi-band score network operating on packed real-imag features.
    Input: x_hat (complex) -> pack real/imag as channels, process with MLP per band.
    """
    def __init__(self, gene_dim: int, hidden_dim: int = 512, band_splits=(0.2, 0.6)):
        super().__init__()
        if torch is None:
            raise ImportError("PyTorch required for MultiBandScoreNet")
        self.gene_dim = gene_dim
        self.band_splits = band_splits
        K = gene_dim // 2 + 1  # rfft size
        k1 = int(K * band_splits[0]); k2 = int(K * band_splits[1])
        self.k1, self.k2, self.K = k1, k2, K
        def mlp(nk):
            return nn.Sequential(
                nn.Linear(2*nk, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, 2*nk)
            )
        self.low = mlp(k1)
        self.mid = mlp(max(1, k2-k1))
        self.high = mlp(max(1, K-k2))

    def forward(self, x_hat, t: float, cond: Dict = None):
        # x_hat: complex spectrum (B, K)
        B, K = x_hat.shape[0], x_hat.shape[-1]
        ri = torch.stack([x_hat.real, x_hat.imag], dim=-1)  # (B, K, 2)
        # split bands
        low = ri[:, :self.k1, :].reshape(B, -1)
        mid = ri[:, self.k1:self.k2, :].reshape(B, -1)
        high = ri[:, self.k2:, :].reshape(B, -1)
        # per-band MLPs
        slo = self.low(low)
        smi = self.mid(mid)
        shi = self.high(high)
        # reshape back
        slo = slo.view(B, self.k1, 2)
        smi = smi.view(B, max(1, self.k2-self.k1), 2)
        shi = shi.view(B, max(1, self.K-self.k2), 2)
        # concat
        out = torch.cat([slo, smi, shi], dim=1)
        # pad/truncate to K
        if out.shape[1] < K:
            pad = K - out.shape[1]
            out = torch.cat([out, torch.zeros(B, pad, 2, device=out.device, dtype=out.dtype)], dim=1)
        elif out.shape[1] > K:
            out = out[:, :K, :]
        # convert back to complex
        return out[...,0] + 1j * out[...,1]
