
# Fourier Extension for scIDiff-v2

This document explains how to enable the **Fourier-domain diffusion** and **spectral Schrödinger Bridge** features introduced in this PR bundle.

## Quick Enable
1. Drop the `scIDiff/Fourier/` folder and the new files into your repo root.
2. Import the Fourier trainer/samplers in your experiment script or config.
3. Set the following in your YAML config:

```yaml
model:
  use_fourier: true
fourier:
  axis: -1
  band_splits: [0.2, 0.6]
sampler:
  type: kspace_heun   # or kspace_em
loss:
  spectral_weight: 0.1
```

## Minimal Code Snippet
```python
from scIDiff.Fourier.transforms import dft, idft
from scIDiff.Fourier.kspace_samplers import KSpaceEulerMaruyama
from scIDiff.Fourier.features import power_spectrum_features
from scIDiff.models.fourier_score_network import MultiBandScoreNet

# x: (batch, genes) torch.float32
y = dft(x)          # -> complex tensor in rfft form
ps = power_spectrum_features(y)
net = MultiBandScoreNet(gene_dim=x.shape[-1])
score_hat = net(y, t, cond={"c_fourier": ps})
```

## Schrödinger Bridge Hooks
```python
from scIDiff.transport.spectral_bridge import spectral_precondition_marginals
mu0_s, mu1_s = spectral_precondition_marginals(mu0, mu1, axis=-1, cutoff=0.6)
```

## Safety & Repro
- Default settings do not change existing behavior unless `model.use_fourier: true`.
- All Fourier functions are deterministic given the same random seeds.
