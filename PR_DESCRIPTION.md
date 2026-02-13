
# scIDiff-v2: Fourier-Domain Diffusion + Spectral Schrödinger Bridge (Feature Branch)

**Date:** 2026-02-13

## Summary
This PR adds an **optional Fourier track** to scIDiff-v2: diffusion, sampling, conditioning, and regularization **in frequency space**. It also introduces **spectral preconditioning** hooks for the Schrödinger Bridge / OT modules. The feature is opt-in and does **not** break existing APIs.

### Highlights
- `scIDiff/Fourier/` new module with:
  - `transforms.py`: DFT/IDFT wrappers on the genes axis (torch.fft), real↔complex packing utilities.
  - `bands.py`: multi-band splitting (low/mid/high) and merge utilities.
  - `features.py`: power spectrum & band-energy conditioning features.
  - `losses_spectral.py`: spectral smoothness and band-weighted score losses.
  - `kspace_samplers.py`: Euler–Maruyama and Heun samplers **in k-space** (frequency domain).
- `models/fourier_score_network.py`: multi-band score network head for frequency-domain score learning.
- `training/fourier_trainer.py`: thin trainer illustrating how to enable k-space diffusion end-to-end.
- `transport/spectral_bridge.py`: optional spectral preconditioning for Schrödinger Bridge paths and costs.
- `configs/fourier_default.yaml`: config flags to turn Fourier features on/off.
- Tests covering transforms, band logic, samplers, and spectral losses.
- Tutorial notebook `notebooks/06_fourier_domain_diffusion.ipynb`.

## Why
- scRNA-seq expression vectors contain **structured low-frequency signal** and **high-frequency technical noise**. Operating in the Fourier basis exposes this structure and stabilizes score learning and SB optimization.
- Prior work has shown Fourier-domain generative modeling (scGFT) and FFT-based spectral compression can improve quality and denoising for scRNA-seq. This PR brings those gains into the scIDiff-v2 pipeline.

## Integration Points (no breaking changes)
1. **Score networks**: optionally instantiate `MultiBandScoreNet` from `models/fourier_score_network.py`.
2. **Samplers**: select `KSpaceEulerMaruyama` or `KSpaceHeun` from `Fourier/kspace_samplers.py` through config.
3. **Conditioning**: pass `c_fourier` from `Fourier/features.py` into the model's conditioning dict.
4. **SB/OT**: call `spectral_precondition_marginals` from `transport/spectral_bridge.py` before Sinkhorn updates.

## Config
See `configs/fourier_default.yaml` for flags:
- `model.use_fourier: true|false`
- `fourier.axis: -1` (gene axis)
- `fourier.band_splits: [0.2, 0.6]`
- `sampler.type: kspace_em|kspace_heun|gene_space`
- `loss.spectral_weight: 0.1`

## Testing
Run minimal tests (do not require GPU):
```bash
pytest tests/test_fourier_transforms.py -q
pytest tests/test_kspace_samplers.py -q
pytest tests/test_spectral_losses.py -q
pytest tests/test_integration_minimal.py -q
```

## Notes
- This PR adds files only. To fully enable Fourier mode, wire imports in your existing `__init__.py`, sampler registry, and training scripts. See inline TODOs.
- Torch is assumed by modules but tests skip gracefully if `torch` is unavailable.
