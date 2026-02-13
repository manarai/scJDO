
# Fourier Analysis Tutorial (Paul15)

This tutorial extends the classic **Paul et al. 2015 hematopoiesis** example by adding an FFT‑based analysis over **DPT pseudotime**. It illustrates how low‑pass reconstruction (via iFFT) highlights smooth lineage programs while attenuating high‑frequency noise.

## Highlights
- Uniform resampling over pseudotime (required for FFT)
- rFFT power spectra for marker genes (e.g., Gata1, Elane)
- Low‑pass iFFT smoothing vs. raw trajectories
- Optional branch‑wise spectral comparison (erythroid vs neutrophil)

## Run
```bash
pip install scanpy matplotlib numpy
jupyter notebook notebooks/tutorial_paul15_hematopoiesis_fourier.ipynb
```

## References
- Scanpy Paul15 tutorial (paga‑paul15)
- Paul et al., Cell (2015)
- Scanpy (PyPI)
