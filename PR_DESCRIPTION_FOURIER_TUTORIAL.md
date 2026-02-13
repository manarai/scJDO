
# Add Fourier tutorial: `notebooks/tutorial_paul15_hematopoiesis_fourier.ipynb`

**Date:** 2026-02-13

## Summary
This PR adds a new tutorial notebook that demonstrates **Fourier analysis along pseudotime** on the classic **Paul et al. 2015** hematopoiesis dataset using Scanpy. It compares raw vs. low‑pass (iFFT) reconstructed trajectories for lineage‑marker genes and includes optional branch‑specific spectra.

- Dataset & pseudotime steps mirror Scanpy's Paul15 tutorial (neighbors → diffusion map → DPT).  
- The FFT pipeline is **resample (uniform grid) → rFFT → power spectrum → low‑pass iFFT**.

## Files added
- `notebooks/tutorial_paul15_hematopoiesis_fourier.ipynb`
- `docs/FOURIER_TUTORIAL_README.md` (short overview & references)

## How to run
```bash
# in your scIDiff_V2 repo
conda activate scidiff   # or your env
pip install scanpy matplotlib numpy
jupyter notebook notebooks/tutorial_paul15_hematopoiesis_fourier.ipynb
```

## Notes
- The tutorial relies on `sc.datasets.paul15()` to fetch the dataset (internet may be required on first run; then cached locally).
- The notebook is self‑contained and does **not** change any core code.

## References
- Scanpy Paul15 tutorial (paga‑paul15)  
- Paul et al., Cell (2015) myeloid progenitors  
- Scanpy (PyPI)
