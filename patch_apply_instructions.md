
# Applying This Patch

1. Create a feature branch in your local clone of `scIDiff-v2`:
```bash
git checkout -b feature/fourier-track
```
2. Copy the contents of this folder into the repo root (preserve paths).
3. Wire imports:
   - Add `from .models.fourier_score_network import MultiBandScoreNet` in `models/__init__.py`.
   - Register k-space samplers in your sampler registry (if any).
4. Run tests and a smoke training run on a tiny dataset.
5. Open a PR and tag reviewers.
