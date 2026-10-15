"""Step 1 — positional layer (embedding).

Produces a latent position ``z_c`` for every cell. Three backends:

- ``"scvi"`` — the v2 default: scVI encoder, frozen after fit.
- ``"fa"``   — Factor Analysis (sklearn).
- ``"pca"``  — PCA (sklearn).

The encoder / loadings are **frozen after fitting**: new data is projected in
with the stored transform. For PCA / FA this is a linear ``transform``; for
scVI we store the fitted ``scvi.model.SCVI`` instance.

Normalisation
-------------
Normalisation lives in this layer only — scVI operates on raw counts
internally (it is a count model), while FA / PCA are given log1p-normalised
counts so the Euclidean geometry downstream makes sense. The operator
layer (:mod:`scjdo.v2.field`) never touches normalised data in Phase C; it
takes raw counts and lets the count-likelihood rate model handle them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np


EmbedBackend = Literal["scvi", "fa", "pca"]


@dataclass
class EmbeddingResult:
    """Fitted embedding + frozen transform."""

    Z: np.ndarray                 # (n_cells, n_latent)
    backend: EmbedBackend
    n_latent: int
    transform: Any                # fitted sklearn / scvi object with .transform or .get_latent_representation
    extras: dict                  # arbitrary diagnostics (e.g. explained_variance, elbo)

    def project(self, X: np.ndarray) -> np.ndarray:
        """Project new data through the frozen transform."""
        if self.backend in ("pca", "fa"):
            return self.transform.transform(X).astype(np.float32)
        raise NotImplementedError(f"project() for backend={self.backend} requires an AnnData — call fit_embedding again.")


def _fit_pca(X: np.ndarray, n_latent: int, seed: int) -> EmbeddingResult:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_latent, random_state=seed)
    Z = pca.fit_transform(X).astype(np.float32)
    return EmbeddingResult(
        Z=Z, backend="pca", n_latent=n_latent, transform=pca,
        extras={"explained_variance_ratio": pca.explained_variance_ratio_.tolist()},
    )


def _fit_fa(X: np.ndarray, n_latent: int, seed: int) -> EmbeddingResult:
    from sklearn.decomposition import FactorAnalysis
    fa = FactorAnalysis(n_components=n_latent, random_state=seed, max_iter=1000)
    Z = fa.fit_transform(X).astype(np.float32)
    return EmbeddingResult(
        Z=Z, backend="fa", n_latent=n_latent, transform=fa, extras={},
    )


def _fit_scvi(
    adata,
    n_latent: int,
    seed: int,
    batch_key: Optional[str],
    n_epochs: int,
) -> EmbeddingResult:
    import scvi
    scvi.settings.seed = seed
    scvi.model.SCVI.setup_anndata(adata, batch_key=batch_key)
    model = scvi.model.SCVI(adata, n_latent=n_latent)
    model.train(max_epochs=n_epochs, check_val_every_n_epoch=max(1, n_epochs // 10))
    Z = model.get_latent_representation().astype(np.float32)
    return EmbeddingResult(
        Z=Z, backend="scvi", n_latent=n_latent, transform=model,
        extras={"n_epochs": n_epochs, "elbo_train": float(model.history["elbo_train"].iloc[-1, 0])
                if "elbo_train" in model.history else float("nan")},
    )


def fit_embedding(
    X: np.ndarray | None = None,
    *,
    adata=None,
    backend: EmbedBackend = "scvi",
    n_latent: int = 20,
    seed: int = 42,
    batch_key: Optional[str] = None,
    n_epochs: int = 400,
) -> EmbeddingResult:
    """Fit the positional layer.

    Parameters
    ----------
    X
        Dense cells × features matrix for ``pca`` / ``fa``. Expected to be
        log1p-normalised. Required for those backends; ignored for ``scvi``.
    adata
        AnnData with raw counts in ``.X``. Required for ``scvi``.
    backend
        ``"scvi"`` (default), ``"fa"``, or ``"pca"``.
    n_latent
        Latent dimension.
    seed
        Reproducibility seed.
    batch_key
        scVI batch covariate; ignored for pca/fa.
    n_epochs
        scVI training epochs.
    """
    if backend == "pca":
        if X is None:
            raise ValueError("backend='pca' requires an X matrix.")
        return _fit_pca(X, n_latent, seed)
    if backend == "fa":
        if X is None:
            raise ValueError("backend='fa' requires an X matrix.")
        return _fit_fa(X, n_latent, seed)
    if backend == "scvi":
        if adata is None:
            raise ValueError("backend='scvi' requires an AnnData with raw counts in .X.")
        return _fit_scvi(adata, n_latent, seed, batch_key, n_epochs)
    raise ValueError(f"Unknown backend={backend!r}; expected one of 'scvi', 'fa', 'pca'.")
