"""Monkey-patch KNNVelocity._knn to use sklearn's NearestNeighbors,
which is O(log N × k × batch) via ball tree rather than the numpy
fallback's O(batch × N × D).  Enables N ≥ 50k fits with FAISS unavailable.
"""

from __future__ import annotations

import numpy as np


def enable_sklearn_knn_speedup():
    from scjdo.models.drift import KNNVelocity
    from sklearn.neighbors import NearestNeighbors

    orig_init = KNNVelocity.__init__

    def new_init(self, X_ref, V_ref, k=15, tau=1.0, use_faiss=None):
        # Skip faiss init entirely (crashes on py3.13 macOS at time of writing).
        orig_init(self, X_ref, V_ref, k=k, tau=tau, use_faiss=False)
        # Build a sklearn tree once
        self._sk = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(self._X_np)

    def new_knn(self, x):
        x_np = x.detach().cpu().numpy().astype(np.float32)
        D, I = self._sk.kneighbors(x_np)
        # sklearn returns Euclidean distance; KNNVelocity expects squared.
        return D ** 2, I

    KNNVelocity.__init__ = new_init
    KNNVelocity._knn = new_knn
