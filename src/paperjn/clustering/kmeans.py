from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans


@dataclass(frozen=True)
class KMeansFit:
    k: int
    labels: np.ndarray
    centers: np.ndarray
    inertia: float


def fit_kmeans(X: np.ndarray, *, k: int, random_seed: int) -> KMeansFit:
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if k < 2:
        raise ValueError("k must be >= 2")
    if X.shape[0] < k:
        raise ValueError(f"Need at least k samples (n={X.shape[0]}, k={k})")

    model = KMeans(n_clusters=k, random_state=random_seed, n_init=10)
    labels = model.fit_predict(X)
    return KMeansFit(k=k, labels=labels, centers=model.cluster_centers_, inertia=float(model.inertia_))

