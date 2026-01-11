from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class PcaFit:
    n_components_used: int
    explained_variance_ratio: np.ndarray
    pca: PCA


def fit_pca_with_cap(
    X: np.ndarray,
    *,
    variance_threshold: float,
    max_components: int,
) -> tuple[np.ndarray, PcaFit]:
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if not (0 < variance_threshold <= 1):
        raise ValueError("variance_threshold must be in (0, 1]")
    if max_components < 1:
        raise ValueError("max_components must be >= 1")

    n_features = X.shape[1]
    cap = min(max_components, n_features)

    pca = PCA(n_components=cap, svd_solver="full")
    X_pca = pca.fit_transform(X)

    cum = np.cumsum(pca.explained_variance_ratio_)
    n_needed = int(np.searchsorted(cum, variance_threshold) + 1)
    n_used = min(n_needed, X_pca.shape[1])
    X_reduced = X_pca[:, :n_used]

    fit = PcaFit(
        n_components_used=n_used,
        explained_variance_ratio=pca.explained_variance_ratio_[:n_used].copy(),
        pca=pca,
    )
    return X_reduced, fit

