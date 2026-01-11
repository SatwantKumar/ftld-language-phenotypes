from __future__ import annotations

import numpy as np


def bh_fdr(p_values: list[float], *, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini–Hochberg FDR control.

    Returns (rejected, q_values) with the same order as input p-values.
    """
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1:
        raise ValueError("p_values must be 1D")
    if p.size == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0, 1]")
    if np.any(~np.isfinite(p)):
        raise ValueError("p_values must be finite")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p_values must be in [0, 1]")

    m = int(p.size)
    order = np.argsort(p)
    p_sorted = p[order]

    ranks = np.arange(1, m + 1, dtype=float)
    q_sorted = p_sorted * m / ranks
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    q = np.empty_like(q_sorted)
    q[order] = q_sorted
    rejected = q <= alpha
    return rejected.astype(bool), q.astype(float)
