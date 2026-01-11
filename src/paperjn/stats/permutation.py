from __future__ import annotations

import numpy as np


def permutation_p_value_ge(observed: float, permuted: np.ndarray) -> float:
    """One-sided p-value for enrichment: P(perm_stat >= obs_stat)."""
    if permuted.ndim != 1:
        raise ValueError("permuted must be 1D")
    b = int(permuted.shape[0])
    return float((1 + np.sum(permuted >= observed)) / (1 + b))

