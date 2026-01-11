import numpy as np

from paperjn.stats.fdr import bh_fdr


def test_bh_fdr_basic() -> None:
    rejected, q = bh_fdr([0.01, 0.02, 0.03, 0.2], alpha=0.05)
    assert rejected.tolist() == [True, True, True, False]
    assert np.allclose(q, [0.04, 0.04, 0.04, 0.2])

