from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def is_ppa_subtype(subtype: str, markers: list[str]) -> bool:
    s = subtype.lower()
    return any(m.lower() in s for m in markers)


@dataclass(frozen=True)
class TwoByTwo:
    a: float  # exposed + outcome (PPA in language cluster)
    b: float  # exposed + no outcome
    c: float  # unexposed + outcome
    d: float  # unexposed + no outcome


@dataclass(frozen=True)
class EffectSizes:
    ppa_share_cluster: float
    ppa_share_stage: float
    rr: float
    rr_ci95_low: float
    rr_ci95_high: float
    or_: float
    or_ci95_low: float
    or_ci95_high: float


def _safe_log(x: float) -> float:
    return float(np.log(x))


def _wald_ci_log(effect: float, se: float) -> tuple[float, float]:
    z = 1.959963984540054  # approx norm.ppf(0.975)
    lo = np.exp(_safe_log(effect) - z * se)
    hi = np.exp(_safe_log(effect) + z * se)
    return float(lo), float(hi)


def compute_effect_sizes(table: TwoByTwo) -> EffectSizes:
    # Shares for reporting (raw counts; no correction).
    n_cluster_raw = table.a + table.b
    n_stage_raw = table.a + table.b + table.c + table.d
    if n_cluster_raw <= 0 or n_stage_raw <= 0:
        raise ValueError("Invalid 2x2 table: non-positive totals")
    p_cluster_raw = table.a / n_cluster_raw
    p_stage_raw = (table.a + table.c) / n_stage_raw

    # Haldane–Anscombe correction for stable RR/OR + CIs (handles zeros).
    a = table.a + 0.5
    b = table.b + 0.5
    c = table.c + 0.5
    d = table.d + 0.5

    n_cluster = a + b
    n_stage = a + b + c + d

    rr = (a / n_cluster) / ((a + c) / n_stage)
    se_log_rr = np.sqrt((1 / a) - (1 / n_cluster) + (1 / (a + c)) - (1 / n_stage))
    rr_lo, rr_hi = _wald_ci_log(rr, float(se_log_rr))

    or_ = (a * d) / (b * c)
    se_log_or = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    or_lo, or_hi = _wald_ci_log(or_, float(se_log_or))

    return EffectSizes(
        ppa_share_cluster=float(p_cluster_raw),
        ppa_share_stage=float(p_stage_raw),
        rr=float(rr),
        rr_ci95_low=rr_lo,
        rr_ci95_high=rr_hi,
        or_=float(or_),
        or_ci95_low=or_lo,
        or_ci95_high=or_hi,
    )
