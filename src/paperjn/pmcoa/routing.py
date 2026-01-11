from __future__ import annotations

from dataclasses import dataclass
import re

import pandas as pd


_EXCLUDE_SEC_RE = re.compile(
    r"\b(methods?|materials?|statistics?|references?|acknowledg(?:e)?ments?|funding|disclosure|conflicts?|ethics)\b",
    flags=re.IGNORECASE,
)
_MUST_INCLUDE_SEC_RE = re.compile(r"\b(cases?|patients?)\b", flags=re.IGNORECASE)
_CANDIDATE_SEC_RE = re.compile(
    r"\b(cases?|patients?|clinical|presentations?|history|course|symptoms?|phenotyp(?:e|ic)?|exam(?:ination)?|neurolog(?:y|ical)?)\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class RoutedSections:
    must_include: list[str]
    candidates: list[str]


def route_candidate_sections(blocks: pd.DataFrame) -> RoutedSections:
    """Rule-based, conservative routing to clinical narrative sections by headings."""
    if blocks.empty:
        return RoutedSections(must_include=[], candidates=[])

    df = blocks.copy()
    df = df[df["source"] == "body"].copy()
    if df.empty:
        return RoutedSections(must_include=[], candidates=[])

    secs = (
        df.groupby("sec_path_str", as_index=False)
        .agg(total_chars=("n_chars", "sum"), n_blocks=("block_id", "size"))
        .sort_values("total_chars", ascending=False)
    )

    must: list[str] = []
    cand: list[str] = []

    for sec in secs["sec_path_str"].astype(str).tolist():
        if not sec:
            continue
        # Must-include (case/patient) overrides generic exclusions like "Methods".
        if _MUST_INCLUDE_SEC_RE.search(sec):
            must.append(sec)
            continue
        if _EXCLUDE_SEC_RE.search(sec):
            continue
        if _CANDIDATE_SEC_RE.search(sec):
            cand.append(sec)

    # Cap candidates to keep LLM routing inputs small; order already by size.
    cand = cand[:40]
    return RoutedSections(must_include=must[:20], candidates=cand)


def select_blocks_for_llm(
    blocks: pd.DataFrame,
    *,
    selected_sec_paths: list[str],
    max_blocks: int = 140,
    max_total_chars: int = 60_000,
) -> pd.DataFrame:
    df = blocks.copy()
    df = df[df["source"] == "body"].copy()
    if selected_sec_paths:
        df = df[df["sec_path_str"].isin(selected_sec_paths)].copy()
    if df.empty:
        return df

    df = df.sort_values("block_index").reset_index(drop=True)

    # Enforce caps for cost control.
    out_rows = []
    total = 0
    for _, r in df.iterrows():
        n = int(r.get("n_chars") or 0)
        if out_rows and (len(out_rows) >= max_blocks or total + n > max_total_chars):
            break
        out_rows.append(r)
        total += n

    return pd.DataFrame(out_rows).reset_index(drop=True)
