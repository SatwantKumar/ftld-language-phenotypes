from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import pandas as pd
from pydantic import ValidationError

from paperjn.llm.openai_client import OpenAIClient
from paperjn.pmcoa.extraction_schema import GPT52ExtractionOutput, NanoRouteOutput
from paperjn.pmcoa.routing import route_candidate_sections, select_blocks_for_llm


def _read_text(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def build_section_previews(
    blocks: pd.DataFrame,
    *,
    sec_paths: list[str],
    max_blocks_per_sec: int = 2,
    max_preview_chars: int = 450,
) -> list[dict[str, Any]]:
    df = blocks.copy()
    df = df[df["source"] == "body"].copy()
    if sec_paths:
        df = df[df["sec_path_str"].isin(sec_paths)].copy()
    if df.empty:
        return []

    df = df.sort_values(["sec_path_str", "block_index"]).reset_index(drop=True)
    previews: list[dict[str, Any]] = []
    for sec, g in df.groupby("sec_path_str", sort=False):
        g = g.sort_values("block_index")
        sample_text = "\n".join(g["text"].astype(str).head(int(max_blocks_per_sec)).tolist()).strip()
        sample_text = sample_text[: int(max_preview_chars)]
        previews.append(
            {
                "section_path": str(sec),
                "n_blocks": int(len(g)),
                "total_chars": int(g["n_chars"].sum()),
                "preview": sample_text,
            }
        )

    previews.sort(key=lambda x: (x.get("total_chars") or 0), reverse=True)
    return previews


def heuristic_pass_a(previews: list[dict[str, Any]]) -> NanoRouteOutput:
    sec_paths = [str(p.get("section_path")) for p in previews if p.get("section_path")]
    joined = " ".join(sec_paths).lower()
    is_case_like = any(k in joined for k in ["case", "patient", "presentation", "clinical"])
    selected = [p for p in sec_paths if any(k in p.lower() for k in ["case", "patient", "presentation", "course"])]
    return NanoRouteOutput(
        is_case_like=bool(is_case_like),
        n_cases_est=None,
        selected_sec_paths=selected[:12],
        case_markers_found=[],
    )


def heuristic_pass_b(selected_blocks: pd.DataFrame) -> GPT52ExtractionOutput:
    if selected_blocks.empty:
        return GPT52ExtractionOutput(n_cases_est=None, segments=[])
    bids = selected_blocks["block_id"].astype(str).head(4).tolist()
    return GPT52ExtractionOutput(
        n_cases_est=None,
        segments=[
            {
                "case_id": "case_unknown",
                "segment_type": "clinical_features_summary",
                "block_ids": bids,
                "include_for_embedding": True,
            }
        ],
    )


# Deterministic guardrail: require patient-level narrative anchors within LLM-selected sections.
_STRICT_SECTION_RE = re.compile(
    r"\b("
    r"case\s*(reports?|presentations?|series|stud(?:y|ies))\b"
    r"|description\s+of\s+the\s+case\b"
    r"|clinical\s+case\b"
    r"|proband\b"
    r"|index\s+case\b"
    r"|patient(?:s)?\b"
    r")",
    flags=re.IGNORECASE,
)

_ANCHOR_RE_LIST: list[tuple[str, re.Pattern[str]]] = [
    (
        "age_year_old",
        re.compile(
            r"\b(?:a|an)\s+\d{1,3}(?:[-\u2010\u2011\u2012\u2013\u2014\u2212 ]+)year(?:[-\u2010\u2011\u2012\u2013\u2014\u2212 ]+)old\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "case_or_patient_number",
        re.compile(r"\b(?:case|patient)\s*(?:#?\s*)?(?:\d+|[ivx]+)\b", flags=re.IGNORECASE),
    ),
    (
        "patient_was_years_old",
        re.compile(
            r"\b(?:the|this)\s+patient\b.{0,80}\b(?:was|is)\b.{0,40}\b\d{1,3}\s+years?\s+old\b",
            flags=re.IGNORECASE,
        ),
    ),
    ("proband", re.compile(r"\bproband\b", flags=re.IGNORECASE)),
    ("index_case", re.compile(r"\bindex\s+case\b", flags=re.IGNORECASE)),
    (
        "patient_presented_or_admitted",
        re.compile(
            r"\b(?:the\s+patient|this\s+patient)\b.{0,80}\b(presented|was admitted|was hospitalized|developed)\b",
            flags=re.IGNORECASE,
        ),
    ),
]


def pass_a_case_like_guardrail(
    out: NanoRouteOutput,
    blocks: pd.DataFrame,
    *,
    max_scan_chars_per_section: int = 12_000,
) -> tuple[bool, list[str]]:
    """Return (passed, matched_markers).

    Designed to reduce cohort-style false positives by requiring patient-level narrative anchors
    in at least one LLM-selected section.
    """
    if not out.selected_sec_paths:
        return False, ["no_selected_sections"]
    if blocks.empty:
        return False, ["no_blocks"]

    df = blocks.copy()
    df = df[df["source"] == "body"].copy()
    if df.empty:
        return False, ["no_body_blocks"]

    for sec in out.selected_sec_paths:
        if not sec or not _STRICT_SECTION_RE.search(str(sec)):
            continue
        sec_df = df[df["sec_path_str"] == sec].sort_values("block_index")
        if sec_df.empty:
            continue
        text = " ".join(sec_df["text"].astype(str).tolist())
        text = text[: int(max_scan_chars_per_section)]

        matched = [f"{name}@{sec}" for name, pat in _ANCHOR_RE_LIST if pat.search(text)]
        if matched:
            return True, matched

    return False, ["no_narrative_anchor"]


def run_pass_a_route(
    *,
    client: OpenAIClient,
    prompt_path: Path,
    title: str | None,
    article_type: str | None,
    blocks: pd.DataFrame,
    dry_run: bool,
    model: str = "gpt-5-nano",
    endpoint: str = "responses",
) -> tuple[NanoRouteOutput, dict[str, Any] | None, dict[str, Any]]:
    routed = route_candidate_sections(blocks)
    sec_paths = list(dict.fromkeys(routed.must_include + routed.candidates))  # stable unique
    previews = build_section_previews(blocks, sec_paths=sec_paths)
    payload = {
        "title": title,
        "article_type": article_type,
        "candidate_sections": previews,
    }

    if dry_run:
        return heuristic_pass_a(previews), None, payload

    sys_prompt = _read_text(prompt_path)
    user_prompt = json.dumps(payload, ensure_ascii=False)

    obj, raw = client.call_json(
        model=model,
        system_prompt=sys_prompt,
        user_prompt=user_prompt,
        max_output_tokens=900,
        endpoint="chat_completions" if endpoint == "chat_completions" else "responses",
    )
    try:
        out = NanoRouteOutput.model_validate(obj)
    except ValidationError as exc:
        raise RuntimeError(f"Pass A output failed schema validation: {exc}") from exc

    # Sanitize: enforce section-path membership to avoid accidental hallucinations.
    allowed = {str(p.get("section_path")) for p in previews if p.get("section_path") is not None}
    selected = [s for s in out.selected_sec_paths if s in allowed]
    if selected != out.selected_sec_paths:
        out = out.model_copy(update={"selected_sec_paths": selected})

    # Deterministic inclusion boundary: patient-level narrative only.
    if out.is_case_like:
        passed, markers = pass_a_case_like_guardrail(out, blocks)
        if not passed:
            out = out.model_copy(
                update={
                    "is_case_like": False,
                    "n_cases_est": None,
                    "case_markers_found": list(out.case_markers_found)
                    + [f"guardrail_failed:{'|'.join(markers)}"],
                }
            )
    return out, raw, payload


def run_pass_b_extract(
    *,
    client: OpenAIClient,
    prompt_path: Path,
    title: str | None,
    blocks: pd.DataFrame,
    selected_sec_paths: list[str],
    dry_run: bool,
    model: str = "gpt-5.2",
    endpoint: str = "responses",
    max_blocks: int = 140,
    max_total_chars: int = 60_000,
) -> tuple[GPT52ExtractionOutput, dict[str, Any] | None, dict[str, Any]]:
    selected_blocks = select_blocks_for_llm(
        blocks, selected_sec_paths=selected_sec_paths, max_blocks=max_blocks, max_total_chars=max_total_chars
    )

    payload = {
        "title": title,
        "n_blocks": int(len(selected_blocks)),
        "blocks": selected_blocks[["block_id", "sec_path_str", "text"]].to_dict(orient="records"),
    }

    if dry_run:
        return heuristic_pass_b(selected_blocks), None, payload

    sys_prompt = _read_text(prompt_path)
    user_prompt = json.dumps(payload, ensure_ascii=False)
    obj, raw = client.call_json(
        model=model,
        system_prompt=sys_prompt,
        user_prompt=user_prompt,
        max_output_tokens=2000,
        endpoint="chat_completions" if endpoint == "chat_completions" else "responses",
    )
    try:
        out = GPT52ExtractionOutput.model_validate(obj)
    except ValidationError as exc:
        raise RuntimeError(f"Pass B output failed schema validation: {exc}") from exc

    valid_ids = set(selected_blocks["block_id"].astype(str).tolist())
    for seg in out.segments:
        bad = [b for b in seg.block_ids if b not in valid_ids]
        if bad:
            raise RuntimeError(f"Pass B returned unknown block_ids for {seg.case_id}/{seg.segment_type}: {bad[:3]}")

    return out, raw, payload
