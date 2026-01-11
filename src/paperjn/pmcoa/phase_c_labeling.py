from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from paperjn.llm.openai_client import OpenAIClient
from paperjn.pmcoa.parse_jats_blocks import parse_jats_xml
from paperjn.pmcoa.phase_c_schema_v1 import (
    CaseLabelV1,
    SymptomTagName,
    SymptomTagV1,
)
from paperjn.pmcoa.registry import load_query_file


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _infer_project_root(path: Path) -> Path:
    p = Path(path).resolve()
    for _ in range(10):
        if (p / "src").exists() and (p / "docs").exists():
            return p
        p = p.parent
    return Path(path).resolve()


def _safe_json(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False)


def _read_last_statuses(labels_csv: Path) -> dict[tuple[str, str], str]:
    if not labels_csv.exists():
        return {}
    out: dict[tuple[str, str], str] = {}
    with labels_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pmcid = (row.get("pmcid") or "").strip()
            case_id = (row.get("case_id") or "").strip()
            if not pmcid or not case_id:
                continue
            out[(pmcid, case_id)] = (row.get("status") or "").strip()
    return out


def _split_from_year(year: Any, cutoff_year: int) -> str | None:
    try:
        y = int(year)
    except Exception:
        return None
    return "discovery" if y <= int(cutoff_year) else "confirmation"


_DX_TRIGGER_RE = re.compile(
    r"\b("
    r"frontotemporal|ftd|ftld|primary\s+progressive\s+aphasia|ppa\b|"
    r"semantic\s+variant|svppa|nonfluent|nfvppa|logopenic|lvppa|"
    r"progressive\s+supranuclear\s+palsy|psp\b|corticobasal|cbs\b|"
    r"motor\s+neuron|als\b|ftd[- ]?mnd|"
    r"diagnos(?:is|ed)|met\s+criteria|fulfill(?:ed|s)\s+criteria|"
    r"bvftd|grn\b|mapt\b|c9orf72\b|tdp-?43|tauopathy|pick\s+disease"
    r")\b",
    flags=re.IGNORECASE,
)


def _select_extra_snippets(
    blocks: pd.DataFrame,
    *,
    exclude_block_ids: set[str],
    max_snippets: int,
) -> list[dict[str, Any]]:
    """Pick a small number of high-yield diagnostic blocks (by regex trigger)."""
    if blocks.empty or max_snippets <= 0:
        return []
    df = blocks.copy()
    df = df[df["source"].isin(["abstract", "body"])].copy()
    if df.empty:
        return []

    df["block_id"] = df["block_id"].astype(str)
    df = df[~df["block_id"].isin(exclude_block_ids)].copy()
    if df.empty:
        return []

    df["text"] = df["text"].astype(str)
    df["_hit"] = df["text"].map(lambda t: bool(_DX_TRIGGER_RE.search(t)))
    df = df[df["_hit"]].copy()
    if df.empty:
        return []

    # Prefer abstract, then shorter, then earlier blocks.
    df["_source_pri"] = df["source"].map(lambda s: 0 if s == "abstract" else 1)
    df = df.sort_values(["_source_pri", "n_chars", "block_index"]).head(int(max_snippets))

    out = []
    for r in df.to_dict(orient="records"):
        out.append(
            {
                "block_id": str(r["block_id"]),
                "source": str(r.get("source") or ""),
                "sec_path_str": str(r.get("sec_path_str") or ""),
                "text": str(r.get("text") or ""),
            }
        )
    return out


def _cap_blocks_for_llm(
    blocks: list[dict[str, Any]],
    *,
    max_blocks: int,
    max_total_chars: int,
) -> list[dict[str, Any]]:
    """Keep stable order and cap the payload size."""
    if not blocks:
        return []
    max_blocks = max(int(max_blocks), 1)
    max_total_chars = max(int(max_total_chars), 1)

    kept: list[dict[str, Any]] = []
    total = 0
    for b in blocks:
        txt = str(b.get("text") or "")
        n = len(txt)
        if kept and total + n > max_total_chars:
            break
        kept.append(b)
        total += n
        if len(kept) >= max_blocks:
            break
    return kept


_LIST_FIELDS = [
    "ftld_inclusion_basis",
    "imaging_modalities",
    "imaging_regions",
    "genes_reported",
    "pathology_types",
    "symptom_tags",
    "evidence_block_ids",
]

_FTLD_BASIS_ALLOWED = {
    "author_reports_ftld_or_ppa_or_psp_or_cbs_or_ftd_mnd",
    "meets_consensus_criteria_explicitly_stated",
    "genetic_confirmed_pathogenic",
    "neuropath_confirmed",
    "specialist_dx_reported",
    "unclear_basis",
}

_IMAGING_MODALITIES_ALLOWED = {"mri", "ct", "fdg_pet", "spect", "other"}
_IMAGING_REGIONS_ALLOWED = {
    "frontal",
    "temporal",
    "parietal",
    "insula",
    "cingulate",
    "basal_ganglia",
    "brainstem",
    "cerebellum",
    "other",
    "unknown",
}
_PATHOLOGY_TYPES_ALLOWED = {"tau", "tdp43", "fus", "mixed", "other", "unknown"}
_INITIAL_DX_ALLOWED = {"ftld", "psychiatric", "ad", "vascular_stroke", "other_neuro", "other", "not_reported", "unknown"}


def _sanitize_label_obj(obj: dict[str, Any]) -> dict[str, Any]:
    """Coerce common model outputs into schema-compatible values (conservative)."""
    out = dict(obj)

    for k in _LIST_FIELDS:
        v = out.get(k)
        if v is None:
            out[k] = []
        elif isinstance(v, list):
            pass
        elif isinstance(v, str) and k in {"genes_reported"}:
            out[k] = [v]
        else:
            out[k] = []

    # imaging regions/modality: map unknowns to "other"/drop empties.
    mods = [str(x).strip().lower() for x in (out.get("imaging_modalities") or []) if str(x).strip()]
    out["imaging_modalities"] = [m for m in mods if m in _IMAGING_MODALITIES_ALLOWED]

    regs_raw = [str(x).strip().lower() for x in (out.get("imaging_regions") or []) if str(x).strip()]
    regs = []
    for r in regs_raw:
        if r in _IMAGING_REGIONS_ALLOWED:
            regs.append(r)
        else:
            regs.append("other")
    out["imaging_regions"] = list(dict.fromkeys(regs))

    # pathology types: normalize common spellings.
    pats_raw = [str(x).strip().lower() for x in (out.get("pathology_types") or []) if str(x).strip()]
    pats = []
    for p in pats_raw:
        p_norm = re.sub(r"[^a-z0-9]+", "", p)
        if "tdp" in p_norm:
            pats.append("tdp43")
        elif "tau" in p_norm:
            pats.append("tau")
        elif "fus" in p_norm:
            pats.append("fus")
        elif p in _PATHOLOGY_TYPES_ALLOWED:
            pats.append(p)
        else:
            pats.append("other")
    out["pathology_types"] = list(dict.fromkeys([p for p in pats if p in _PATHOLOGY_TYPES_ALLOWED]))

    # Symptom tags: ensure list of dicts and evidence lists.
    tags = out.get("symptom_tags") or []
    cleaned_tags: list[dict[str, Any]] = []
    if isinstance(tags, list):
        for t in tags:
            if not isinstance(t, dict):
                continue
            tag = t.get("tag")
            status = t.get("status")
            ev = t.get("evidence_block_ids")
            if ev is None or not isinstance(ev, list):
                ev = []
            cleaned_tags.append({"tag": tag, "status": status, "evidence_block_ids": ev})

    # Common model mistake: emit tag objects as top-level keys instead of in symptom_tags[].
    if not cleaned_tags:
        for tag_name in SymptomTagName.__args__:  # type: ignore[attr-defined]
            tv = out.get(tag_name)
            if not isinstance(tv, dict):
                continue
            status = tv.get("status")
            ev = tv.get("evidence_block_ids")
            if ev is None or not isinstance(ev, list):
                ev = []
            cleaned_tags.append({"tag": tag_name, "status": status, "evidence_block_ids": ev})
    out["symptom_tags"] = cleaned_tags

    # FTLD inclusion basis: filter invalid entries; enforce empty for exclude.
    tier = str(out.get("ftld_inclusion_tier") or "").strip()
    basis_raw = out.get("ftld_inclusion_basis") or []
    basis_clean: list[str] = []
    if isinstance(basis_raw, list):
        for b in basis_raw:
            b_str = str(b).strip()
            if b_str in _FTLD_BASIS_ALLOWED:
                basis_clean.append(b_str)
    basis_clean = list(dict.fromkeys(basis_clean))
    if tier == "exclude":
        out["ftld_inclusion_basis"] = []
    elif tier == "ftld_broad" and not basis_clean:
        out["ftld_inclusion_basis"] = ["unclear_basis"]
    else:
        out["ftld_inclusion_basis"] = basis_clean

    # Initial dx category: map common misplacements from non-FTLD category values.
    init = out.get("initial_dx_category")
    if init is None:
        pass
    else:
        init_str = str(init).strip()
        mapping = {
            "other_neurodegenerative_non_ftld": "other_neuro",
            "other_neurologic": "other_neuro",
            "vascular_stroke_or_poststroke_aphasia": "vascular_stroke",
            "primary_psychiatric_or_functional": "psychiatric",
        }
        mapped = mapping.get(init_str) or mapping.get(init_str.lower())
        out["initial_dx_category"] = init_str if init_str in _INITIAL_DX_ALLOWED else (mapped if mapped in _INITIAL_DX_ALLOWED else None)

    # Non-FTLD primary category: enforce allowed enum for excluded cases.
    non_ftld_allowed = {
        "vascular_stroke_or_poststroke_aphasia",
        "primary_psychiatric_or_functional",
        "brain_tumor_or_mass",
        "infection_inflammatory_or_autoimmune",
        "tbi_or_structural",
        "toxic_metabolic_or_medication",
        "epilepsy_seizure_related",
        "other_neurodegenerative_non_ftld",
        "other_neurologic",
        "unclear",
    }
    ncat = out.get("non_ftld_primary_category")
    if tier == "exclude":
        ncat_str = str(ncat).strip() if ncat is not None else ""
        if ncat_str not in non_ftld_allowed:
            out["non_ftld_primary_category"] = "unclear"
        else:
            out["non_ftld_primary_category"] = ncat_str
    else:
        # Ensure null for included cases.
        out["non_ftld_primary_category"] = None

    # Case-level evidence block ids
    ev = out.get("evidence_block_ids")
    if ev is None or not isinstance(ev, list):
        out["evidence_block_ids"] = []
    return out


def _ensure_all_symptom_tags(label: CaseLabelV1) -> CaseLabelV1:
    """Fill missing symptom tags deterministically with not_reported."""
    have = {t.tag for t in (label.symptom_tags or [])}
    want = set(SymptomTagName.__args__)  # type: ignore[attr-defined]
    missing = sorted(want - have)
    if not missing:
        return label
    tags = list(label.symptom_tags)
    for tag in missing:
        tags.append(SymptomTagV1(tag=tag, status="not_reported", evidence_block_ids=[]))
    return label.model_copy(update={"symptom_tags": tags})


def _label_one_case(
    *,
    client: OpenAIClient,
    prompt_path: Path,
    payload: dict[str, Any],
    model: str,
    endpoint: str,
    dry_run: bool,
) -> tuple[CaseLabelV1, dict[str, Any] | None]:
    if dry_run:
        # Deterministic minimal stub for plumbing tests only.
        stub = {
            "label_schema_version": "pmcoa_case_labels_v1",
            "pmcid": payload.get("pmcid"),
            "case_id": payload.get("case_id"),
            "ftld_inclusion_tier": "exclude",
            "ftld_inclusion_basis": [],
            "non_ftld_primary_category": "unclear",
            "non_ftld_specific_dx_free_text": None,
            "ftld_syndrome_reported": None,
            "ftld_syndrome_inferred": None,
            "symptom_duration_months": None,
            "age_at_onset_years": None,
            "age_at_presentation_years": None,
            "imaging_modalities": [],
            "imaging_laterality": None,
            "imaging_regions": [],
            "genetics_status": None,
            "genes_reported": [],
            "neuropath_status": None,
            "pathology_types": [],
            "misdiagnosed_prior_to_ftld": None,
            "initial_dx_category": None,
            "symptom_tags": [
                {"tag": t, "status": "not_reported", "evidence_block_ids": []} for t in SymptomTagName.__args__  # type: ignore[attr-defined]
            ],
            "label_confidence": "low",
            "needs_fulltext_review": True,
            "evidence_block_ids": [payload["blocks"][0]["block_id"]] if payload.get("blocks") else ["missing"],
            "notes": "dry_run stub",
        }
        return _ensure_all_symptom_tags(CaseLabelV1.model_validate(stub)), None

    sys_prompt = prompt_path.read_text(encoding="utf-8")
    user_prompt = json.dumps(payload, ensure_ascii=False)

    endpoint_literal = "chat_completions" if endpoint == "chat_completions" else "responses"

    def _call(max_output_tokens: int) -> tuple[dict[str, Any], dict[str, Any]]:
        return client.call_json(
            model=model,
            system_prompt=sys_prompt,
            user_prompt=user_prompt,
            max_output_tokens=int(max_output_tokens),
            endpoint=endpoint_literal,
        )

    try:
        obj, raw = _call(2000)
    except RuntimeError as exc:
        msg = str(exc)
        if "status not completed: incomplete" in msg and "max_output_tokens" in msg:
            obj, raw = _call(4000)
        else:
            raise

    if not isinstance(obj, dict):
        raise RuntimeError("Phase C model did not return a JSON object.")
    obj = _sanitize_label_obj(obj)
    try:
        label = CaseLabelV1.model_validate(obj)
    except ValidationError as exc:
        raise RuntimeError(f"Phase C label output failed schema validation: {exc}") from exc
    label = _ensure_all_symptom_tags(label)
    return label, raw


def _flatten_label_row(
    *,
    label: CaseLabelV1,
    meta: dict[str, Any],
    model: str,
    escalated: bool,
    status: str,
    error: str | None,
    n_blocks_provided: int,
    n_chars_provided: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "label_schema_version": label.label_schema_version,
        "pmcid": label.pmcid,
        "case_id": label.case_id,
        "year": meta.get("year"),
        "title": meta.get("title"),
        "pmid": meta.get("pmid"),
        "doi": meta.get("doi"),
        "split": meta.get("split"),
        "rater_type": "llm",
        "rater_id": "pass_c_v1",
        "model": model,
        "escalated": bool(escalated),
        "status": status,
        "error": error,
        "ftld_inclusion_tier": label.ftld_inclusion_tier,
        "ftld_inclusion_basis_json": _safe_json(label.ftld_inclusion_basis),
        "non_ftld_primary_category": label.non_ftld_primary_category,
        "non_ftld_specific_dx_free_text": label.non_ftld_specific_dx_free_text,
        "ftld_syndrome_reported": label.ftld_syndrome_reported,
        "ftld_syndrome_inferred": label.ftld_syndrome_inferred,
        "symptom_duration_months": label.symptom_duration_months,
        "age_at_onset_years": label.age_at_onset_years,
        "age_at_presentation_years": label.age_at_presentation_years,
        "imaging_modalities_json": _safe_json(label.imaging_modalities),
        "imaging_laterality": label.imaging_laterality,
        "imaging_regions_json": _safe_json(label.imaging_regions),
        "genetics_status": label.genetics_status,
        "genes_reported_json": _safe_json(label.genes_reported),
        "neuropath_status": label.neuropath_status,
        "pathology_types_json": _safe_json(label.pathology_types),
        "misdiagnosed_prior_to_ftld": label.misdiagnosed_prior_to_ftld,
        "initial_dx_category": label.initial_dx_category,
        "label_confidence": label.label_confidence,
        "needs_fulltext_review": bool(label.needs_fulltext_review),
        "case_evidence_block_ids_json": _safe_json(label.evidence_block_ids),
        "notes": label.notes,
        "n_blocks_provided": int(n_blocks_provided),
        "n_chars_provided": int(n_chars_provided),
    }

    tag_map = {t.tag: t for t in label.symptom_tags}
    for tag in SymptomTagName.__args__:  # type: ignore[attr-defined]
        t = tag_map.get(tag)
        row[f"tag__{tag}__status"] = t.status if t else "not_reported"
        row[f"tag__{tag}__evidence_block_ids_json"] = _safe_json(t.evidence_block_ids if t else [])
    return row


def _write_performance_report(df: pd.DataFrame, *, out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        out_md.write_text("# Phase C v1 Performance Report\n\nNo rows.\n", encoding="utf-8")
        return

    # When resuming/retrying, multiple rows per case can exist. Always report on the latest row per case.
    if "pmcid" in df.columns and "case_id" in df.columns:
        df_latest = df.drop_duplicates(subset=["pmcid", "case_id"], keep="last").copy()
    else:
        df_latest = df.copy()

    ok = df_latest[df_latest["status"] == "ok"].copy()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

    lines: list[str] = []
    lines.append("# Phase C v1 Performance Report")
    lines.append("")
    lines.append(f"Generated: {ts}")
    lines.append("")
    lines.append("## Run summary")
    lines.append(f"- Total cases attempted: {len(df_latest)}")
    lines.append(f"- Labeled ok: {len(ok)}")
    lines.append(f"- Errors: {int((df_latest['status'] == 'error').sum())}")
    lines.append(f"- Escalation rate: {float(ok['escalated'].mean()) if len(ok) else 0.0:.3f}")
    lines.append("")

    err_df = df_latest[df_latest["status"] == "error"].copy()
    if len(err_df) and "error" in err_df.columns:
        lines.append("## Errors (latest only)")
        err_df["error_type"] = err_df["error"].astype(str).str.split(":", n=1).str[0]
        vc = err_df["error_type"].value_counts()
        for k, v in vc.items():
            lines.append(f"- {k}: {int(v)}")
        lines.append("")

    if len(ok):
        lines.append("## Inclusion tiers (ok only)")
        tier_counts = ok["ftld_inclusion_tier"].value_counts(dropna=False)
        for k, v in tier_counts.items():
            lines.append(f"- {k}: {int(v)}")
        lines.append("")

        inc = ok[ok["ftld_inclusion_tier"].isin(["ftld_strict", "ftld_broad"])].copy()
        if len(inc):
            lines.append("## Syndromes (included only)")
            syn = inc["ftld_syndrome_reported"].value_counts(dropna=False)
            for k, v in syn.items():
                lines.append(f"- {k}: {int(v)}")
            lines.append("")

        lines.append("## Confidence (ok only)")
        conf = ok["label_confidence"].value_counts(dropna=False)
        for k, v in conf.items():
            lines.append(f"- {k}: {int(v)}")
        lines.append("")

        lines.append("## Symptom tags (ok only; exploratory)")
        tag_cols = [c for c in ok.columns if c.startswith("tag__") and c.endswith("__status")]
        for c in sorted(tag_cols):
            tag = c.split("__", 2)[1]
            vc = ok[c].value_counts(dropna=False)
            top = ", ".join([f"{k}={int(v)}" for k, v in vc.items()])
            lines.append(f"- {tag}: {top}")
        lines.append("")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class PhaseCOutputs:
    run_dir: Path
    labels_csv: Path
    run_config_json: Path
    report_md: Path


def label_phase_c_v1(
    *,
    segments_csv: Path,
    xml_dir: Path | None,
    out_dir: Path | None,
    max_cases: int | None,
    include_pmcids: list[str] | None,
    random_seed: int,
    dry_run: bool,
    overwrite: bool,
    resume: bool,
    retry_errors: bool,
    pass_c_model: str,
    pass_c_escalate_model: str,
    num_shards: int,
    shard_index: int,
    openai_endpoint: str,
    openai_timeout_s: int,
    openai_request_delay_s: float,
    max_blocks: int,
    max_total_chars: int,
    max_extra_snippets: int,
    prompt_c_path: Path | None,
    save_debug_bundles: bool,
) -> PhaseCOutputs:
    """Phase C v1: case-level FTLD phenotype + exploratory symptom tags."""
    segments_csv = Path(segments_csv).resolve()
    if not segments_csv.exists():
        raise FileNotFoundError(f"segments.csv not found: {segments_csv}")

    if overwrite and resume:
        raise ValueError("Invalid combination: --overwrite with --resume. Choose one.")

    project_root = _infer_project_root(segments_csv)
    prompt_c = (prompt_c_path or (project_root / "prompts" / "pmcoa" / "pass_c_label_v1.md")).resolve()
    if not prompt_c.exists():
        raise FileNotFoundError(f"Pass C prompt not found: {prompt_c}")
    prompt_text = prompt_c.read_text(encoding="utf-8")

    cutoff_year = None
    qpath = project_root / "queries" / "pmcoa_search.yaml"
    if qpath.exists():
        try:
            qfile = load_query_file(qpath)
            cutoff_year = qfile.split.cutoff_year
        except Exception:
            cutoff_year = None
    cutoff_year = int(cutoff_year or 2021)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if out_dir is None:
        if resume:
            raise ValueError("--resume requires an explicit --out-dir (stable run directory).")
        run_dir = (segments_csv.parent / "phase_c" / f"label__{ts}").resolve()
    else:
        run_dir = Path(out_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    labels_csv = run_dir / "case_labels_long.csv"
    run_config_json = run_dir / "run_config.json"
    report_md = run_dir / "performance_report.md"

    if labels_csv.exists() and not overwrite and not resume:
        raise FileExistsError(f"Output exists: {labels_csv} (use --resume or a new --out-dir).")

    client0 = OpenAIClient.from_env()
    client = OpenAIClient(
        api_key=client0.api_key,
        base_url=client0.base_url,
        timeout_s=int(openai_timeout_s),
        max_retries=client0.max_retries,
        request_delay_s=float(openai_request_delay_s),
        user_agent=client0.user_agent,
    )
    if not dry_run and not client.is_configured():
        raise RuntimeError("OPENAI_API_KEY is not set. Re-run with --dry-run or export OPENAI_API_KEY.")

    xml_dir_path = Path(xml_dir).resolve() if xml_dir is not None else None
    if xml_dir_path is not None and not xml_dir_path.exists():
        raise FileNotFoundError(f"xml_dir not found: {xml_dir_path}")

    # Load segments.
    usecols = [
        "pmcid",
        "case_id",
        "pmid",
        "doi",
        "year",
        "title",
        "block_ids_json",
        "segment_type",
        "include_for_embedding",
    ]
    df = pd.read_csv(segments_csv, usecols=usecols)
    df["pmcid"] = df["pmcid"].astype(str)
    df["case_id"] = df["case_id"].astype(str)

    # Group to case units.
    case_keys = df[["pmcid", "case_id"]].drop_duplicates().reset_index(drop=True)
    if case_keys.empty:
        raise ValueError("No case units found in segments.csv (pmcid+case_id).")

    include_pmcids = [str(p).strip() for p in (include_pmcids or []) if str(p).strip()]
    include_set = {p for p in include_pmcids if p.startswith("PMC")}
    forced = pd.DataFrame(columns=case_keys.columns)
    if include_set:
        forced = case_keys[case_keys["pmcid"].isin(include_set)].copy()
        forced = forced.sort_values(["pmcid", "case_id"]).reset_index(drop=True)
        case_keys = case_keys[~case_keys["pmcid"].isin(include_set)].reset_index(drop=True)

    num_shards = int(num_shards)
    shard_index = int(shard_index)
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1.")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must be in [0, num_shards-1].")

    def _in_shard(pmcid: str, case_id: str) -> bool:
        if num_shards == 1:
            return True
        h = hashlib.sha1(f"{pmcid}::{case_id}".encode("utf-8")).hexdigest()
        return (int(h[:8], 16) % num_shards) == shard_index

    # Deterministic sampling for test runs.
    if max_cases is not None:
        n_total = int(max_cases)
        n_forced = int(len(forced))
        n_remaining = max(n_total - n_forced, 0)
        n_remaining = min(n_remaining, int(len(case_keys)))
        sampled = (
            case_keys.sample(n=n_remaining, random_state=int(random_seed)).reset_index(drop=True)
            if n_remaining > 0
            else case_keys.head(0)
        )
        sampled = sampled[sampled.apply(lambda r: _in_shard(str(r["pmcid"]), str(r["case_id"])), axis=1)].reset_index(drop=True)
        case_keys = pd.concat([forced, sampled], ignore_index=True)
    else:
        case_keys = case_keys[case_keys.apply(lambda r: _in_shard(str(r["pmcid"]), str(r["case_id"])), axis=1)].reset_index(drop=True)
        if not forced.empty:
            case_keys = pd.concat([forced, case_keys], ignore_index=True)

    # Run config for audit.
    run_config = {
        "created_utc": ts,
        "segments_csv": str(segments_csv),
        "xml_dir": str(xml_dir_path) if xml_dir_path else None,
        "dry_run": bool(dry_run),
        "sampling": {
            "max_cases": int(max_cases) if max_cases is not None else None,
            "seed": int(random_seed),
            "include_pmcids": sorted(include_set) if include_set else [],
        },
        "sharding": {"num_shards": int(num_shards), "shard_index": int(shard_index)},
        "models": {"pass_c": pass_c_model, "pass_c_escalate": pass_c_escalate_model},
        "openai": {
            "base_url": client.base_url,
            "endpoint": openai_endpoint,
            "timeout_s": int(openai_timeout_s),
            "request_delay_s": float(openai_request_delay_s),
        },
        "limits": {"max_blocks": int(max_blocks), "max_total_chars": int(max_total_chars), "max_extra_snippets": int(max_extra_snippets)},
        "split_cutoff_year": int(cutoff_year),
        "prompt": {"pass_c_path": str(prompt_c), "pass_c_sha256": _sha256_text(prompt_text)},
    }
    run_config_json.write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    debug_dir = run_dir / "debug" if save_debug_bundles else None
    if debug_dir is not None:
        (debug_dir / "inputs").mkdir(parents=True, exist_ok=True)
        (debug_dir / "responses").mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "label_schema_version",
        "pmcid",
        "case_id",
        "year",
        "title",
        "pmid",
        "doi",
        "split",
        "rater_type",
        "rater_id",
        "model",
        "escalated",
        "status",
        "error",
        "ftld_inclusion_tier",
        "ftld_inclusion_basis_json",
        "non_ftld_primary_category",
        "non_ftld_specific_dx_free_text",
        "ftld_syndrome_reported",
        "ftld_syndrome_inferred",
        "symptom_duration_months",
        "age_at_onset_years",
        "age_at_presentation_years",
        "imaging_modalities_json",
        "imaging_laterality",
        "imaging_regions_json",
        "genetics_status",
        "genes_reported_json",
        "neuropath_status",
        "pathology_types_json",
        "misdiagnosed_prior_to_ftld",
        "initial_dx_category",
        "label_confidence",
        "needs_fulltext_review",
        "case_evidence_block_ids_json",
        "notes",
        "n_blocks_provided",
        "n_chars_provided",
    ]
    for tag in SymptomTagName.__args__:  # type: ignore[attr-defined]
        fieldnames.append(f"tag__{tag}__status")
        fieldnames.append(f"tag__{tag}__evidence_block_ids_json")

    # Resume support.
    existing = _read_last_statuses(labels_csv) if resume and labels_csv.exists() else {}
    done_statuses = {"ok"}
    if retry_errors:
        processed_prior = {k for k, st in existing.items() if st in done_statuses}
    else:
        processed_prior = set(existing.keys())

    mode = "a" if resume and labels_csv.exists() and labels_csv.stat().st_size > 0 else "w"
    with labels_csv.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()

        # Process grouped by pmcid for efficient XML parsing.
        for pmcid, g_keys in case_keys.groupby("pmcid", sort=False):
            pmcid = str(pmcid)
            pmcid_cases = [(pmcid, str(cid)) for cid in g_keys["case_id"].astype(str).tolist()]
            pmcid_cases = [k for k in pmcid_cases if k not in processed_prior]
            if not pmcid_cases:
                continue

            blocks_df = None
            if xml_dir_path is not None:
                xml_path = xml_dir_path / f"{pmcid}.xml"
                if xml_path.exists():
                    try:
                        parsed = parse_jats_xml(xml_path, pmcid=pmcid)
                        blocks_df = parsed.blocks
                    except Exception:
                        blocks_df = None

            # Per-case processing.
            for _, row_key in g_keys.iterrows():
                case_id = str(row_key["case_id"])
                key = (pmcid, case_id)
                if key in processed_prior:
                    continue

                # Collect segment rows for this case.
                seg = df[(df["pmcid"] == pmcid) & (df["case_id"] == case_id)].copy()
                meta = {
                    "pmid": (seg["pmid"].dropna().astype(str).iloc[0] if "pmid" in seg.columns and seg["pmid"].notna().any() else None),
                    "doi": (seg["doi"].dropna().astype(str).iloc[0] if "doi" in seg.columns and seg["doi"].notna().any() else None),
                    "year": (seg["year"].dropna().iloc[0] if "year" in seg.columns and seg["year"].notna().any() else None),
                    "title": (seg["title"].dropna().astype(str).iloc[0] if "title" in seg.columns and seg["title"].notna().any() else None),
                }
                meta["split"] = _split_from_year(meta.get("year"), cutoff_year) if cutoff_year else None

                block_ids: list[str] = []
                for s in seg["block_ids_json"].astype(str).tolist():
                    try:
                        ids = json.loads(s)
                        if isinstance(ids, list):
                            block_ids.extend([str(x) for x in ids])
                    except Exception:
                        continue
                block_ids = list(dict.fromkeys([b for b in block_ids if b]))
                block_id_set = set(block_ids)

                blocks_payload: list[dict[str, Any]] = []
                if blocks_df is not None and block_id_set:
                    bdf = blocks_df.copy()
                    bdf["block_id"] = bdf["block_id"].astype(str)
                    have = bdf[bdf["block_id"].isin(block_id_set)]
                    for r in have.sort_values("block_index").to_dict(orient="records"):
                        blocks_payload.append(
                            {
                                "block_id": str(r["block_id"]),
                                "source": str(r.get("source") or ""),
                                "sec_path_str": str(r.get("sec_path_str") or ""),
                                "text": str(r.get("text") or ""),
                            }
                        )

                # Extra snippets: diagnosis/criteria/genetics/pathology anchors outside extracted blocks.
                extra = []
                if blocks_df is not None:
                    extra = _select_extra_snippets(blocks_df, exclude_block_ids=set(b.get("block_id") for b in blocks_payload), max_snippets=int(max_extra_snippets))
                blocks_payload.extend(extra)

                blocks_payload = _cap_blocks_for_llm(
                    blocks_payload,
                    max_blocks=int(max_blocks),
                    max_total_chars=int(max_total_chars),
                )

                year_val = meta.get("year")
                try:
                    year_val = int(year_val) if year_val is not None else None
                except Exception:
                    year_val = None

                payload = {
                    "pmcid": pmcid,
                    "case_id": case_id,
                    "title": meta.get("title"),
                    "year": year_val,
                    "blocks": blocks_payload,
                }

                status = "ok"
                err = None
                model_used = str(pass_c_model)
                escalated = False
                raw = None
                try:
                    label, raw = _label_one_case(
                        client=client,
                        prompt_path=prompt_c,
                        payload=payload,
                        model=str(pass_c_model),
                        endpoint=str(openai_endpoint),
                        dry_run=bool(dry_run),
                    )

                    # Escalation rule: low confidence, needs review, or broad/unclear with missing basis.
                    should_escalate = (
                        (not bool(dry_run))
                        and (label.label_confidence == "low" or bool(label.needs_fulltext_review))
                    )
                    if should_escalate and pass_c_escalate_model and (str(pass_c_escalate_model) != str(pass_c_model)):
                        escalated = True
                        model_used = str(pass_c_escalate_model)
                        label, raw = _label_one_case(
                            client=client,
                            prompt_path=prompt_c,
                            payload=payload,
                            model=str(pass_c_escalate_model),
                            endpoint=str(openai_endpoint),
                            dry_run=False,
                        )

                    n_chars = int(sum(len(str(b.get("text") or "")) for b in blocks_payload))
                    out_row = _flatten_label_row(
                        label=label,
                        meta=meta,
                        model=model_used,
                        escalated=bool(escalated),
                        status="ok",
                        error=None,
                        n_blocks_provided=int(len(blocks_payload)),
                        n_chars_provided=int(n_chars),
                    )
                except Exception as exc:
                    status = "error"
                    err = f"{type(exc).__name__}: {exc}"
                    # Best-effort minimal row for traceability.
                    out_row = {
                        "label_schema_version": "pmcoa_case_labels_v1",
                        "pmcid": pmcid,
                        "case_id": case_id,
                        "year": meta.get("year"),
                        "title": meta.get("title"),
                        "pmid": meta.get("pmid"),
                        "doi": meta.get("doi"),
                        "split": meta.get("split"),
                        "rater_type": "llm",
                        "rater_id": "pass_c_v1",
                        "model": model_used,
                        "escalated": bool(escalated),
                        "status": status,
                        "error": err,
                        "ftld_inclusion_tier": None,
                        "ftld_inclusion_basis_json": "[]",
                        "non_ftld_primary_category": None,
                        "non_ftld_specific_dx_free_text": None,
                        "ftld_syndrome_reported": None,
                        "ftld_syndrome_inferred": None,
                        "symptom_duration_months": None,
                        "age_at_onset_years": None,
                        "age_at_presentation_years": None,
                        "imaging_modalities_json": "[]",
                        "imaging_laterality": None,
                        "imaging_regions_json": "[]",
                        "genetics_status": None,
                        "genes_reported_json": "[]",
                        "neuropath_status": None,
                        "pathology_types_json": "[]",
                        "misdiagnosed_prior_to_ftld": None,
                        "initial_dx_category": None,
                        "label_confidence": None,
                        "needs_fulltext_review": None,
                        "case_evidence_block_ids_json": "[]",
                        "notes": None,
                        "n_blocks_provided": int(len(blocks_payload)),
                        "n_chars_provided": int(sum(len(str(b.get("text") or "")) for b in blocks_payload)),
                    }
                    for tag in SymptomTagName.__args__:  # type: ignore[attr-defined]
                        out_row.setdefault(f"tag__{tag}__status", None)
                        out_row.setdefault(f"tag__{tag}__evidence_block_ids_json", "[]")

                writer.writerow(out_row)
                f.flush()

                if debug_dir is not None:
                    (debug_dir / "inputs" / f"{pmcid}__{case_id}.json").write_text(
                        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
                    )
                    if raw is not None:
                        (debug_dir / "responses" / f"{pmcid}__{case_id}.json").write_text(
                            json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8"
                        )

    # Write report for whatever is currently in the labels CSV.
    df_out = pd.read_csv(labels_csv)
    _write_performance_report(df_out, out_md=report_md)
    return PhaseCOutputs(run_dir=run_dir, labels_csv=labels_csv, run_config_json=run_config_json, report_md=report_md)
