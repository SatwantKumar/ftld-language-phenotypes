from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import csv
import numpy as np
import pandas as pd

from paperjn.llm.openai_client import OpenAIClient
from paperjn.pmcoa.rater_validation import SYMPTOM_TAGS_V1
from paperjn.utils.paths import ensure_dir


def _utc_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _as_int01(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value) if int(value) in (0, 1) else None
    if isinstance(value, float) and np.isnan(value):
        return None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    if s in {"0", "1"}:
        return int(s)
    return None


def _read_packet_text(*, share_dir: Path, rel_path: str, max_chars: int | None) -> str:
    p = (share_dir / rel_path).resolve()
    text = p.read_text(encoding="utf-8", errors="ignore")
    if max_chars is not None and int(max_chars) > 0 and len(text) > int(max_chars):
        return text[: int(max_chars)] + "\n\n[TRUNCATED]"
    return text


def _system_prompt(*, rater_label: str) -> str:
    return (
        "You are a board-certified Behavioral Neurologist performing blinded chart-review style ratings.\n"
        "Follow the rubric strictly, and only use information explicitly present in the provided text.\n"
        f"Rater label: {rater_label}\n"
        "\n"
        "Output must be a single JSON object (no extra text)."
    )


def _user_prompt(*, packet_text: str) -> str:
    tag_list = ", ".join(SYMPTOM_TAGS_V1)
    return (
        "Task: Fill a clinician rating sheet for one case packet.\n"
        "\n"
        "Rules:\n"
        "- Use ONLY the provided packet text. Do not assume anything not stated.\n"
        "- All labels are binary: 1=yes/present, 0=no/not stated.\n"
        "- First decide text_adequate:\n"
        "  - 1 if there is enough clinical description to rate FTLD/PPA and symptom tags.\n"
        "  - 0 if insufficient/garbled/too little clinical narrative.\n"
        "- If text_adequate=0, set all other fields to null.\n"
        "- is_ftld_spectrum: 1 only if an FTLD-spectrum neurodegenerative syndrome is explicitly described/diagnosed.\n"
        "- is_ppa: 1 only for neurodegenerative PPA or a clear progressive language-led syndrome; stroke/post-stroke aphasia => 0.\n"
        "- Symptom tags: set tag=1 only if explicitly present; otherwise 0.\n"
        "\n"
        f"Symptom tags (keys must match exactly): {tag_list}\n"
        "\n"
        "Return JSON with this schema:\n"
        "{\n"
        '  "text_adequate": 0|1,\n'
        '  "is_ftld_spectrum": 0|1|null,\n'
        '  "is_ppa": 0|1|null,\n'
        '  "tags": { "<tag>": 0|1|null, ... },\n'
        '  "notes": "optional short string"\n'
        "}\n"
        "\n"
        "Packet text:\n"
        "-----\n"
        f"{packet_text}\n"
        "-----\n"
    )


def _empty_template_row(row: pd.Series) -> dict[str, object]:
    out: dict[str, object] = {}
    for col in row.index:
        out[col] = row[col]
    # ensure label columns exist even if template changed
    for col in ["text_adequate", "is_ftld_spectrum", "is_ppa"]:
        out.setdefault(col, "")
    for tag in SYMPTOM_TAGS_V1:
        out.setdefault(f"tag__{tag}", "")
    out.setdefault("notes", "")
    return out


def _coerce_llm_result(obj: dict[str, Any]) -> tuple[dict[str, object], str | None]:
    err: str | None = None

    text_ok = _as_int01(obj.get("text_adequate"))
    if text_ok is None:
        err = "missing/invalid text_adequate"
        text_ok = 0

    if int(text_ok) == 0:
        tags_out = {f"tag__{t}": "" for t in SYMPTOM_TAGS_V1}
        return (
            {
                "text_adequate": 0,
                "is_ftld_spectrum": "",
                "is_ppa": "",
                **tags_out,
                "notes": str(obj.get("notes") or "").strip(),
            },
            err,
        )

    tags_obj = obj.get("tags")
    if not isinstance(tags_obj, dict):
        tags_obj = {}
        err = err or "missing/invalid tags"

    tags_out: dict[str, object] = {}
    for tag in SYMPTOM_TAGS_V1:
        v = _as_int01(tags_obj.get(tag))
        tags_out[f"tag__{tag}"] = "" if v is None else int(v)

    ftld = _as_int01(obj.get("is_ftld_spectrum"))
    ppa = _as_int01(obj.get("is_ppa"))

    out = {
        "text_adequate": int(text_ok),
        "is_ftld_spectrum": "" if ftld is None else int(ftld),
        "is_ppa": "" if ppa is None else int(ppa),
        **tags_out,
        "notes": str(obj.get("notes") or "").strip(),
    }
    return out, err


@dataclass(frozen=True)
class InsilicoRaterRunOutputs:
    out_dir: Path
    filled_csv: Path
    log_csv: Path
    run_config_json: Path


def run_insilico_rater(
    *,
    share_dir: Path,
    template_csv: Path,
    out_dir: Path | None,
    rater_label: str,
    model: str,
    temperature: float,
    endpoint: Literal["responses", "chat_completions"],
    request_delay_s: float,
    max_output_tokens: int,
    max_packet_chars: int | None,
    resume: bool,
    max_cases: int | None,
) -> InsilicoRaterRunOutputs:
    """Fill a clinician template using an LLM (virtual rater)."""
    share_dir = Path(share_dir).resolve()
    template_csv = Path(template_csv).resolve()
    if not template_csv.exists():
        raise FileNotFoundError(f"Template CSV not found: {template_csv}")
    if not share_dir.exists():
        raise FileNotFoundError(f"share_dir not found: {share_dir}")

    df = pd.read_csv(template_csv)
    required = ["case_uid", "pmcid", "case_id", "packet_path"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Template missing required columns: {missing}")

    if max_cases is not None:
        df = df.head(int(max_cases)).copy()

    ts = _utc_slug()
    if out_dir is None:
        out_dir = (share_dir.parent.parent / "phase_f_insilico" / f"insilico_rater__{ts}").resolve()
    out_dir = ensure_dir(Path(out_dir).resolve())

    model_slug = model.replace("/", "_")
    filled_csv = out_dir / f"rater__INSILICO__{rater_label}__{model_slug}.csv"
    log_csv = out_dir / f"log__INSILICO__{rater_label}.csv"
    run_config_json = out_dir / "run_config.json"

    run_config = {
        "created_utc": ts,
        "share_dir": str(share_dir),
        "template_csv": str(template_csv),
        "rater_label": str(rater_label),
        "model": str(model),
        "temperature": float(temperature),
        "endpoint": str(endpoint),
        "request_delay_s": float(request_delay_s),
        "max_output_tokens": int(max_output_tokens),
        "max_packet_chars": int(max_packet_chars) if max_packet_chars is not None else None,
        "resume": bool(resume),
        "max_cases": int(max_cases) if max_cases is not None else None,
        "schema": {
            "required_cols": required,
            "tags": SYMPTOM_TAGS_V1,
        },
    }
    run_config_json.write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    # Resume support: skip case_uids already present in the output CSV.
    done: set[str] = set()
    if resume and filled_csv.exists():
        prior = pd.read_csv(filled_csv)
        if "case_uid" in prior.columns:
            done = set(prior["case_uid"].astype(str).tolist())

    client0 = OpenAIClient.from_env()
    client = OpenAIClient(
        api_key=client0.api_key,
        base_url=client0.base_url,
        timeout_s=client0.timeout_s,
        max_retries=client0.max_retries,
        request_delay_s=float(request_delay_s),
        user_agent="paperjn/0.1 (in-silico clinician rater)",
    )
    if not client.is_configured():
        raise RuntimeError("OPENAI_API_KEY is not set in environment.")

    out_rows: list[dict[str, object]] = []
    log_rows: list[dict[str, object]] = []

    def _sanitize(v: object) -> object:
        if v is None:
            return ""
        if isinstance(v, float) and np.isnan(v):
            return ""
        return v

    fieldnames = list(df.columns)
    for col in ["text_adequate", "is_ftld_spectrum", "is_ppa"]:
        if col not in fieldnames:
            fieldnames.append(col)
    for tag in SYMPTOM_TAGS_V1:
        col = f"tag__{tag}"
        if col not in fieldnames:
            fieldnames.append(col)
    if "notes" not in fieldnames:
        fieldnames.append("notes")

    for _, row in df.iterrows():
        case_uid = str(row["case_uid"])
        if case_uid in done:
            continue

        packet_rel = str(row["packet_path"])
        packet_text = _read_packet_text(share_dir=share_dir, rel_path=packet_rel, max_chars=max_packet_chars)

        sys = _system_prompt(rater_label=rater_label)
        usr = _user_prompt(packet_text=packet_text)

        status = "ok"
        err = None
        parsed: dict[str, object] = {}
        try:
            obj, _raw = client.call_json(
                model=str(model),
                system_prompt=sys,
                user_prompt=usr,
                temperature=float(temperature),
                reasoning_effort="auto",
                max_output_tokens=int(max_output_tokens),
                endpoint=str(endpoint),  # type: ignore[arg-type]
            )
            parsed, err = _coerce_llm_result(obj)
        except Exception as exc:
            status = "error"
            err = f"{type(exc).__name__}: {exc}"
            parsed = {}

        out = _empty_template_row(row)
        # overlay labels (if any)
        for k, v in parsed.items():
            out[k] = v
        out_rows.append(out)

        log_rows.append(
            {
                "case_uid": case_uid,
                "pmcid": str(row.get("pmcid", "")),
                "case_id": str(row.get("case_id", "")),
                "status": status,
                "error": err or "",
            }
        )

        # Append incrementally to be crash-resilient.
        write_header = not filled_csv.exists()
        with filled_csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
            for r in out_rows:
                w.writerow({k: _sanitize(r.get(k, "")) for k in fieldnames})

        log_header = not log_csv.exists()
        with log_csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["case_uid", "pmcid", "case_id", "status", "error"])
            if log_header:
                w.writeheader()
            for r in log_rows:
                w.writerow({k: _sanitize(r.get(k, "")) for k in ["case_uid", "pmcid", "case_id", "status", "error"]})

        out_rows.clear()
        log_rows.clear()

    # Ensure files exist even if no new rows were written.
    if not filled_csv.exists():
        df.to_csv(filled_csv, index=False)
    if not log_csv.exists():
        pd.DataFrame([], columns=["case_uid", "pmcid", "case_id", "status", "error"]).to_csv(log_csv, index=False)

    return InsilicoRaterRunOutputs(
        out_dir=out_dir,
        filled_csv=filled_csv,
        log_csv=log_csv,
        run_config_json=run_config_json,
    )
