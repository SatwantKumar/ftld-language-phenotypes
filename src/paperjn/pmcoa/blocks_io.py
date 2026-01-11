from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from paperjn.pmcoa.parse_jats_blocks import ParsedJATS, parse_jats_xml


@dataclass(frozen=True)
class JATSParseResult:
    pmcid: str
    status: str  # ok|skipped|error
    xml_path: Path
    out_blocks_jsonl: Path | None
    out_meta_json: Path | None
    n_blocks_total: int | None
    n_blocks_body: int | None
    n_blocks_abstract: int | None
    article_type: str | None
    has_body: bool | None
    error: str | None


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_blocks_jsonl(path: Path, blocks: pd.DataFrame) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in blocks.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return int(n)


def parse_jats_file(
    *,
    xml_path: Path,
    out_dir: Path,
    overwrite: bool = False,
) -> JATSParseResult:
    xml_path = Path(xml_path)
    pmcid = xml_path.stem
    out_blocks = out_dir / f"{pmcid}.blocks.jsonl"
    out_meta = out_dir / f"{pmcid}.meta.json"

    if out_blocks.exists() and out_meta.exists() and not overwrite:
        try:
            meta = json.loads(out_meta.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        return JATSParseResult(
            pmcid=pmcid,
            status="skipped",
            xml_path=xml_path,
            out_blocks_jsonl=out_blocks,
            out_meta_json=out_meta,
            n_blocks_total=int(meta.get("n_blocks_total")) if meta.get("n_blocks_total") is not None else None,
            n_blocks_body=int(meta.get("n_blocks_body")) if meta.get("n_blocks_body") is not None else None,
            n_blocks_abstract=int(meta.get("n_blocks_abstract"))
            if meta.get("n_blocks_abstract") is not None
            else None,
            article_type=str(meta.get("article_type")) if meta.get("article_type") is not None else None,
            has_body=bool(meta.get("has_body")) if meta.get("has_body") is not None else None,
            error=None,
        )

    try:
        parsed: ParsedJATS = parse_jats_xml(xml_path, pmcid=pmcid)
        _write_json(out_meta, dict(parsed.metadata))
        _write_blocks_jsonl(out_blocks, parsed.blocks)
        meta = parsed.metadata
        return JATSParseResult(
            pmcid=pmcid,
            status="ok",
            xml_path=xml_path,
            out_blocks_jsonl=out_blocks,
            out_meta_json=out_meta,
            n_blocks_total=int(meta.get("n_blocks_total")) if meta.get("n_blocks_total") is not None else None,
            n_blocks_body=int(meta.get("n_blocks_body")) if meta.get("n_blocks_body") is not None else None,
            n_blocks_abstract=int(meta.get("n_blocks_abstract"))
            if meta.get("n_blocks_abstract") is not None
            else None,
            article_type=str(meta.get("article_type")) if meta.get("article_type") is not None else None,
            has_body=bool(meta.get("has_body")) if meta.get("has_body") is not None else None,
            error=None,
        )
    except Exception as exc:
        return JATSParseResult(
            pmcid=pmcid,
            status="error",
            xml_path=xml_path,
            out_blocks_jsonl=None,
            out_meta_json=None,
            n_blocks_total=None,
            n_blocks_body=None,
            n_blocks_abstract=None,
            article_type=None,
            has_body=None,
            error=str(exc),
        )


def parse_jats_dir(
    *,
    xml_dir: Path,
    out_dir: Path,
    max_papers: int | None,
    random_seed: int,
    overwrite: bool,
) -> tuple[pd.DataFrame, Path]:
    xml_dir = Path(xml_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_xml = sorted(xml_dir.glob("PMC*.xml"))
    if not all_xml:
        raise ValueError(f"No PMC XML files found in: {xml_dir}")

    if max_papers is not None:
        n = min(int(max_papers), len(all_xml))
        all_xml = (
            pd.Series([str(p) for p in all_xml])
            .sample(n=n, random_state=int(random_seed))
            .map(Path)
            .tolist()
        )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = out_dir / f"jats_parse_log__{ts}.csv"

    fieldnames = [
        "pmcid",
        "status",
        "xml_path",
        "out_blocks_jsonl",
        "out_meta_json",
        "n_blocks_total",
        "n_blocks_body",
        "n_blocks_abstract",
        "article_type",
        "has_body",
        "error",
    ]

    rows: list[dict[str, Any]] = []
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for xml_path in all_xml:
            res = parse_jats_file(xml_path=xml_path, out_dir=out_dir, overwrite=overwrite)
            rec = {
                "pmcid": res.pmcid,
                "status": res.status,
                "xml_path": str(res.xml_path),
                "out_blocks_jsonl": str(res.out_blocks_jsonl) if res.out_blocks_jsonl else None,
                "out_meta_json": str(res.out_meta_json) if res.out_meta_json else None,
                "n_blocks_total": res.n_blocks_total,
                "n_blocks_body": res.n_blocks_body,
                "n_blocks_abstract": res.n_blocks_abstract,
                "article_type": res.article_type,
                "has_body": res.has_body,
                "error": res.error,
            }
            writer.writerow(rec)
            f.flush()
            rows.append(rec)

    return pd.DataFrame(rows), log_path

