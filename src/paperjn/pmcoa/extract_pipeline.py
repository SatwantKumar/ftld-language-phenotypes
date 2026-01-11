from __future__ import annotations

import csv
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from paperjn.config import ProjectConfig
from paperjn.llm.openai_client import OpenAIClient
from paperjn.nlp.leakage import audit_text_for_leakage, remove_blacklisted_terms
from paperjn.pmcoa.llm_extraction import run_pass_a_route, run_pass_b_extract
from paperjn.pmcoa.parse_jats_blocks import parse_jats_xml
from paperjn.pmcoa.routing import route_candidate_sections


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _infer_project_root(path: Path) -> Path:
    p = Path(path).resolve()
    for _ in range(8):
        if (p / "src").exists() and (p / "docs").exists():
            return p
        p = p.parent
    return Path(path).resolve()


def _safe_json(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False)


def _segment_uid(pmcid: str, case_id: str, segment_type: str, block_ids: list[str]) -> str:
    key = "\n".join([pmcid, case_id, segment_type, "|".join(block_ids)])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class PMCOAExtractionOutputs:
    run_dir: Path
    paper_log_csv: Path
    segments_csv: Path
    run_config_json: Path


def _read_last_statuses(paper_log_csv: Path) -> dict[str, str]:
    if not paper_log_csv.exists():
        return {}
    statuses: dict[str, str] = {}
    with paper_log_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pmcid = (row.get("pmcid") or "").strip()
            if not pmcid:
                continue
            statuses[pmcid] = (row.get("status") or "").strip()
    return statuses


def _read_csv_header(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        return next(reader, None)


def _assert_csv_header(path: Path, expected: list[str], *, label: str) -> None:
    header = _read_csv_header(path)
    if header is None:
        return
    if header != expected:
        raise RuntimeError(
            f"{label} header mismatch for resume: {path}. "
            "Start a new out_dir (or overwrite) to change output schema."
        )


def _safe_load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _ensure_locked_run_config(
    *,
    run_config_json: Path,
    current: dict[str, Any],
    resume: bool,
    overwrite: bool,
) -> None:
    if overwrite:
        run_config_json.write_text(json.dumps(current, indent=2), encoding="utf-8")
        return

    if resume and run_config_json.exists():
        prev = _safe_load_json(run_config_json)
        prev_prompts = prev.get("prompts") or {}
        cur_prompts = current.get("prompts") or {}
        # Lock on prompt hashes + model names + endpoint; these define the extraction procedure.
        checks = [
            ("prompts.pass_a_sha256", prev_prompts.get("pass_a_sha256"), cur_prompts.get("pass_a_sha256")),
            ("prompts.pass_b_sha256", prev_prompts.get("pass_b_sha256"), cur_prompts.get("pass_b_sha256")),
            ("models.pass_a", (prev.get("models") or {}).get("pass_a"), (current.get("models") or {}).get("pass_a")),
            ("models.pass_b", (prev.get("models") or {}).get("pass_b"), (current.get("models") or {}).get("pass_b")),
            (
                "openai.endpoint",
                (prev.get("openai") or {}).get("endpoint"),
                (current.get("openai") or {}).get("endpoint"),
            ),
        ]
        mismatches = [name for name, a, b in checks if a is not None and b is not None and a != b]
        if mismatches:
            raise RuntimeError(
                "Run config mismatch for resume (locked spec violated). "
                f"Mismatched: {', '.join(mismatches)}. "
                "Start a new out_dir to change prompts/models/endpoint."
            )
        return

    # First run (or non-resume): write config.
    run_config_json.write_text(json.dumps(current, indent=2), encoding="utf-8")


def _process_one_xml(
    *,
    xml_path: Path,
    reg_map: dict[str, dict[str, Any]],
    client: OpenAIClient,
    prompt_a: Path,
    prompt_b: Path,
    pass_a_model: str,
    pass_b_model: str,
    openai_endpoint: str,
    dry_run: bool,
    max_blocks: int,
    max_total_chars: int,
    blacklist: list[str],
    paper_writer: csv.DictWriter,
    seg_writer: csv.DictWriter,
    f_paper,
    f_seg,
    seen_segment_uids: set[str] | None,
    debug_dir: Path | None,
) -> str:
    xml_path = Path(xml_path)
    pmcid = xml_path.stem
    meta_row = reg_map.get(pmcid, {})
    pmid = meta_row.get("pmid")
    doi = meta_row.get("doi")
    year = meta_row.get("year")

    status = "ok"
    err: str | None = None
    pass_a_out = None
    pass_b_out = None
    payload_a: dict[str, Any] | None = None
    payload_b: dict[str, Any] | None = None
    raw_a: dict[str, Any] | None = None
    raw_b: dict[str, Any] | None = None

    try:
        parsed = parse_jats_xml(xml_path, pmcid=pmcid)
        title = str(meta_row.get("title") or parsed.metadata.get("title") or "")
        article_type = parsed.metadata.get("article_type")

        pass_a_out, raw_a, payload_a = run_pass_a_route(
            client=client,
            prompt_path=prompt_a,
            title=title,
            article_type=article_type,
            blocks=parsed.blocks,
            dry_run=bool(dry_run),
            model=pass_a_model,
            endpoint=openai_endpoint,
        )

        routed = route_candidate_sections(parsed.blocks)
        candidate_sec_paths = list(dict.fromkeys(routed.must_include + routed.candidates))
        allowed = set(candidate_sec_paths)
        pass_a_selected = [s for s in pass_a_out.selected_sec_paths if s in allowed]
        if pass_a_out.is_case_like and (pass_a_selected or routed.must_include):
            final_sec_paths = list(dict.fromkeys(routed.must_include + pass_a_selected))
            if not final_sec_paths:
                final_sec_paths = candidate_sec_paths
            pass_b_out, raw_b, payload_b = run_pass_b_extract(
                client=client,
                prompt_path=prompt_b,
                title=title,
                blocks=parsed.blocks,
                selected_sec_paths=final_sec_paths,
                dry_run=bool(dry_run),
                model=pass_b_model,
                endpoint=openai_endpoint,
                max_blocks=int(max_blocks),
                max_total_chars=int(max_total_chars),
            )
        else:
            status = "not_case_like"
            final_sec_paths = []

        # Save debug bundles (input/output) per paper if requested.
        if debug_dir is not None:
            if payload_a is not None:
                (debug_dir / "pass_a" / f"{pmcid}__input.json").write_text(
                    json.dumps(payload_a, indent=2, ensure_ascii=False), encoding="utf-8"
                )
            if raw_a is not None:
                (debug_dir / "pass_a" / f"{pmcid}__response.json").write_text(
                    json.dumps(raw_a, indent=2, ensure_ascii=False), encoding="utf-8"
                )
            if payload_b is not None:
                (debug_dir / "pass_b" / f"{pmcid}__input.json").write_text(
                    json.dumps(payload_b, indent=2, ensure_ascii=False), encoding="utf-8"
                )
            if raw_b is not None:
                (debug_dir / "pass_b" / f"{pmcid}__response.json").write_text(
                    json.dumps(raw_b, indent=2, ensure_ascii=False), encoding="utf-8"
                )

        # Write segment rows.
        n_segments = 0
        if pass_b_out is not None:
            # Deterministic de-duplication: each block_id can appear in at most one segment.
            seg_type_pri = {
                "case_presentation": 0,
                "disease_course": 1,
                "clinical_features_summary": 2,
                "neuropsych_language_testing": 3,
                "imaging_biomarkers": 4,
                "pathology_genetics": 5,
                "treatment_response": 6,
                "other": 7,
            }

            def _case_pri(case_id: str) -> tuple[int, str]:
                if case_id == "case_unknown":
                    return (10_000, case_id)
                if case_id.startswith("case_"):
                    try:
                        return (int(case_id.split("_", 1)[1]), case_id)
                    except Exception:
                        return (9999, case_id)
                return (9999, case_id)

            id_to_row = {
                str(r["block_id"]): r
                for r in parsed.blocks[["block_id", "block_index", "sec_path_str", "text"]].to_dict(
                    orient="records"
                )
            }

            used_blocks: set[str] = set()
            deduped_segments = []
            for seg in sorted(
                pass_b_out.segments,
                key=lambda s: (
                    0 if s.include_for_embedding else 1,
                    seg_type_pri.get(s.segment_type, 99),
                    _case_pri(s.case_id),
                ),
            ):
                kept = [b for b in seg.block_ids if b not in used_blocks]
                if not kept:
                    continue
                used_blocks.update(kept)
                if kept != seg.block_ids:
                    seg = seg.model_copy(update={"block_ids": kept})
                deduped_segments.append(seg)
            pass_b_out = pass_b_out.model_copy(update={"segments": deduped_segments})

            for seg in pass_b_out.segments:
                # Stable ordering by original block_index.
                seg_rows = [id_to_row[b] for b in seg.block_ids]
                seg_rows = sorted(seg_rows, key=lambda x: int(x.get("block_index") or 0))
                text_raw = "\n".join(str(r.get("text") or "").strip() for r in seg_rows).strip()
                sec_paths = list(dict.fromkeys([str(r.get("sec_path_str") or "") for r in seg_rows if r]))
                sec_paths = [s for s in sec_paths if s]

                text_clean = remove_blacklisted_terms(text_raw, blacklist) if blacklist else text_raw
                aud_raw = audit_text_for_leakage(text_raw, blacklist) if blacklist else None
                aud_clean = audit_text_for_leakage(text_clean, blacklist) if blacklist else None

                seg_uid = _segment_uid(pmcid, seg.case_id, seg.segment_type, seg.block_ids)
                already_seen = seen_segment_uids is not None and seg_uid in seen_segment_uids
                if seen_segment_uids is not None:
                    seen_segment_uids.add(seg_uid)

                seg_row = {
                    "segment_uid": seg_uid,
                    "pmcid": pmcid,
                    "pmid": pmid,
                    "doi": doi,
                    "year": year,
                    "title": title,
                    "article_type": article_type,
                    "case_id": seg.case_id,
                    "segment_type": seg.segment_type,
                    "include_for_embedding": bool(seg.include_for_embedding),
                    "block_ids_json": _safe_json(seg.block_ids),
                    "sec_paths_json": _safe_json(sec_paths),
                    "n_blocks": int(len(seg.block_ids)),
                    "n_chars_raw": int(len(text_raw)),
                    "n_chars_clean": int(len(text_clean)),
                    "leakage_n_matches_raw": int(aud_raw.n_matches) if aud_raw else 0,
                    "leakage_n_matches_clean": int(aud_clean.n_matches) if aud_clean else 0,
                    "text_raw": text_raw,
                    "text_clean": text_clean,
                }
                if not already_seen:
                    seg_writer.writerow(seg_row)
                    f_seg.flush()
                n_segments += 1

        # Summarize paper.
        paper_row = {
            "pmcid": pmcid,
            "pmid": pmid,
            "doi": doi,
            "year": year,
            "title": title,
            "article_type": article_type,
            "status": status,
            "error": err,
            "n_blocks_total": parsed.metadata.get("n_blocks_total") if "parsed" in locals() else None,
            "n_blocks_body": parsed.metadata.get("n_blocks_body") if "parsed" in locals() else None,
            "n_candidate_sections": len(payload_a.get("candidate_sections", [])) if payload_a else 0,
            "pass_a_is_case_like": bool(pass_a_out.is_case_like) if pass_a_out else None,
            "pass_a_n_cases_est": pass_a_out.n_cases_est if pass_a_out else None,
            "pass_a_selected_sec_paths_json": _safe_json(pass_a_out.selected_sec_paths) if pass_a_out else "[]",
            "final_selected_sec_paths_json": _safe_json(final_sec_paths) if "final_sec_paths" in locals() else "[]",
            "n_blocks_selected_for_pass_b": int(payload_b.get("n_blocks"))
            if payload_b and payload_b.get("n_blocks") is not None
            else 0,
            "pass_b_n_cases_est": pass_b_out.n_cases_est if pass_b_out else None,
            "n_segments": int(n_segments),
            "dry_run": bool(dry_run),
        }
        paper_writer.writerow(paper_row)
        f_paper.flush()
        return status

    except Exception as exc:
        status = "error"
        err = str(exc)
        paper_row = {
            "pmcid": pmcid,
            "pmid": pmid,
            "doi": doi,
            "year": year,
            "title": meta_row.get("title"),
            "article_type": None,
            "status": status,
            "error": err,
            "n_blocks_total": None,
            "n_blocks_body": None,
            "n_candidate_sections": 0,
            "pass_a_is_case_like": None,
            "pass_a_n_cases_est": None,
            "pass_a_selected_sec_paths_json": "[]",
            "final_selected_sec_paths_json": "[]",
            "n_blocks_selected_for_pass_b": 0,
            "pass_b_n_cases_est": None,
            "n_segments": 0,
            "dry_run": bool(dry_run),
        }
        paper_writer.writerow(paper_row)
        f_paper.flush()
        return status


def extract_from_xml_dir(
    *,
    xml_dir: Path,
    out_dir: Path | None,
    registry_csv: Path | None,
    config: ProjectConfig | None,
    max_papers: int | None,
    random_seed: int,
    dry_run: bool,
    overwrite: bool,
    resume: bool,
    retry_errors: bool,
    watch: bool,
    watch_sleep_s: float,
    stop_after_idle_s: float | None,
    sample_random: bool,
    save_debug_bundles: bool,
    pass_a_model: str,
    pass_b_model: str,
    openai_endpoint: str,
    openai_request_delay_s: float,
    max_blocks: int,
    max_total_chars: int,
    prompt_a_path: Path | None,
    prompt_b_path: Path | None,
) -> PMCOAExtractionOutputs:
    xml_dir = Path(xml_dir).resolve()
    if not xml_dir.exists():
        raise FileNotFoundError(f"xml_dir not found: {xml_dir}")

    if overwrite and resume:
        raise ValueError("Invalid combination: --overwrite with --resume. Choose one.")

    project_root = _infer_project_root(xml_dir)
    prompt_a = (prompt_a_path or (project_root / "prompts" / "pmcoa" / "pass_a_route_v2.md")).resolve()
    prompt_b = (prompt_b_path or (project_root / "prompts" / "pmcoa" / "pass_b_extract_v2.md")).resolve()

    if not prompt_a.exists():
        raise FileNotFoundError(f"Pass A prompt not found: {prompt_a}")
    if not prompt_b.exists():
        raise FileNotFoundError(f"Pass B prompt not found: {prompt_b}")

    prompt_a_text = prompt_a.read_text(encoding="utf-8")
    prompt_b_text = prompt_b.read_text(encoding="utf-8")

    has_any_xml = any(xml_dir.glob("PMC*.xml"))
    if not has_any_xml and not watch:
        raise ValueError(f"No PMC XML files found in: {xml_dir}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if out_dir is None:
        if resume:
            raise ValueError("--resume requires an explicit --out-dir (stable run directory).")
        out_base = xml_dir.parent / "extractions"
        run_dir = (out_base / f"extract__{ts}").resolve()
    else:
        run_dir = Path(out_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    paper_log_csv = run_dir / "paper_log.csv"
    segments_csv = run_dir / "segments.csv"
    run_config_json = run_dir / "run_config.json"

    if (paper_log_csv.exists() or segments_csv.exists()) and not overwrite and not resume:
        raise FileExistsError(
            f"Output files already exist in {run_dir}. Use --overwrite or choose a new --out-dir."
        )

    client0 = OpenAIClient.from_env()
    client = OpenAIClient(
        api_key=client0.api_key,
        base_url=client0.base_url,
        timeout_s=client0.timeout_s,
        max_retries=client0.max_retries,
        request_delay_s=float(openai_request_delay_s),
        user_agent=client0.user_agent,
    )

    if not dry_run and not client.is_configured():
        raise RuntimeError("OPENAI_API_KEY is not set. Re-run with --dry-run or export OPENAI_API_KEY.")

    reg_map: dict[str, dict[str, Any]] = {}
    if registry_csv is not None:
        reg = pd.read_csv(registry_csv)
        if "pmcid" in reg.columns:
            reg = reg.drop_duplicates(subset=["pmcid"], keep="first")
            reg_map = {str(r["pmcid"]): r for r in reg.to_dict(orient="records")}

    # Run config for audit.
    run_config = {
        "created_utc": ts,
        "xml_dir": str(xml_dir),
        "registry_csv": str(registry_csv) if registry_csv else None,
        "dry_run": bool(dry_run),
        "models": {"pass_a": pass_a_model, "pass_b": pass_b_model},
        "openai": {
            "base_url": client.base_url,
            "endpoint": openai_endpoint,
            "request_delay_s": float(openai_request_delay_s),
        },
        "limits": {"max_blocks": int(max_blocks), "max_total_chars": int(max_total_chars)},
        "prompts": {
            "pass_a_path": str(prompt_a),
            "pass_b_path": str(prompt_b),
            "pass_a_sha256": _sha256_text(prompt_a_text),
            "pass_b_sha256": _sha256_text(prompt_b_text),
        },
        "leakage_blacklist_size": int(len(config.leakage_blacklist)) if config else 0,
    }
    _ensure_locked_run_config(
        run_config_json=run_config_json,
        current=run_config,
        resume=bool(resume),
        overwrite=bool(overwrite),
    )

    debug_dir = run_dir / "debug" if save_debug_bundles else None
    if debug_dir is not None:
        (debug_dir / "pass_a").mkdir(parents=True, exist_ok=True)
        (debug_dir / "pass_b").mkdir(parents=True, exist_ok=True)

    paper_fields = [
        "pmcid",
        "pmid",
        "doi",
        "year",
        "title",
        "article_type",
        "status",
        "error",
        "n_blocks_total",
        "n_blocks_body",
        "n_candidate_sections",
        "pass_a_is_case_like",
        "pass_a_n_cases_est",
        "pass_a_selected_sec_paths_json",
        "final_selected_sec_paths_json",
        "n_blocks_selected_for_pass_b",
        "pass_b_n_cases_est",
        "n_segments",
        "dry_run",
    ]
    seg_fields = [
        "segment_uid",
        "pmcid",
        "pmid",
        "doi",
        "year",
        "title",
        "article_type",
        "case_id",
        "segment_type",
        "include_for_embedding",
        "block_ids_json",
        "sec_paths_json",
        "n_blocks",
        "n_chars_raw",
        "n_chars_clean",
        "leakage_n_matches_raw",
        "leakage_n_matches_clean",
        "text_raw",
        "text_clean",
    ]

    blacklist = config.leakage_blacklist if config else []

    if resume:
        if paper_log_csv.exists() and paper_log_csv.stat().st_size > 0:
            _assert_csv_header(paper_log_csv, paper_fields, label="paper_log.csv")
        if segments_csv.exists() and segments_csv.stat().st_size > 0:
            _assert_csv_header(segments_csv, seg_fields, label="segments.csv")

    # Resume support: append to existing CSVs and skip already processed PMCIDs.
    existing_status = _read_last_statuses(paper_log_csv) if resume and paper_log_csv.exists() else {}
    done_statuses = {"ok", "not_case_like"}
    if retry_errors:
        processed_prior = {p for p, st in existing_status.items() if st in done_statuses}
    else:
        processed_prior = set(existing_status.keys())

    # Additional sanity for resume: ensure we won't double-write segments after partial crashes.
    seen_segment_uids: set[str] | None = None
    if resume and segments_csv.exists() and segments_csv.stat().st_size > 0:
        with segments_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "segment_uid" not in reader.fieldnames:
                raise RuntimeError(f"segments.csv missing required header 'segment_uid': {segments_csv}")
            seen_segment_uids = set()
            for row in reader:
                uid = (row.get("segment_uid") or "").strip()
                if uid:
                    seen_segment_uids.add(uid)

    processed_this_run: set[str] = set()

    mode_paper = "a" if resume and paper_log_csv.exists() and paper_log_csv.stat().st_size > 0 else "w"
    mode_seg = "a" if resume and segments_csv.exists() and segments_csv.stat().st_size > 0 else "w"

    with paper_log_csv.open(mode_paper, newline="", encoding="utf-8") as f_paper, segments_csv.open(
        mode_seg, newline="", encoding="utf-8"
    ) as f_seg:
        paper_writer = csv.DictWriter(f_paper, fieldnames=paper_fields)
        seg_writer = csv.DictWriter(f_seg, fieldnames=seg_fields)
        if mode_paper == "w":
            paper_writer.writeheader()
        if mode_seg == "w":
            seg_writer.writeheader()

        target_pmcids: set[str] | None = None
        if registry_csv is not None:
            reg = pd.read_csv(registry_csv, usecols=["pmcid"])
            target_pmcids = {str(x) for x in reg["pmcid"].astype(str).tolist() if str(x).startswith("PMC")}

        n_processed_total = 0
        last_activity = time.time()

        while True:
            # Discover new XML files each loop (supports --watch).
            xml_paths = sorted(xml_dir.glob("PMC*.xml"))
            candidates = []
            for p in xml_paths:
                pmcid = p.stem
                if pmcid in processed_prior or pmcid in processed_this_run:
                    continue
                candidates.append(p)

            # Optionally cap total papers for this invocation (batch runs).
            if max_papers is not None:
                remaining = int(max_papers) - int(n_processed_total)
                if remaining <= 0:
                    break
                candidates = candidates[:remaining]

            if not candidates:
                if not watch:
                    break
                if target_pmcids is not None:
                    # Stop if we've attempted all target PMCIDs.
                    attempted = set(existing_status.keys()) | processed_this_run
                    if len(attempted & target_pmcids) >= len(target_pmcids):
                        break
                if stop_after_idle_s is not None and (time.time() - last_activity) >= float(stop_after_idle_s):
                    break
                time.sleep(max(float(watch_sleep_s), 1.0))
                continue

            # Optional random sampling among candidates (useful for debugging).
            if sample_random and max_papers is not None and len(candidates) > 1:
                n = len(candidates)
                candidates = (
                    pd.Series([str(p) for p in candidates])
                    .sample(n=n, random_state=int(random_seed))
                    .map(Path)
                    .tolist()
                )

            for xml_path in candidates:
                xml_path = Path(xml_path)
                pmcid = xml_path.stem
                processed_this_run.add(pmcid)

                status = _process_one_xml(
                    xml_path=xml_path,
                    reg_map=reg_map,
                    client=client,
                    prompt_a=prompt_a,
                    prompt_b=prompt_b,
                    pass_a_model=str(pass_a_model),
                    pass_b_model=str(pass_b_model),
                    openai_endpoint=str(openai_endpoint),
                    dry_run=bool(dry_run),
                    max_blocks=int(max_blocks),
                    max_total_chars=int(max_total_chars),
                    blacklist=blacklist,
                    paper_writer=paper_writer,
                    seg_writer=seg_writer,
                    f_paper=f_paper,
                    f_seg=f_seg,
                    seen_segment_uids=seen_segment_uids,
                    debug_dir=debug_dir,
                )
                existing_status[pmcid] = status
                n_processed_total += 1
                last_activity = time.time()

                if max_papers is not None and n_processed_total >= int(max_papers):
                    break

        # End of watch loop

    return PMCOAExtractionOutputs(
        run_dir=run_dir, paper_log_csv=paper_log_csv, segments_csv=segments_csv, run_config_json=run_config_json
    )
