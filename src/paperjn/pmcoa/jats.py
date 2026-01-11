from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import time
from typing import Any, Literal

import pandas as pd

from paperjn.pmcoa.ncbi_eutils import EUtilsClient, _require_requests


_ARTICLE_TYPE_RE = re.compile(r'\barticle-type="([^"]+)"')
_API_KEY_RE = re.compile(r"(api_key=)([^&\\s]+)")


def _redact_api_key(text: str | None) -> str | None:
    if text is None:
        return None
    return _API_KEY_RE.sub(r"\1REDACTED", str(text))


@dataclass(frozen=True)
class JATSFetchResult:
    pmcid: str
    status: str  # ok|skipped|error
    out_xml: Path | None
    bytes_written: int | None
    http_status: int | None
    content_type: str | None
    article_type: str | None
    has_body: bool | None
    n_sec_tags: int | None
    error: str | None


def _efetch_url(client: EUtilsClient) -> str:
    return f"{client.base_url}/efetch.fcgi"


def _extract_article_type(xml_head: str) -> str | None:
    m = _ARTICLE_TYPE_RE.search(xml_head)
    return m.group(1) if m else None


def _scan_xml_metrics(xml_text: str) -> tuple[bool, int]:
    has_body = "<body" in xml_text.lower()
    n_sec = xml_text.lower().count("<sec")
    return has_body, int(n_sec)


def fetch_jats_xml(
    client: EUtilsClient,
    *,
    pmc_id: str,
    out_xml: Path,
    overwrite: bool = False,
    timeout_s: int = 90,
) -> JATSFetchResult:
    """Fetch JATS/XML via NCBI E-utilities efetch and write to disk (streaming)."""
    pmcid = str(pmc_id)
    if not pmcid:
        return JATSFetchResult(
            pmcid=pmcid,
            status="error",
            out_xml=None,
            bytes_written=None,
            http_status=None,
            content_type=None,
            article_type=None,
            has_body=None,
            n_sec_tags=None,
            error="Empty pmc_id",
        )

    out_xml.parent.mkdir(parents=True, exist_ok=True)
    if out_xml.exists() and not overwrite:
        # Best-effort metrics from the existing file.
        try:
            head = out_xml.read_text(encoding="utf-8", errors="ignore")[:200_000]
            article_type = _extract_article_type(head)
            has_body, n_sec = _scan_xml_metrics(head)
        except Exception:
            article_type, has_body, n_sec = None, None, None
        return JATSFetchResult(
            pmcid=pmcid,
            status="skipped",
            out_xml=out_xml,
            bytes_written=out_xml.stat().st_size,
            http_status=None,
            content_type=None,
            article_type=article_type,
            has_body=has_body,
            n_sec_tags=n_sec,
            error=None,
        )

    requests = _require_requests()

    params: dict[str, Any] = {"db": "pmc", "id": pmcid, "retmode": "xml", "rettype": "full"}
    if client.tool:
        params["tool"] = client.tool
    if client.email:
        params["email"] = client.email
    if client.api_key:
        params["api_key"] = client.api_key

    url = _efetch_url(client)
    headers = {"User-Agent": client.user_agent}

    backoff_s = 0.8
    backoff_cap_s = 10.0
    last_error: str | None = None
    for attempt in range(10):
        time.sleep(max(float(client.request_delay_s), 0.0))
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout_s, stream=True)
        except Exception as exc:
            last_error = _redact_api_key(f"Request error: {type(exc).__name__}: {exc}")
            time.sleep(min(backoff_s, backoff_cap_s))
            backoff_s = min(backoff_s * 1.7, backoff_cap_s)
            continue

        try:
            if resp.status_code == 429:
                retry_after = None
                try:
                    retry_after = float(resp.headers.get("Retry-After")) if resp.headers else None
                except Exception:
                    retry_after = None
                time.sleep(max(min(backoff_s, backoff_cap_s), retry_after or 0.0))
                backoff_s = min(backoff_s * 1.7, backoff_cap_s)
                last_error = "HTTP 429 (rate limited)"
                continue
            if resp.status_code >= 500:
                time.sleep(min(backoff_s, backoff_cap_s))
                backoff_s = min(backoff_s * 1.7, backoff_cap_s)
                last_error = f"HTTP {resp.status_code} (server error)"
                continue

            try:
                resp.raise_for_status()
            except Exception:
                reason = str(getattr(resp, "reason", "") or "").strip()
                msg = f"HTTP error: status={int(resp.status_code)}" + (f" {reason}" if reason else "")
                return JATSFetchResult(
                    pmcid=pmcid,
                    status="error",
                    out_xml=None,
                    bytes_written=None,
                    http_status=int(resp.status_code),
                    content_type=str(resp.headers.get("Content-Type")) if resp.headers else None,
                    article_type=None,
                    has_body=None,
                    n_sec_tags=None,
                    error=msg,
                )

            # Write atomically to avoid partial XML files if interrupted mid-download.
            tmp_xml = out_xml.with_name(out_xml.name + ".part")
            if tmp_xml.exists():
                try:
                    tmp_xml.unlink()
                except Exception:
                    pass

            bytes_written = 0
            try:
                with tmp_xml.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 256):
                        if not chunk:
                            continue
                        f.write(chunk)
                        bytes_written += len(chunk)
                tmp_xml.replace(out_xml)
            except Exception as exc:
                last_error = _redact_api_key(f"Download/write error: {type(exc).__name__}: {exc}")
                try:
                    if tmp_xml.exists():
                        tmp_xml.unlink()
                except Exception:
                    pass
                time.sleep(min(backoff_s, backoff_cap_s))
                backoff_s = min(backoff_s * 1.7, backoff_cap_s)
                continue
        finally:
            try:
                resp.close()
            except Exception:
                pass

        # Lightweight metrics from the first ~200k chars.
        try:
            head = out_xml.read_text(encoding="utf-8", errors="ignore")[:200_000]
            article_type = _extract_article_type(head)
            has_body, n_sec = _scan_xml_metrics(head)
        except Exception:
            article_type, has_body, n_sec = None, None, None

        return JATSFetchResult(
            pmcid=pmcid,
            status="ok",
            out_xml=out_xml,
            bytes_written=int(bytes_written),
            http_status=int(resp.status_code),
            content_type=str(resp.headers.get("Content-Type")) if resp.headers else None,
            article_type=article_type,
            has_body=bool(has_body),
            n_sec_tags=int(n_sec),
            error=None,
        )

    return JATSFetchResult(
        pmcid=pmcid,
        status="error",
        out_xml=None,
        bytes_written=None,
        http_status=None,
        content_type=None,
        article_type=None,
        has_body=None,
        n_sec_tags=None,
        error=_redact_api_key(last_error) or "Repeated failures from NCBI efetch",
    )


def fetch_jats_from_registry(
    *,
    client: EUtilsClient,
    registry_csv: Path,
    out_dir: Path,
    max_papers: int | None,
    random_seed: int,
    overwrite: bool,
    batch_size: int = 25,
    batch_sleep_s: float = 10.0,
    batch_jitter_s: float = 2.0,
    mode: Literal["random", "sequential"] = "random",
) -> tuple[pd.DataFrame, Path]:
    df = pd.read_csv(registry_csv)
    if "pmcid" not in df.columns:
        raise ValueError("Registry CSV missing required column: pmcid")

    eligible = df.copy()
    eligible["pmcid"] = eligible["pmcid"].astype(str)
    eligible = eligible[eligible["pmcid"].str.startswith("PMC", na=False)].copy()
    if eligible.empty:
        raise ValueError("No eligible PMCIDs found in registry.")

    if mode == "random":
        n = min(int(max_papers or len(eligible)), int(len(eligible)))
        sample = eligible.sample(n=n, random_state=int(random_seed))
    else:
        # Stable order: year asc, then pmcid.
        years = pd.to_numeric(eligible.get("year"), errors="coerce")
        eligible = eligible.assign(_year_sort=years)
        eligible = eligible.sort_values(["_year_sort", "pmcid"], ascending=[True, True], na_position="last").drop(
            columns=["_year_sort"]
        )
        if max_papers is not None:
            n = min(int(max_papers), int(len(eligible)))
            sample = eligible.head(n)
        else:
            sample = eligible

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = out_dir / f"jats_fetch_log__{ts}.csv"

    import csv as _csv

    fieldnames = [
        "pmcid",
        "pmid",
        "doi",
        "year",
        "title",
        "status",
        "out_xml",
        "bytes_written",
        "http_status",
        "content_type",
        "article_type",
        "has_body",
        "n_sec_tags",
        "error",
    ]

    rng = __import__("numpy").random.default_rng(int(random_seed))
    rows: list[dict[str, Any]] = []
    n_total = int(len(sample))
    batch_size = max(int(batch_size), 1)
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in enumerate(sample.to_dict(orient="records")):
            pmcid = str(row["pmcid"])
            out_xml = out_dir / f"{pmcid}.xml"
            try:
                res = fetch_jats_xml(client, pmc_id=pmcid, out_xml=out_xml, overwrite=overwrite)
            except Exception as exc:
                res = JATSFetchResult(
                    pmcid=pmcid,
                    status="error",
                    out_xml=None,
                    bytes_written=None,
                    http_status=None,
                    content_type=None,
                    article_type=None,
                    has_body=None,
                    n_sec_tags=None,
                    error=f"Unhandled exception: {type(exc).__name__}: {exc}",
                )
            rec = {
                "pmcid": pmcid,
                "pmid": row.get("pmid"),
                "doi": row.get("doi"),
                "year": row.get("year"),
                "title": row.get("title"),
                "status": res.status,
                "out_xml": str(res.out_xml) if res.out_xml else None,
                "bytes_written": res.bytes_written,
                "http_status": res.http_status,
                "content_type": res.content_type,
                "article_type": res.article_type,
                "has_body": res.has_body,
                "n_sec_tags": res.n_sec_tags,
                "error": _redact_api_key(res.error),
            }
            writer.writerow(rec)
            f.flush()
            rows.append(rec)

            # Batch-level pacing (additional to per-request delay)
            if (i + 1) % batch_size == 0 and (i + 1) < n_total:
                sleep_s = max(float(batch_sleep_s), 0.0) + float(rng.uniform(0.0, max(float(batch_jitter_s), 0.0)))
                time.sleep(sleep_s)

    out_df = pd.DataFrame(rows)
    return out_df, log_path
