from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd


_FILENAME_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_filename(text: str, *, max_len: int = 60) -> str:
    txt = _FILENAME_SAFE_RE.sub("_", text.strip())
    txt = txt.strip("._-")
    if not txt:
        return "paper"
    return txt[:max_len]


def _suggest_filename(row: pd.Series) -> str:
    pmcid = str(row.get("pmcid") or "PMCID_UNKNOWN")
    year = row.get("year")
    year_str = str(int(year)) if pd.notna(year) else "YEAR_UNKNOWN"
    author = str(row.get("first_author") or "")
    author_last = author.split(" ")[0] if author else "Author"
    title = str(row.get("title") or "")
    return "__".join(
        [
            _sanitize_filename(pmcid, max_len=20),
            _sanitize_filename(year_str, max_len=10),
            _sanitize_filename(author_last, max_len=20),
            _sanitize_filename(title, max_len=60),
        ]
    ) + ".pdf"


@dataclass(frozen=True)
class ManifestOutputs:
    manifest_csv: Path


def build_download_manifest(
    *,
    registry_csv: Path,
    out_csv: Path,
    pdf_dir: Path,
) -> ManifestOutputs:
    df = pd.read_csv(registry_csv)
    df = df.copy()
    df["suggested_pdf_filename"] = df.apply(_suggest_filename, axis=1)

    # Ensure uniqueness of suggested filenames.
    seen: dict[str, int] = {}
    unique = []
    for fn in df["suggested_pdf_filename"].astype(str).tolist():
        n = seen.get(fn, 0)
        if n == 0:
            unique.append(fn)
        else:
            stem = fn[:-4] if fn.lower().endswith(".pdf") else fn
            unique.append(f"{stem}__dup{n}.pdf")
        seen[fn] = n + 1
    df["suggested_pdf_filename"] = unique
    df["local_pdf_path"] = df["suggested_pdf_filename"].map(lambda fn: str((pdf_dir / fn).as_posix()))

    # Screening + download workflow fields
    df["screen_include"] = ""  # y/n
    df["screen_reason"] = ""
    df["download_status"] = ""  # todo/downloaded/failed
    df["notes"] = ""

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_text(df.to_csv(index=False), encoding="utf-8")
    return ManifestOutputs(manifest_csv=out_csv)

