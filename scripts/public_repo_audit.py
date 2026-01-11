#!/usr/bin/env python3
"""
Public-repo audit for accidental disclosure.

This repository is intended to be public. Do not commit:
  - downloaded full text (JATS/XML) or extracted narrative segments
  - per-case Phase C label files
  - clinician rating workbooks or per-case rater exports
  - manuscript sources (markdown, figures, docx)

This script performs a lightweight, name/path-based audit to catch common mistakes.
It is not a substitute for human review.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Finding:
    path: Path
    reason: str


ALLOWED_UNDER_DATA_OR_RESULTS = {
    Path("data/README.md"),
    Path("data/external/.gitkeep"),
    Path("data/interim/.gitkeep"),
    Path("data/processed/.gitkeep"),
    Path("data/raw/.gitkeep"),
    Path("results/.gitkeep"),
    Path("results/README.md"),
}

FORBIDDEN_EXTS = {
    ".xlsx",
    ".xls",
    ".parquet",
    ".pq",
    ".pkl",
    ".pickle",
    ".joblib",
    ".npz",
    ".pt",
    ".pth",
    ".ckpt",
    ".h5",
    ".hdf5",
}

FORBIDDEN_FILENAMES = {
    "adjudication_template.csv",
    "segments.csv",
    "case_labels_long.csv",
    "case_labels_wide.csv",
    "case_labels.csv",
}


def _iter_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts]


def audit_repo(root: Path) -> list[Finding]:
    findings: list[Finding] = []

    for abs_path in _iter_files(root):
        rel = abs_path.relative_to(root)
        rel_posix = rel.as_posix()

        if rel.parts and rel.parts[0] == "manuscript":
            findings.append(Finding(rel, "Manuscript sources are excluded from this public repo"))
            continue

        if rel.suffix.lower() in FORBIDDEN_EXTS:
            findings.append(Finding(rel, f"Forbidden file type: {rel.suffix.lower()}"))
            continue

        if rel.parts and rel.parts[0] in {"data", "results"} and rel not in ALLOWED_UNDER_DATA_OR_RESULTS:
            findings.append(Finding(rel, f"Must not commit artifacts under `{rel.parts[0]}/`"))
            continue

        name_lc = rel.name.lower()
        if name_lc in FORBIDDEN_FILENAMES:
            findings.append(Finding(rel, "Forbidden per-case artifact filename"))
            continue

        if rel.suffix.lower() == ".csv":
            if ("segments" in name_lc) or ("case_labels" in name_lc) or ("case-labels" in name_lc):
                findings.append(Finding(rel, "Looks like extracted segments or per-case label export"))
                continue

            if ("rater" in name_lc) or ("ratings" in name_lc) or ("kappa" in name_lc) or ("adjudication" in name_lc):
                findings.append(Finding(rel, "Looks like clinician rater output (may contain per-case labels)"))
                continue

        if "ratings" in rel_posix.lower() and rel.suffix.lower() in {".md", ".txt"}:
            # Docs are OK, but filenames resembling the private workbook can be a warning sign.
            if "ftd ratings" in rel_posix.lower():
                findings.append(Finding(rel, "Filename resembles private clinician ratings workbook"))
                continue

    return findings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repo root to audit (default: parent of scripts/).",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if any findings are detected.",
    )
    args = ap.parse_args()

    root = args.root.resolve()
    findings = audit_repo(root)
    if findings:
        print("Public-repo audit findings:")
        for f in findings:
            print(f"- {f.path}: {f.reason}")
        if args.check:
            return 2

    print("Public-repo audit: OK (no obvious sensitive artifacts detected).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
