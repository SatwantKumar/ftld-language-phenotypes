from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from paperjn.pmcoa.rater_validation import _allocate_stratified_counts  # type: ignore
from paperjn.pmcoa.rater_workbook import make_rater_workbooks
from paperjn.utils.paths import ensure_dir


def _utc_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@dataclass(frozen=True)
class RaterSubsetOutputs:
    out_dir: Path
    share_dir: Path
    internal_dir: Path
    rater1_template_csv: Path
    rater2_template_csv: Path
    rater1_workbook_xlsx: Path
    rater2_workbook_xlsx: Path
    rater_instructions_md: Path
    manifest_internal_csv: Path
    coverage_report_md: Path
    meta_json: Path
    share_sha256_csv: Path


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_share_checksums(*, share_dir: Path, internal_dir: Path) -> Path:
    rows = []
    for path in sorted([p for p in share_dir.rglob("*") if p.is_file()]):
        rel = path.relative_to(share_dir)
        rows.append((str(rel), _sha256_file(path), path.stat().st_size))
    out_csv = internal_dir / "share_sha256.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["share_rel_path", "sha256", "bytes"])
        w.writerows(rows)
    (internal_dir / "share_file_list.txt").write_text("\n".join(r[0] for r in rows) + "\n", encoding="utf-8")
    return out_csv


def make_rater_subset_n(
    *,
    parent_rater_sample_dir: Path,
    out_dir: Path | None,
    sample_size: int,
    random_seed: int,
) -> RaterSubsetOutputs:
    """Create a locked, stratified subset of an existing rater sample directory.

    Stratification (locked): `split × matched_cluster`.
    Rating columns (locked): `text_adequate`, `is_ftld_spectrum`, `is_ppa`, `notes`.
    """
    parent_rater_sample_dir = Path(parent_rater_sample_dir).resolve()
    parent_share = parent_rater_sample_dir / "share"
    parent_internal = parent_rater_sample_dir / "internal"
    parent_manifest = parent_internal / "sample_manifest_internal.csv"
    parent_packets = parent_share / "packets"
    if not parent_manifest.exists():
        raise FileNotFoundError(f"Missing parent internal manifest: {parent_manifest}")
    if not parent_packets.exists():
        raise FileNotFoundError(f"Missing parent packets dir: {parent_packets}")

    df = pd.read_csv(parent_manifest)
    required = {"case_uid", "pmcid", "case_id", "split", "matched_cluster"}
    missing = sorted([c for c in required if c not in df.columns])
    if missing:
        raise ValueError(f"Parent manifest missing required columns: {missing}")

    df["split"] = df["split"].astype(str)
    df["matched_cluster"] = pd.to_numeric(df["matched_cluster"], errors="coerce").astype(int)
    df["pmcid"] = df["pmcid"].astype(str)
    df["case_id"] = df["case_id"].astype(str)
    df["case_uid"] = df["case_uid"].astype(str)

    total = int(len(df))
    if total == 0:
        raise RuntimeError("Parent manifest has no rows.")
    n = int(min(int(sample_size), total))

    df["stratum"] = df["split"].astype(str) + "|" + df["matched_cluster"].astype(str)
    strata_counts = df.groupby("stratum").size().sort_index()
    targets = _allocate_stratified_counts(strata_counts=strata_counts, sample_size=n, random_seed=int(random_seed))

    rng = np.random.default_rng(int(random_seed))
    picks = []
    for stratum, take_n in targets.items():
        sub = df[df["stratum"] == stratum].copy()
        if int(take_n) >= len(sub):
            picks.append(sub)
            continue
        rs = int(rng.integers(0, 2**31 - 1))
        picks.append(sub.sample(n=int(take_n), random_state=rs))
    sample = pd.concat(picks, ignore_index=True)
    if len(sample) != n:
        raise RuntimeError(f"Sampling produced n={len(sample)} rows (expected {n}).")
    sample = sample.sort_values(["split", "matched_cluster", "pmcid", "case_id"]).reset_index(drop=True)

    ts = _utc_slug()
    if out_dir is None:
        out_dir = (parent_rater_sample_dir.parent / f"rater_subset_n{n}__{ts}").resolve()
    out_dir = ensure_dir(Path(out_dir).resolve())
    share_dir = ensure_dir(out_dir / "share")
    internal_dir = ensure_dir(out_dir / "internal")
    packets_dir = ensure_dir(share_dir / "packets")

    # Copy packets (keep original case_uid so it cross-references the parent sample for audit).
    for case_uid in sample["case_uid"].astype(str).tolist():
        src = parent_packets / f"{case_uid}.md"
        if not src.exists():
            raise FileNotFoundError(f"Missing packet in parent sample: {src}")
        shutil.copy2(src, packets_dir / src.name)

    # Share templates (blinded)
    tmpl = sample[["case_uid", "pmcid", "case_id"]].copy()
    tmpl["packet_path"] = tmpl["case_uid"].astype(str).apply(lambda x: str(Path("packets") / f"{x}.md"))
    tmpl["text_adequate"] = ""
    tmpl["is_ftld_spectrum"] = ""
    tmpl["is_ppa"] = ""
    tmpl["notes"] = ""

    rater1_template_csv = share_dir / "rater1_template.csv"
    rater2_template_csv = share_dir / "rater2_template.csv"
    tmpl.to_csv(rater1_template_csv, index=False)
    tmpl.to_csv(rater2_template_csv, index=False)

    # Instructions (reduced-scope rubric)
    rater_instructions_md = share_dir / "RATER_INSTRUCTIONS.md"
    lines: list[str] = []
    lines.append("# Clinician Rating Instructions (Blinded; Reduced Subset)")
    lines.append("")
    lines.append("This file is the reduced-scope human validation subset (locked).")
    lines.append("")
    lines.append("## Columns to fill (binary)")
    lines.append("- `text_adequate`: 1 if enough clinical narrative is present to rate; 0 if insufficient/garbled.")
    lines.append("- If `text_adequate=0`, leave the remaining fields blank for that row.")
    lines.append("- `is_ftld_spectrum`: 1 only if an FTLD-spectrum neurodegenerative syndrome is explicitly described/diagnosed; else 0.")
    lines.append("- `is_ppa`: 1 only for neurodegenerative PPA / progressive language-led syndrome; stroke/post-stroke aphasia => 0.")
    lines.append("- `notes`: optional brief comment if ambiguous.")
    lines.append("")
    lines.append("## How to rate")
    lines.append("- Use ONLY the provided packet text; do not look up the paper.")
    lines.append("- Do not use AI/LLM tools while rating.")
    lines.append("- `packet_path` is relative to the `share/` folder.")
    lines.append("")
    rater_instructions_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Internal manifest (keeps linkage to parent sample + preserves internal columns)
    internal_df = sample.copy()
    internal_df["packet_rel_path"] = internal_df["case_uid"].astype(str).apply(lambda x: str(Path("packets") / f"{x}.md"))
    internal_df["packet_path"] = internal_df["packet_rel_path"].map(lambda rp: str((packets_dir / Path(rp).name).resolve()))
    manifest_internal_csv = internal_dir / "sample_manifest_internal.csv"
    internal_df.to_csv(manifest_internal_csv, index=False)

    # Coverage report (subset vs parent, by split×matched_cluster)
    def _dist(x: pd.DataFrame) -> pd.DataFrame:
        out = x.groupby(["split", "matched_cluster"], dropna=False).size().reset_index(name="n")
        return out.sort_values(["split", "matched_cluster"])

    parent_dist = _dist(df)
    subset_dist = _dist(sample)

    coverage_report_md = internal_dir / "coverage_report.md"
    rep_lines: list[str] = []
    rep_lines.append("# Rater Subset Coverage Report (n=locked)")
    rep_lines.append("")
    rep_lines.append(f"- Parent sample: `{parent_rater_sample_dir}` (n={int(len(df))})")
    rep_lines.append(f"- Subset: `{out_dir}` (n={int(n)})")
    rep_lines.append(f"- Random seed: {int(random_seed)}")
    rep_lines.append("- Stratification: split × matched_cluster")
    rep_lines.append("")
    rep_lines.append("## Subset distribution (split × matched_cluster)")
    rep_lines.append(subset_dist.to_markdown(index=False))
    rep_lines.append("")
    rep_lines.append("## Parent distribution (split × matched_cluster)")
    rep_lines.append(parent_dist.to_markdown(index=False))
    rep_lines.append("")
    coverage_report_md.write_text("\n".join(rep_lines) + "\n", encoding="utf-8")

    meta = {
        "created_utc": ts,
        "parent_rater_sample_dir": str(parent_rater_sample_dir),
        "sample_size_requested": int(sample_size),
        "sample_size_actual": int(n),
        "random_seed": int(random_seed),
        "stratification": "split|matched_cluster",
        "rating_columns": ["text_adequate", "is_ftld_spectrum", "is_ppa", "notes"],
        "case_uids": sample["case_uid"].astype(str).tolist(),
        "parent_manifest_sha256": _sha256_file(parent_manifest),
    }
    meta_json = internal_dir / "meta.json"
    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Build workbooks with embedded packet text.
    wb_outputs = make_rater_workbooks(rater_sample_dir=out_dir, out_dir=share_dir)
    r1_xlsx = wb_outputs.rater1_xlsx
    r2_xlsx = wb_outputs.rater2_xlsx

    share_sha256_csv = _write_share_checksums(share_dir=share_dir, internal_dir=internal_dir)

    return RaterSubsetOutputs(
        out_dir=out_dir,
        share_dir=share_dir,
        internal_dir=internal_dir,
        rater1_template_csv=rater1_template_csv,
        rater2_template_csv=rater2_template_csv,
        rater1_workbook_xlsx=r1_xlsx,
        rater2_workbook_xlsx=r2_xlsx,
        rater_instructions_md=rater_instructions_md,
        manifest_internal_csv=manifest_internal_csv,
        coverage_report_md=coverage_report_md,
        meta_json=meta_json,
        share_sha256_csv=share_sha256_csv,
    )

