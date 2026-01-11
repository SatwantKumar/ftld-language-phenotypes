from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from paperjn.utils.paths import ensure_dir


SYMPTOM_TAGS_V1 = [
    "apraxia_of_speech",
    "agrammatism",
    "semantic_loss",
    "behavioral_change_disinhibition_or_apathy",
    "compulsions_or_rigid_routines",
    "parkinsonism",
    "oculomotor_vertical_gaze_palsy",
    "limb_apraxia_or_alien_limb",
    "mnd_signs",
    "psychosis_hallucinations",
]


def _utc_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_json_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return []
    try:
        parsed = json.loads(s)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(x) for x in parsed if str(x).strip()]


def _allocate_stratified_counts(
    *,
    strata_counts: pd.Series,
    sample_size: int,
    random_seed: int,
) -> pd.Series:
    """Allocate integer sample counts per stratum with >=1 per stratum (when feasible)."""
    if sample_size < 1:
        raise ValueError("sample_size must be >= 1")
    if strata_counts.empty:
        raise ValueError("No strata available for sampling.")
    if (strata_counts < 1).any():
        raise ValueError("strata_counts must be >= 1 for all strata.")

    total = int(strata_counts.sum())
    n_strata = int(len(strata_counts))
    n = int(min(sample_size, total))

    # If n < n_strata, we can't guarantee 1 per stratum.
    if n < n_strata:
        rng = np.random.default_rng(int(random_seed))
        chosen = rng.choice(strata_counts.index.to_numpy(), size=n, replace=False)
        out = pd.Series(0, index=strata_counts.index, dtype=int)
        out.loc[chosen] = 1
        return out

    expected = (n * strata_counts / total).astype(float)
    base = np.floor(expected).astype(int)
    targets = base.copy()
    targets[targets < 1] = 1
    targets = targets.clip(upper=strata_counts.astype(int))

    cur = int(targets.sum())
    if cur < n:
        rema = (expected - base).sort_values(ascending=False)
        for key in rema.index:
            if cur >= n:
                break
            if int(targets.loc[key]) < int(strata_counts.loc[key]):
                targets.loc[key] += 1
                cur += 1
    elif cur > n:
        # Reduce from largest targets while keeping >=1.
        for key in targets.sort_values(ascending=False).index:
            if cur <= n:
                break
            if int(targets.loc[key]) > 1:
                targets.loc[key] -= 1
                cur -= 1

    if int(targets.sum()) != n:
        raise RuntimeError("Failed to allocate exact stratified sample counts.")
    return targets.astype(int)


@dataclass(frozen=True)
class RaterSampleOutputs:
    out_dir: Path
    sample_manifest_internal_csv: Path
    rater1_template_csv: Path
    rater2_template_csv: Path
    packets_dir: Path
    rater_instructions_md: Path
    coverage_report_md: Path
    meta_json: Path


def make_rater_sample(
    *,
    phase_d_run_dir: Path,
    out_dir: Path | None,
    sample_size: int,
    random_seed: int,
    max_chars_per_case: int,
    include_title_in_packets: bool,
    text_field: str,
) -> RaterSampleOutputs:
    """Create a clinician-facing sample with per-case text packets and blank rating templates."""
    phase_d_run_dir = Path(phase_d_run_dir).resolve()
    run_cfg_path = phase_d_run_dir / "run_config.json"
    if not run_cfg_path.exists():
        raise FileNotFoundError(f"Missing Phase D run_config.json: {run_cfg_path}")
    run_cfg = json.loads(run_cfg_path.read_text(encoding="utf-8"))
    segments_csv = Path(run_cfg["segments_csv"]).resolve()

    case_table_path = phase_d_run_dir / "tables" / "case_table_with_clusters.csv"
    if not case_table_path.exists():
        raise FileNotFoundError(f"Missing case table: {case_table_path}")

    case_df = pd.read_csv(case_table_path)
    if "status" in case_df.columns:
        case_df = case_df[case_df["status"].astype(str) == "ok"].copy()
    case_df["matched_cluster"] = case_df["cluster_matched_to_discovery"].astype(int)
    case_df["split"] = case_df["split"].astype(str)
    case_df["ftld_syndrome_reported"] = case_df["ftld_syndrome_reported"].astype(str)

    total_cases = int(len(case_df))
    if total_cases == 0:
        raise RuntimeError("No cases found in Phase D case table.")
    n = int(min(sample_size, total_cases))

    # Stratify on split × matched_cluster × syndrome (publication-ready, reviewer-proof sampling).
    case_df["stratum"] = (
        case_df["split"].astype(str)
        + "|"
        + case_df["matched_cluster"].astype(str)
        + "|"
        + case_df["ftld_syndrome_reported"].astype(str)
    )
    strata_counts = case_df.groupby("stratum").size().sort_index()
    targets = _allocate_stratified_counts(strata_counts=strata_counts, sample_size=n, random_seed=random_seed)

    rng = np.random.default_rng(int(random_seed))
    picks = []
    for stratum, take_n in targets.items():
        sub = case_df[case_df["stratum"] == stratum].copy()
        if int(take_n) >= len(sub):
            picks.append(sub)
            continue
        rs = int(rng.integers(0, 2**31 - 1))
        picks.append(sub.sample(n=int(take_n), random_state=rs))

    sample_df = pd.concat(picks, ignore_index=True)
    if len(sample_df) != n:
        raise RuntimeError(f"Sampling produced n={len(sample_df)} cases (expected {n}).")

    sample_df = sample_df.sort_values(["split", "matched_cluster", "pmcid", "case_id"]).reset_index(drop=True)
    sample_df.insert(0, "case_uid", [f"RAT{(i+1):04d}" for i in range(len(sample_df))])

    if out_dir is None:
        out_dir = (phase_d_run_dir.parent / "validation" / f"rater_sample__{_utc_slug()}").resolve()
    out_dir = ensure_dir(Path(out_dir).resolve())
    internal_dir = ensure_dir(out_dir / "internal")
    share_dir = ensure_dir(out_dir / "share")
    packets_dir = ensure_dir(share_dir / "packets")

    # Pull extracted clinical segments for the selected cases.
    seg_cols = [
        "pmcid",
        "case_id",
        "segment_uid",
        "segment_type",
        "include_for_embedding",
        "leakage_n_matches_clean",
        "sec_paths_json",
        "text_raw",
        "text_clean",
    ]
    seg = pd.read_csv(segments_csv, usecols=[c for c in seg_cols if c in pd.read_csv(segments_csv, nrows=0).columns])
    seg["pmcid"] = seg["pmcid"].astype(str)
    seg["case_id"] = seg["case_id"].astype(str)
    if "include_for_embedding" in seg.columns:
        seg["include_for_embedding"] = seg["include_for_embedding"].astype(bool)
        seg = seg[seg["include_for_embedding"]].copy()
    if "leakage_n_matches_clean" in seg.columns:
        leak = pd.to_numeric(seg["leakage_n_matches_clean"], errors="coerce").fillna(0).astype(int)
        seg = seg[leak == 0].copy()

    if text_field not in {"text_raw", "text_clean"}:
        raise ValueError("text_field must be one of: text_raw|text_clean")
    if text_field not in seg.columns:
        raise RuntimeError(f"Segments CSV missing {text_field}: {segments_csv}")

    key = sample_df[["pmcid", "case_id"]].drop_duplicates()
    seg = seg.merge(key, on=["pmcid", "case_id"], how="inner")
    seg = seg.sort_values(["pmcid", "case_id", "segment_uid"]).reset_index(drop=True)

    packet_paths: list[str] = []
    packet_rel_paths: list[str] = []
    for row in sample_df.itertuples(index=False):
        pmcid = str(row.pmcid)
        case_id = str(row.case_id)
        case_uid = str(row.case_uid)
        parts = []
        parts.append(f"# {case_uid}")
        parts.append("")
        parts.append(f"- PMCID: {pmcid}")
        parts.append(f"- Case ID: {case_id}")
        if include_title_in_packets and hasattr(row, "title"):
            parts.append(f"- Title: {getattr(row, 'title', '')}")
        parts.append("")

        s = seg[(seg["pmcid"] == pmcid) & (seg["case_id"] == case_id)].copy()
        if s.empty:
            parts.append("_No eligible extracted segments found for this case._")
        else:
            for seg_row in s.itertuples(index=False):
                seg_type = getattr(seg_row, "segment_type", "segment")
                sec_paths = _parse_json_list(getattr(seg_row, "sec_paths_json", None))
                header = f"## {seg_type}"
                if sec_paths:
                    header += f" ({' > '.join(sec_paths)})"
                parts.append(header)
                parts.append("")
                txt = str(getattr(seg_row, text_field, "") or "").strip()
                if max_chars_per_case and len(txt) > int(max_chars_per_case):
                    txt = txt[: int(max_chars_per_case)] + "\n\n[TRUNCATED]"
                parts.append(txt if txt else "_(empty)_")
                parts.append("")

        packet_path = packets_dir / f"{case_uid}.md"
        packet_path.write_text("\n".join(parts).strip() + "\n", encoding="utf-8")
        packet_paths.append(str(packet_path))
        packet_rel_paths.append(str(Path("packets") / f"{case_uid}.md"))

    sample_manifest = sample_df.copy()
    sample_manifest["packet_path"] = packet_paths
    sample_manifest["packet_rel_path"] = packet_rel_paths
    sample_manifest_internal_csv = internal_dir / "sample_manifest_internal.csv"
    sample_manifest.to_csv(sample_manifest_internal_csv, index=False)

    # Rater templates (binary labels; blank cells allowed).
    # Keep raters blinded to split/cluster/syndrome to reduce bias.
    template = sample_manifest[["case_uid", "pmcid", "case_id", "packet_rel_path"]].copy()
    template = template.rename(columns={"packet_rel_path": "packet_path"})
    template["text_adequate"] = ""
    template["is_ftld_spectrum"] = ""
    template["is_ppa"] = ""
    for tag in SYMPTOM_TAGS_V1:
        template[f"tag__{tag}"] = ""
    template["notes"] = ""

    rater1_template_csv = share_dir / "rater1_template.csv"
    rater2_template_csv = share_dir / "rater2_template.csv"
    template.to_csv(rater1_template_csv, index=False)
    template.to_csv(rater2_template_csv, index=False)

    # Clinician instructions (shareable)
    rater_instructions_md = share_dir / "RATER_INSTRUCTIONS.md"
    instr_lines: list[str] = []
    instr_lines.append("# Clinician Rating Instructions (Blinded)")
    instr_lines.append("")
    instr_lines.append("## What you are rating")
    instr_lines.append(
        "- Each row is one case unit (identified by `case_uid`). Please rate using only the provided packet text."
    )
    instr_lines.append("- Do not look up the original paper; treat the packet as the source document.")
    instr_lines.append("")
    instr_lines.append("## How to fill the CSV")
    instr_lines.append("- Use `1` = yes/present, `0` = no/not present or not stated, blank = not rateable.")
    instr_lines.append("- Open the case packet referenced by `packet_path` (relative to the `share/` folder).")
    instr_lines.append("- First fill `text_adequate`:")
    instr_lines.append("  - `1` if there is enough clinical description to rate FTLD/PPA and tags.")
    instr_lines.append("  - `0` if the packet is insufficient, garbled, or lacks clinical detail.")
    instr_lines.append("- If `text_adequate = 0`, leave all other label fields blank.")
    instr_lines.append("")
    instr_lines.append("## Label definitions")
    instr_lines.append("- `is_ftld_spectrum`: `1` only if an FTLD-spectrum neurodegenerative syndrome is explicitly described/diagnosed in the packet; otherwise `0`.")
    instr_lines.append(
        "- `is_ppa`: `1` only for neurodegenerative PPA (or a clear progressive language-led syndrome); stroke/post-stroke aphasia should be `0`."
    )
    instr_lines.append("")
    instr_lines.append("## Symptom tags (binary `present` only)")
    for tag in SYMPTOM_TAGS_V1:
        instr_lines.append(f"- `tag__{tag}`: `1` only if explicitly present; otherwise `0`.")
    instr_lines.append("")
    instr_lines.append("- `notes`: optional comments (e.g., uncertainty, key excerpts).")
    instr_lines.append("")
    rater_instructions_md.write_text("\n".join(instr_lines) + "\n", encoding="utf-8")

    # Coverage report
    def _dist(df: pd.DataFrame) -> pd.DataFrame:
        out = (
            df.groupby(["split", "matched_cluster", "ftld_syndrome_reported"], dropna=False)
            .size()
            .reset_index(name="n")
        )
        return out.sort_values(["split", "matched_cluster", "n"], ascending=[True, True, False])

    full_dist = _dist(case_df)
    sample_dist = _dist(sample_manifest)

    lines: list[str] = []
    lines.append("# Rater Sample Coverage Report")
    lines.append("")
    lines.append(f"Phase D run: `{phase_d_run_dir}`")
    lines.append(f"- Sample size: {n} / {total_cases}")
    lines.append(f"- Random seed: {int(random_seed)}")
    lines.append("- Stratification: split × matched_cluster × ftld_syndrome_reported")
    lines.append(f"- Text field: `{text_field}` (truncation cap: {int(max_chars_per_case)} chars/segment)")
    lines.append("")
    lines.append("## Sample Distribution")
    lines.append(sample_dist.to_markdown(index=False))
    lines.append("")
    lines.append("## Full Distribution")
    lines.append(full_dist.to_markdown(index=False))
    lines.append("")

    coverage_report_md = internal_dir / "coverage_report.md"
    coverage_report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    meta = {
        "phase_d_run_dir": str(phase_d_run_dir),
        "segments_csv": str(segments_csv),
        "sample_size_requested": int(sample_size),
        "sample_size_actual": int(n),
        "random_seed": int(random_seed),
        "stratification": "split|matched_cluster|ftld_syndrome_reported",
        "max_chars_per_case": int(max_chars_per_case),
        "include_title_in_packets": bool(include_title_in_packets),
        "text_field": str(text_field),
        "tags": SYMPTOM_TAGS_V1,
        "rater_columns": [
            "text_adequate",
            "is_ftld_spectrum",
            "is_ppa",
            *[f"tag__{t}" for t in SYMPTOM_TAGS_V1],
            "notes",
        ],
    }
    meta_json = internal_dir / "meta.json"
    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return RaterSampleOutputs(
        out_dir=out_dir,
        sample_manifest_internal_csv=sample_manifest_internal_csv,
        rater1_template_csv=rater1_template_csv,
        rater2_template_csv=rater2_template_csv,
        packets_dir=packets_dir,
        rater_instructions_md=rater_instructions_md,
        coverage_report_md=coverage_report_md,
        meta_json=meta_json,
    )


@dataclass(frozen=True)
class RaterKappaOutputs:
    out_dir: Path
    kappa_csv: Path
    kappa_md: Path
    adjudication_csv: Path


def _coerce_binary(series: pd.Series) -> pd.Series:
    s = series.copy()
    s = s.replace({"": np.nan, "NA": np.nan, "NaN": np.nan, "nan": np.nan})
    s = pd.to_numeric(s, errors="coerce")
    # allow only 0/1
    s = s.where(s.isin([0, 1]))
    return s.astype("float")


def score_two_raters(
    *,
    rater1_csv: Path,
    rater2_csv: Path,
    out_dir: Path | None,
) -> RaterKappaOutputs:
    """Compute per-field Cohen's kappa and emit an adjudication sheet for disagreements."""
    r1 = pd.read_csv(rater1_csv)
    r2 = pd.read_csv(rater2_csv)
    if "case_uid" not in r1.columns or "case_uid" not in r2.columns:
        raise ValueError("Both rater files must include a case_uid column.")

    key = ["case_uid"]
    df = r1.merge(r2, on=key, how="inner", suffixes=("_r1", "_r2"))
    if df.empty:
        raise RuntimeError("No overlapping case_uid rows between rater files.")

    # If `text_adequate` is present, compute agreement for other labels only among rows where both raters
    # judged the packet adequate (text_adequate == 1).
    adequate_mask = pd.Series(True, index=df.index)
    if "text_adequate_r1" in df.columns and "text_adequate_r2" in df.columns:
        a_ok = _coerce_binary(df["text_adequate_r1"])
        b_ok = _coerce_binary(df["text_adequate_r2"])
        adequate_mask = (a_ok == 1) & (b_ok == 1)

    fields = ["text_adequate", "is_ftld_spectrum", "is_ppa"] + [f"tag__{t}" for t in SYMPTOM_TAGS_V1]
    rows: list[dict[str, Any]] = []

    for field in fields:
        a_col = f"{field}_r1"
        b_col = f"{field}_r2"
        if a_col not in df.columns or b_col not in df.columns:
            continue
        a = _coerce_binary(df[a_col])
        b = _coerce_binary(df[b_col])
        keep = a.notna() & b.notna()
        if field != "text_adequate":
            keep = keep & adequate_mask
        n = int(keep.sum())
        if n == 0:
            rows.append(
                {
                    "field": field,
                    "n_pairs": 0,
                    "kappa": float("nan"),
                    "agreement": float("nan"),
                    "r1_positive_rate": float("nan"),
                    "r2_positive_rate": float("nan"),
                }
            )
            continue
        a2 = a[keep].astype(int).to_numpy()
        b2 = b[keep].astype(int).to_numpy()
        kappa = float(cohen_kappa_score(a2, b2))
        agreement = float((a2 == b2).mean())
        rows.append(
            {
                "field": field,
                "n_pairs": n,
                "kappa": kappa,
                "agreement": agreement,
                "r1_positive_rate": float(a2.mean()),
                "r2_positive_rate": float(b2.mean()),
            }
        )

    if out_dir is None:
        out_dir = Path(rater1_csv).resolve().parent / f"kappa__{_utc_slug()}"
    out_dir = ensure_dir(Path(out_dir).resolve())

    kappa_df = pd.DataFrame(rows).sort_values(["field"])
    kappa_csv = out_dir / "kappa_report.csv"
    kappa_df.to_csv(kappa_csv, index=False)

    lines: list[str] = []
    lines.append("# Rater Agreement (Cohen's κ)")
    lines.append("")
    lines.append(f"- Rater 1: `{Path(rater1_csv).resolve()}`")
    lines.append(f"- Rater 2: `{Path(rater2_csv).resolve()}`")
    if "text_adequate_r1" in df.columns and "text_adequate_r2" in df.columns:
        lines.append(f"- Rows with both `text_adequate=1`: {int(adequate_mask.sum())} / {int(len(df))}")
    lines.append("")
    lines.append(kappa_df.to_markdown(index=False))
    lines.append("")

    kappa_md = out_dir / "kappa_report.md"
    kappa_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Adjudication sheet (only disagreements where both provided values)
    adjud_rows = []
    for _, row in df.iterrows():
        any_disagree = False
        out = {"case_uid": row["case_uid"]}
        for col in ["pmcid", "case_id", "split", "matched_cluster", "packet_path", "ftld_syndrome_reported"]:
            c1 = f"{col}_r1"
            if c1 in row.index:
                out[col] = row[c1]
        a_ok_val = None
        b_ok_val = None
        if "text_adequate_r1" in row.index and "text_adequate_r2" in row.index:
            a_ok_val = _coerce_binary(pd.Series([row["text_adequate_r1"]])).iloc[0]
            b_ok_val = _coerce_binary(pd.Series([row["text_adequate_r2"]])).iloc[0]
        both_adequate = bool(a_ok_val == 1 and b_ok_val == 1) if a_ok_val is not None and b_ok_val is not None else True

        for field in fields:
            a_col = f"{field}_r1"
            b_col = f"{field}_r2"
            if a_col not in row.index or b_col not in row.index:
                continue
            a = _coerce_binary(pd.Series([row[a_col]])).iloc[0]
            b = _coerce_binary(pd.Series([row[b_col]])).iloc[0]
            if pd.isna(a) or pd.isna(b):
                continue
            if field != "text_adequate" and not both_adequate:
                continue
            if int(a) != int(b):
                any_disagree = True
            out[f"{field}_r1"] = "" if pd.isna(a) else int(a)
            out[f"{field}_r2"] = "" if pd.isna(b) else int(b)
            out[f"{field}_final"] = ""
        if any_disagree:
            adjud_rows.append(out)

    adjudication_csv = out_dir / "adjudication_template.csv"
    pd.DataFrame(adjud_rows).to_csv(adjudication_csv, index=False)

    return RaterKappaOutputs(
        out_dir=out_dir,
        kappa_csv=kappa_csv,
        kappa_md=kappa_md,
        adjudication_csv=adjudication_csv,
    )
