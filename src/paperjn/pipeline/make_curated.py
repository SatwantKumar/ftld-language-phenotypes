from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from paperjn.config import ProjectConfig
from paperjn.nlp.leakage import audit_text_for_leakage, remove_blacklisted_terms
from paperjn.nlp.text import normalize_text, normalize_whitespace
from paperjn.utils.paths import ensure_dir, resolve_path


@dataclass(frozen=True)
class CuratedBuildOutputs:
    curated_csv: Path
    pre_leakage_report_csv: Path
    post_leakage_report_csv: Path


def _project_root_from_config_path(config_path: Path) -> Path:
    if config_path.parent.name == "configs":
        return config_path.parent.parent.resolve()
    return config_path.parent.resolve()


def _drop_leaky_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    drop_cols: list[str] = []
    for c in df.columns:
        if c.startswith("Embeddings_") or c.startswith("Cluster_"):
            drop_cols.append(c)
    for c in ["Enriched Description"]:
        if c in df.columns:
            drop_cols.append(c)
    drop_cols = sorted(set(drop_cols))
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df, drop_cols


def make_curated_table(
    config: ProjectConfig,
    *,
    config_path: Path,
    input_csv: str | Path,
    overwrite: bool = False,
) -> CuratedBuildOutputs:
    """Create `data/raw/curated_ftd_table.csv` from an existing project CSV.

    Sanity checks:
    - Uses the configured text column (primary: human-authored `Description`)
    - Drops obvious derived columns (Embeddings_*, Cluster_*, Enriched Description)
    - Filters to configured stages and non-empty required fields
    - Produces pre- and post-redaction leakage reports (post must be zero)
    """
    project_root = _project_root_from_config_path(config_path)

    input_path = resolve_path(project_root, str(input_csv))
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    out_path = resolve_path(project_root, config.paths.curated_input_csv)
    ensure_dir(out_path.parent)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing curated CSV: {out_path}")

    df = pd.read_csv(input_path)
    df, dropped = _drop_leaky_columns(df)

    required = [config.columns.stage, config.columns.subtype, config.columns.text]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    df = df.copy()
    df = df[df[config.columns.stage].isin(config.stages)]
    df = df[df[config.columns.text].notna()]
    df = df[df[config.columns.subtype].notna()]
    df[config.columns.text] = df[config.columns.text].astype(str)
    df[config.columns.subtype] = df[config.columns.subtype].astype(str)
    df[config.columns.stage] = df[config.columns.stage].astype(str)
    df = df[df[config.columns.text].str.strip().astype(bool)]
    df = df.reset_index(drop=True)

    # Leakage audit (pre and post) on normalized text.
    norm = df[config.columns.text].map(normalize_text)
    pre = norm.map(lambda t: audit_text_for_leakage(t, config.leakage_blacklist))
    pre_n = pre.map(lambda r: r.n_matches)

    redacted = norm.map(
        lambda t: normalize_whitespace(remove_blacklisted_terms(t, config.leakage_blacklist, replacement=" "))
    )
    post = redacted.map(lambda t: audit_text_for_leakage(t, config.leakage_blacklist))
    post_n = post.map(lambda r: r.n_matches)

    results_dir = ensure_dir(resolve_path(project_root, config.paths.results_dir))
    logs_dir = ensure_dir(results_dir / "logs")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    pre_report = pd.DataFrame(
        {
            "row": np.arange(len(df)),
            "temporal_stage": df[config.columns.stage].astype(str),
            "subtype": df[config.columns.subtype].astype(str),
            "n_matches": pre_n,
        }
    )
    post_report = pd.DataFrame(
        {
            "row": np.arange(len(df)),
            "temporal_stage": df[config.columns.stage].astype(str),
            "subtype": df[config.columns.subtype].astype(str),
            "n_matches": post_n,
        }
    )

    pre_path = logs_dir / f"leakage_pre_audit_{run_id}.csv"
    post_path = logs_dir / f"leakage_post_audit_{run_id}.csv"
    pre_report.to_csv(pre_path, index=False)
    post_report.to_csv(post_path, index=False)

    if int(post_n.sum()) != 0:
        raise RuntimeError(
            f"Post-redaction leakage is non-zero (should be 0). See: {post_path}"
        )

    df.to_csv(out_path, index=False)

    meta = {
        "created_utc": run_id,
        "input_csv": str(input_path),
        "output_csv": str(out_path),
        "dropped_columns": dropped,
        "rows": int(len(df)),
        "stage_counts": df[config.columns.stage].value_counts().to_dict(),
        "pre_leakage_total_matches": int(pre_n.sum()),
        "post_leakage_total_matches": int(post_n.sum()),
        "pre_leakage_report_csv": str(pre_path),
        "post_leakage_report_csv": str(post_path),
    }
    (logs_dir / f"make_curated_{run_id}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return CuratedBuildOutputs(
        curated_csv=out_path, pre_leakage_report_csv=pre_path, post_leakage_report_csv=post_path
    )

