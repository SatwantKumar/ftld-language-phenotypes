from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from paperjn.stats.enrichment import TwoByTwo, compute_effect_sizes
from paperjn.stats.permutation import permutation_p_value_ge
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
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return None
        if float(value) in (0.0, 1.0):
            return int(float(value))
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    if s in {"0", "1"}:
        return int(s)
    return None


def _language_cluster(labels: np.ndarray, is_ppa: np.ndarray, *, k: int) -> tuple[int, float]:
    shares = []
    for cl in range(int(k)):
        mask = labels == cl
        shares.append(float(is_ppa[mask].mean()) if np.any(mask) else 0.0)
    max_share = float(np.max(shares))
    cl_id = int(min(i for i, s in enumerate(shares) if float(s) == max_share))
    return cl_id, max_share


def _selection_aware_max_share_p(
    *, labels: np.ndarray, is_ppa: np.ndarray, k: int, permutations: int, random_seed: int, observed_max: float
) -> float:
    rng = np.random.default_rng(int(random_seed))
    cluster_idx = [np.where(labels == cl)[0] for cl in range(int(k))]
    perm_stats = np.zeros(int(permutations), dtype=np.float32)
    for b in range(int(permutations)):
        perm_is = rng.permutation(is_ppa)
        shares = []
        for idx in cluster_idx:
            shares.append(float(np.mean(perm_is[idx])) if idx.size else 0.0)
        perm_stats[b] = float(np.max(shares))
    return permutation_p_value_ge(float(observed_max), perm_stats)


def _fixed_cluster_share_p(
    *, target_mask: np.ndarray, is_ppa: np.ndarray, permutations: int, random_seed: int, observed: float
) -> float:
    rng = np.random.default_rng(int(random_seed))
    perm_stats = np.zeros(int(permutations), dtype=np.float32)
    for b in range(int(permutations)):
        perm_is = rng.permutation(is_ppa)
        perm_stats[b] = float(np.mean(perm_is[target_mask])) if target_mask.any() else 0.0
    return permutation_p_value_ge(float(observed), perm_stats)


def _enrichment_table(*, labels: np.ndarray, is_ppa: np.ndarray, target_cluster: int) -> TwoByTwo:
    in_cluster = labels == int(target_cluster)
    n_total = int(len(labels))
    n_cluster = int(in_cluster.sum())
    ppa_total = int(is_ppa.sum())
    ppa_cluster = int(is_ppa[in_cluster].sum())
    a = float(ppa_cluster)
    b = float(n_cluster - ppa_cluster)
    c = float(ppa_total - ppa_cluster)
    d = float((n_total - n_cluster) - (ppa_total - ppa_cluster))
    return TwoByTwo(a=a, b=b, c=c, d=d)


@dataclass(frozen=True)
class PhaseDRelabelOutputs:
    run_dir: Path
    run_config_json: Path
    report_md: Path
    case_table_csv: Path
    replication_summary_csv: Path


def run_phase_d_relabel_with_rater_overrides(
    *,
    base_phase_d_run_dir: Path,
    rater_csv: Path,
    out_dir: Path | None,
    exclude_non_ftld: bool,
    rater_source: str,
) -> PhaseDRelabelOutputs:
    """Create a Phase D-like run directory with `is_ppa` overridden for a subset of cases.

    Intended for human or in-silico rater validation sensitivity runs: cluster assignments remain fixed;
    only endpoint labels are updated.
    """
    base_phase_d_run_dir = Path(base_phase_d_run_dir).resolve()
    rater_csv = Path(rater_csv).resolve()
    if not base_phase_d_run_dir.exists():
        raise FileNotFoundError(f"Base Phase D run dir not found: {base_phase_d_run_dir}")
    if not rater_csv.exists():
        raise FileNotFoundError(f"Rater CSV not found: {rater_csv}")

    base_tables = base_phase_d_run_dir / "tables"
    base_case_table = base_tables / "case_table_with_clusters.csv"
    base_rep = base_tables / "replication_summary.csv"
    base_synd = base_tables / "syndrome_replication_summary.csv"
    if not base_case_table.exists():
        raise FileNotFoundError(f"Missing base case table: {base_case_table}")
    if not base_rep.exists():
        raise FileNotFoundError(f"Missing base replication summary: {base_rep}")
    if not base_synd.exists():
        raise FileNotFoundError(f"Missing base syndrome replication summary: {base_synd}")

    base_cfg_path = base_phase_d_run_dir / "run_config.json"
    if not base_cfg_path.exists():
        raise FileNotFoundError(f"Missing base run_config.json: {base_cfg_path}")
    base_cfg = json.loads(base_cfg_path.read_text(encoding="utf-8"))
    k_fixed = int(base_cfg.get("k_fixed") or 4)
    permutations = int(base_cfg.get("permutations") or 5000)
    random_seed = int(base_cfg.get("random_seed") or 42)
    pca_fit_on = str(base_cfg.get("pca_fit_on") or "discovery")
    matching = base_cfg.get("matching", {}).get("discovery_to_confirmation", {})
    disc_to_conf = {int(k): int(v) for k, v in matching.items()} if isinstance(matching, dict) else {}

    ts = _utc_slug()
    if out_dir is None:
        out_dir = (base_phase_d_run_dir.parent / f"relabel__{rater_source}__{ts}").resolve()
    run_dir = ensure_dir(Path(out_dir).resolve())
    tables_dir = ensure_dir(run_dir / "tables")

    # Copy stable tables for audit (and to satisfy downstream packet tooling).
    for name in [
        "cluster_similarity_matrix.csv",
        "cluster_summary_discovery.csv",
        "cluster_summary_confirmation.csv",
        "syndrome_replication_summary.csv",
        "cluster_characterization_by_matched_cluster.csv",
        "numeric_characterization_by_matched_cluster.csv",
        "syndrome_composition_by_matched_cluster.csv",
        "flow_table.csv",
        "symptom_tags_by_cluster_discovery.csv",
        "symptom_tags_by_cluster_confirmation.csv",
        "symptom_tags_by_cluster_matched.csv",
    ]:
        src = base_tables / name
        if src.exists():
            (tables_dir / name).write_bytes(src.read_bytes())

    df = pd.read_csv(base_case_table)
    if "status" in df.columns:
        df = df[df["status"].astype(str) == "ok"].copy()
    df["pmcid"] = df["pmcid"].astype(str)
    df["case_id"] = df["case_id"].astype(str)
    df["split"] = df["split"].astype(str)
    df["cluster_split"] = pd.to_numeric(df["cluster_split"], errors="coerce").astype(int)
    df["cluster_matched_to_discovery"] = pd.to_numeric(df["cluster_matched_to_discovery"], errors="coerce").astype(int)

    # Preserve original endpoint for traceability.
    if "is_ppa_llm" not in df.columns:
        df["is_ppa_llm"] = df["is_ppa"].astype(bool) if "is_ppa" in df.columns else False
    df["is_ppa_source"] = "llm_phase_c"

    rat = pd.read_csv(rater_csv)
    for col in ["pmcid", "case_id"]:
        if col not in rat.columns:
            raise ValueError(f"Rater CSV missing required column: {col}")
    rat["pmcid"] = rat["pmcid"].astype(str)
    rat["case_id"] = rat["case_id"].astype(str)

    # Apply overrides only when text is adequate.
    # Conservative: if text_adequate != 1, do not override.
    override_rows = rat.copy()
    override_rows["text_adequate_bin"] = override_rows.get("text_adequate", np.nan).map(_as_int01)
    override_rows = override_rows[override_rows["text_adequate_bin"] == 1].copy()

    override_rows["is_ppa_bin"] = override_rows.get("is_ppa", np.nan).map(_as_int01)
    override_rows["is_ftld_bin"] = override_rows.get("is_ftld_spectrum", np.nan).map(_as_int01)

    merged = df.merge(
        override_rows[["pmcid", "case_id", "is_ppa_bin", "is_ftld_bin"]],
        on=["pmcid", "case_id"],
        how="left",
    )

    n_match = int(merged["is_ppa_bin"].notna().sum())
    n_excluded = 0

    if exclude_non_ftld:
        # Drop cases explicitly rated as non-FTLD.
        mask_ex = merged["is_ftld_bin"] == 0
        n_excluded = int(mask_ex.sum())
        merged = merged[~mask_ex].copy()

    # Override is_ppa when provided.
    is_ppa_new = merged["is_ppa"].astype(bool) if "is_ppa" in merged.columns else merged["is_ppa_llm"].astype(bool)
    has_override = merged["is_ppa_bin"].notna()
    override_bool = merged["is_ppa_bin"].fillna(0).astype(int).astype(bool)
    is_ppa_new = is_ppa_new.where(~has_override, override_bool)
    merged["is_ppa"] = is_ppa_new.astype(bool)
    merged.loc[has_override, "is_ppa_source"] = str(rater_source)

    # Recompute primary replication summary using fixed cluster assignments.
    disc_mask = merged["split"] == "discovery"
    conf_mask = merged["split"] == "confirmation"
    disc = merged[disc_mask].copy()
    conf = merged[conf_mask].copy()
    if disc.empty or conf.empty:
        raise RuntimeError("Relabel run has empty discovery or confirmation set after filtering.")

    disc_labels = disc["cluster_split"].to_numpy(dtype=int)
    conf_labels = conf["cluster_split"].to_numpy(dtype=int)
    y_disc = disc["is_ppa"].to_numpy(dtype=bool)
    y_conf = conf["is_ppa"].to_numpy(dtype=bool)

    lang_disc, disc_max = _language_cluster(disc_labels, y_disc, k=k_fixed)
    disc_p = _selection_aware_max_share_p(
        labels=disc_labels,
        is_ppa=y_disc.astype(int),
        k=k_fixed,
        permutations=permutations,
        random_seed=random_seed,
        observed_max=disc_max,
    )
    disc_eff = compute_effect_sizes(_enrichment_table(labels=disc_labels, is_ppa=y_disc.astype(int), target_cluster=lang_disc))

    # Determine matched confirmation cluster (disc -> conf mapping).
    if not disc_to_conf:
        # Derive from case table mapping if run_config missing.
        mapping_rows = conf[["cluster_split", "cluster_matched_to_discovery"]].drop_duplicates()
        conf_to_disc = {
            int(r["cluster_split"]): int(r["cluster_matched_to_discovery"])
            for r in mapping_rows.to_dict(orient="records")
        }
        disc_to_conf = {v: k for k, v in conf_to_disc.items()}
    if int(lang_disc) not in disc_to_conf:
        raise RuntimeError(f"Missing discovery->confirmation mapping for cluster {lang_disc}.")
    lang_conf = int(disc_to_conf[int(lang_disc)])

    conf_mask_cluster = conf_labels == int(lang_conf)
    conf_share = float(y_conf[conf_mask_cluster].mean()) if conf_mask_cluster.any() else 0.0
    conf_p = _fixed_cluster_share_p(
        target_mask=conf_mask_cluster,
        is_ppa=y_conf.astype(int),
        permutations=permutations,
        random_seed=random_seed,
        observed=conf_share,
    )
    conf_eff = compute_effect_sizes(_enrichment_table(labels=conf_labels, is_ppa=y_conf.astype(int), target_cluster=lang_conf))

    rep_row = {
        "k_fixed": int(k_fixed),
        "pca_fit_on": str(pca_fit_on),
        "n_discovery": int(len(disc)),
        "n_confirmation": int(len(conf)),
        "language_cluster_discovery": int(lang_disc),
        "language_cluster_confirmation_matched": int(lang_conf),
        "discovery_max_ppa_share": float(disc_max),
        "discovery_perm_p_value_selection_aware": float(disc_p),
        "confirmation_ppa_share_matched_cluster": float(conf_share),
        "confirmation_perm_p_value_fixed_cluster": float(conf_p),
        "discovery_rr": float(disc_eff.rr),
        "discovery_rr_ci95_low": float(disc_eff.rr_ci95_low),
        "discovery_rr_ci95_high": float(disc_eff.rr_ci95_high),
        "discovery_or": float(disc_eff.or_),
        "discovery_or_ci95_low": float(disc_eff.or_ci95_low),
        "discovery_or_ci95_high": float(disc_eff.or_ci95_high),
        "confirmation_rr": float(conf_eff.rr),
        "confirmation_rr_ci95_low": float(conf_eff.rr_ci95_low),
        "confirmation_rr_ci95_high": float(conf_eff.rr_ci95_high),
        "confirmation_or": float(conf_eff.or_),
        "confirmation_or_ci95_low": float(conf_eff.or_ci95_low),
        "confirmation_or_ci95_high": float(conf_eff.or_ci95_high),
        "permutations_B": int(permutations),
    }
    replication_summary_csv = tables_dir / "replication_summary.csv"
    pd.DataFrame([rep_row]).to_csv(replication_summary_csv, index=False)

    # Write updated case table.
    case_table_csv = tables_dir / "case_table_with_clusters.csv"
    merged.to_csv(case_table_csv, index=False)

    # Run config + report
    run_config_json = run_dir / "run_config.json"
    out_cfg = dict(base_cfg)
    out_cfg["run_id"] = f"{ts}__relabel__{rater_source}"
    out_cfg["created_utc"] = ts
    out_cfg["base_phase_d_run_dir"] = str(base_phase_d_run_dir)
    out_cfg["rater_overrides_csv"] = str(rater_csv)
    out_cfg["rater_source"] = str(rater_source)
    out_cfg["exclude_non_ftld"] = bool(exclude_non_ftld)
    out_cfg["override_counts"] = {"n_is_ppa_overrides": int(n_match), "n_excluded_non_ftld": int(n_excluded)}
    run_config_json.write_text(json.dumps(out_cfg, indent=2), encoding="utf-8")

    report_md = run_dir / "relabel_report.md"
    lines: list[str] = []
    lines.append("# Phase D Relabel Sensitivity Run")
    lines.append("")
    lines.append(f"- Base Phase D run: `{base_phase_d_run_dir}`")
    lines.append(f"- Rater overrides: `{rater_csv}`")
    lines.append(f"- Rater source: `{rater_source}`")
    lines.append(f"- Exclude non-FTLD (rater): {bool(exclude_non_ftld)} (excluded n={int(n_excluded)})")
    lines.append(f"- is_ppa overrides applied: n={int(n_match)}")
    lines.append("")
    lines.append("## Updated primary replication summary")
    lines.append(pd.DataFrame([rep_row]).to_markdown(index=False))
    lines.append("")
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return PhaseDRelabelOutputs(
        run_dir=run_dir,
        run_config_json=run_config_json,
        report_md=report_md,
        case_table_csv=case_table_csv,
        replication_summary_csv=replication_summary_csv,
    )
