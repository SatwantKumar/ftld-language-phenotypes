from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paperjn.stats.enrichment import TwoByTwo, compute_effect_sizes
from paperjn.utils.paths import ensure_dir


PPA_SYNDROMES = {"svPPA", "nfvPPA", "lvPPA", "PPA_unspecified"}

IMAGING_MODALITIES_V1 = ["mri", "ct", "fdg_pet", "spect", "other"]
IMAGING_REGIONS_V1 = [
    "frontal",
    "temporal",
    "parietal",
    "insula",
    "cingulate",
    "basal_ganglia",
    "brainstem",
    "cerebellum",
    "other",
    "unknown",
]
IMAGING_LATERALITY_V1 = ["left", "right", "bilateral", "diffuse", "unknown", "missing"]
GENETICS_STATUS_V1 = ["confirmed_pathogenic", "reported_uncertain", "tested_negative", "not_reported", "unclear", "missing"]
NEUROPATH_STATUS_V1 = ["confirmed", "not_reported", "unclear", "missing"]
PATHOLOGY_TYPES_V1 = ["tau", "tdp43", "fus", "mixed", "other", "unknown"]
KEY_GENES_V1 = ["MAPT", "GRN", "C9ORF72"]
INITIAL_DX_V1 = [
    "ftld",
    "psychiatric",
    "ad",
    "vascular_stroke",
    "other_neuro",
    "other",
    "not_reported",
    "unknown",
    "missing",
]
MISDX_V1 = ["yes", "no", "not_reported", "unknown", "missing"]


def _utc_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _find_results_root(path: Path) -> Path | None:
    """Return the nearest ancestor named `results` (project results root), else None."""
    for p in [path] + list(path.parents):
        if p.name == "results":
            return p
    return None


def _wilson_ci(k: int, n: int, *, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = float(k) / float(n)
    denom = 1.0 + (z * z) / float(n)
    center = (p + (z * z) / (2.0 * float(n))) / denom
    half = (z * np.sqrt((p * (1.0 - p)) / float(n) + (z * z) / (4.0 * float(n * n)))) / denom
    lo = max(0.0, float(center - half))
    hi = min(1.0, float(center + half))
    return lo, hi


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


def _direction(delta: float, *, eps: float) -> str:
    if not np.isfinite(delta):
        return "unknown"
    if delta > eps:
        return "enriched"
    if delta < -eps:
        return "depleted"
    return "neutral"


def _effect_sizes_for_binary(*, y: np.ndarray, in_cluster: np.ndarray) -> tuple[dict[str, float], dict[str, int]]:
    y = y.astype(bool)
    in_cluster = in_cluster.astype(bool)
    n_total = int(len(y))
    n_cluster = int(in_cluster.sum())
    pos_total = int(y.sum())
    pos_cluster = int((y & in_cluster).sum())
    a = pos_cluster
    b = n_cluster - pos_cluster
    c = pos_total - pos_cluster
    d = (n_total - n_cluster) - c
    eff = compute_effect_sizes(TwoByTwo(a=float(a), b=float(b), c=float(c), d=float(d)))
    return (
        {
            "share_cluster": float(pos_cluster / n_cluster) if n_cluster else float("nan"),
            "share_total": float(pos_total / n_total) if n_total else float("nan"),
            "rr": float(eff.rr),
            "rr_ci95_low": float(eff.rr_ci95_low),
            "rr_ci95_high": float(eff.rr_ci95_high),
            "or": float(eff.or_),
            "or_ci95_low": float(eff.or_ci95_low),
            "or_ci95_high": float(eff.or_ci95_high),
        },
        {
            "n_total": n_total,
            "n_cluster": n_cluster,
            "pos_total": pos_total,
            "pos_cluster": pos_cluster,
        },
    )


@dataclass(frozen=True)
class PhaseEOutputs:
    run_dir: Path
    run_config_json: Path
    report_md: Path
    results_packet_md: Path
    primary_precision_csv: Path
    syndrome_composition_csv: Path
    symptom_tags_csv: Path
    optional_fields_csv: Path
    secondary_precision_csv: Path
    top_syndromes_csv: Path
    top_tags_csv: Path
    fig_ppa_share_png: Path


def run_phase_e_pool_after_replication(
    *,
    phase_d_run_dir: Path,
    out_dir: Path | None = None,
    alpha: float = 0.05,
    direction_eps: float = 0.01,
    max_top_syndromes: int = 5,
    max_top_tags: int = 5,
    force: bool = False,
) -> PhaseEOutputs:
    """Phase E: post-replication pooled characterization (no new confirmatory p-values)."""
    phase_d_run_dir = Path(phase_d_run_dir).resolve()
    tables_in = phase_d_run_dir / "tables"
    case_table_path = tables_in / "case_table_with_clusters.csv"
    rep_path = tables_in / "replication_summary.csv"
    synd_rep_path = tables_in / "syndrome_replication_summary.csv"
    if not case_table_path.exists():
        raise FileNotFoundError(f"Missing Phase D case table: {case_table_path}")
    if not rep_path.exists():
        raise FileNotFoundError(f"Missing Phase D replication summary: {rep_path}")
    if not synd_rep_path.exists():
        raise FileNotFoundError(f"Missing Phase D syndrome replication summary: {synd_rep_path}")

    df = pd.read_csv(case_table_path)
    if "status" in df.columns:
        df = df[df["status"].astype(str) == "ok"].copy()
    df["split"] = df["split"].astype(str)
    if "cluster_matched_to_discovery" not in df.columns:
        raise RuntimeError("Phase D case table missing cluster_matched_to_discovery.")
    df["matched_cluster"] = pd.to_numeric(df["cluster_matched_to_discovery"], errors="coerce").astype(int)
    if "is_ppa" not in df.columns:
        # Backstop: infer from syndrome label.
        df["is_ppa"] = df["ftld_syndrome_reported"].astype(str).isin(PPA_SYNDROMES)
    df["is_ppa"] = df["is_ppa"].astype(bool)

    rep = pd.read_csv(rep_path).iloc[0].to_dict()
    k_fixed = int(rep["k_fixed"])
    lang_cluster = int(rep["language_cluster_discovery"])
    p_conf = float(rep["confirmation_perm_p_value_fixed_cluster"])

    # Gate pooling on replication
    conf = df[df["split"] == "confirmation"].copy()
    if conf.empty:
        raise RuntimeError("No confirmation rows found.")
    conf_baseline = float(conf["is_ppa"].mean())
    conf_in_lang = conf["matched_cluster"].astype(int) == int(lang_cluster)
    conf_share_lang = float(conf.loc[conf_in_lang, "is_ppa"].mean()) if conf_in_lang.any() else float("nan")
    direction_ok = bool(conf_share_lang > conf_baseline)
    replicate_ok = bool(np.isfinite(p_conf) and p_conf < 0.05 and direction_ok)
    if not replicate_ok and not force:
        raise RuntimeError(
            "Phase E gate failed (replication not achieved): "
            f"confirmation p={p_conf:.4g}, share_lang={conf_share_lang:.3f}, baseline={conf_baseline:.3f}. "
            "Use --force to run anyway (not reviewer-proof)."
        )

    ts = _utc_slug()
    if out_dir is None:
        results_root = _find_results_root(phase_d_run_dir)
        if results_root is None:
            # Fallback: sibling of the Phase D run (still deterministic, but less standardized).
            out_dir = (phase_d_run_dir.parent / f"phase_e__{ts}").resolve()
        else:
            out_dir = (results_root / "phase_e" / f"pool_after_replication__{ts}").resolve()
    run_dir = ensure_dir(Path(out_dir).resolve())
    tables_out = ensure_dir(run_dir / "tables")
    figures_out = ensure_dir(run_dir / "figures")

    # Primary precision table (split-wise + pooled) for the replicated language cluster
    primary_rows: list[dict[str, object]] = []
    for split_name, sdf in [("discovery", df[df["split"] == "discovery"]), ("confirmation", conf), ("pooled", df)]:
        in_cluster = sdf["matched_cluster"].astype(int).to_numpy() == int(lang_cluster)
        y = sdf["is_ppa"].astype(bool).to_numpy()
        eff, counts = _effect_sizes_for_binary(y=y, in_cluster=in_cluster)
        lo, hi = _wilson_ci(int(counts["pos_cluster"]), int(counts["n_cluster"]))
        base_lo, base_hi = _wilson_ci(int(counts["pos_total"]), int(counts["n_total"]))
        primary_rows.append(
            {
                "endpoint": "PPA",
                "split": split_name,
                "k_fixed": k_fixed,
                "target_matched_cluster": int(lang_cluster),
                "n_total": counts["n_total"],
                "n_cluster": counts["n_cluster"],
                "pos_total": counts["pos_total"],
                "pos_cluster": counts["pos_cluster"],
                "share_cluster": eff["share_cluster"],
                "share_cluster_ci_low": lo,
                "share_cluster_ci_high": hi,
                "share_split": eff["share_total"],
                "share_split_ci_low": base_lo,
                "share_split_ci_high": base_hi,
                "rr": eff["rr"],
                "rr_ci95_low": eff["rr_ci95_low"],
                "rr_ci95_high": eff["rr_ci95_high"],
                "or": eff["or"],
                "or_ci95_low": eff["or_ci95_low"],
                "or_ci95_high": eff["or_ci95_high"],
                "note": "Pooled is post-replication precision only; no pooled p-values.",
            }
        )
    primary_precision_csv = tables_out / "primary_precision_split_vs_pooled.csv"
    pd.DataFrame(primary_rows).to_csv(primary_precision_csv, index=False)

    # Syndrome composition (split-wise + pooled)
    synd_rows: list[dict[str, object]] = []
    for split_name, sdf in [("discovery", df[df["split"] == "discovery"]), ("confirmation", conf), ("pooled", df)]:
        for cl in range(k_fixed):
            cdf = sdf[sdf["matched_cluster"] == int(cl)]
            n_cluster = int(len(cdf))
            if n_cluster == 0:
                continue
            vc = cdf["ftld_syndrome_reported"].astype(str).value_counts(dropna=False)
            for syndrome, n_cases in vc.items():
                n_cases_int = int(n_cases)
                prop = float(n_cases_int / n_cluster) if n_cluster else float("nan")
                lo, hi = _wilson_ci(n_cases_int, n_cluster)
                synd_rows.append(
                    {
                        "split": split_name,
                        "matched_cluster": int(cl),
                        "ftld_syndrome_reported": str(syndrome),
                        "n_cases": n_cases_int,
                        "cluster_n": n_cluster,
                        "prop_within_cluster": prop,
                        "prop_ci_low": lo,
                        "prop_ci_high": hi,
                    }
                )
    syndrome_composition_csv = tables_out / "syndrome_composition_by_matched_cluster_split_vs_pooled.csv"
    pd.DataFrame(synd_rows).to_csv(syndrome_composition_csv, index=False)

    pooled_synd = pd.DataFrame(synd_rows)
    pooled_synd = pooled_synd[pooled_synd["split"] == "pooled"].copy()
    top_synd_rows = []
    for cl in range(k_fixed):
        sub = pooled_synd[pooled_synd["matched_cluster"] == int(cl)].sort_values(
            ["prop_within_cluster", "n_cases"], ascending=[False, False]
        )
        top = sub.head(int(max_top_syndromes))
        for _, r in top.iterrows():
            top_synd_rows.append(
                {
                    "matched_cluster": int(cl),
                    "ftld_syndrome_reported": str(r["ftld_syndrome_reported"]),
                    "prop_pooled": float(r["prop_within_cluster"]),
                    "n_cases_pooled": int(r["n_cases"]),
                    "cluster_n_pooled": int(r["cluster_n"]),
                }
            )
    top_syndromes_csv = tables_out / "cluster_profile_top_syndromes_pooled.csv"
    pd.DataFrame(top_synd_rows).to_csv(top_syndromes_csv, index=False)

    # Symptom tags (split-wise + pooled)
    tag_cols = [c for c in df.columns if c.startswith("tag__") and c.endswith("__status")]
    if not tag_cols:
        raise RuntimeError("No symptom tag status columns found in Phase D case table.")
    tag_rows: list[dict[str, object]] = []
    for split_name, sdf in [("discovery", df[df["split"] == "discovery"]), ("confirmation", conf), ("pooled", df)]:
        for cl in range(k_fixed):
            cdf = sdf[sdf["matched_cluster"] == int(cl)]
            n_cluster = int(len(cdf))
            if n_cluster == 0:
                continue
            for tag_col in tag_cols:
                tag = tag_col.split("__", 2)[1]
                status = cdf[tag_col].astype(str)
                present_n = int((status == "present").sum())
                prop = float(present_n / n_cluster) if n_cluster else float("nan")
                lo, hi = _wilson_ci(present_n, n_cluster)
                tag_rows.append(
                    {
                        "split": split_name,
                        "matched_cluster": int(cl),
                        "tag": str(tag),
                        "n_cases": n_cluster,
                        "present_n": present_n,
                        "present_prop": prop,
                        "present_ci_low": lo,
                        "present_ci_high": hi,
                        "explicitly_absent_n": int((status == "explicitly_absent").sum()),
                        "uncertain_n": int((status == "uncertain").sum()),
                        "not_reported_n": int((status == "not_reported").sum()),
                    }
                )
    symptom_tags_csv = tables_out / "symptom_tags_by_matched_cluster_split_vs_pooled.csv"
    pd.DataFrame(tag_rows).to_csv(symptom_tags_csv, index=False)

    pooled_tags = pd.DataFrame(tag_rows)
    pooled_tags = pooled_tags[pooled_tags["split"] == "pooled"].copy()
    top_tag_rows = []
    for cl in range(k_fixed):
        sub = pooled_tags[pooled_tags["matched_cluster"] == int(cl)].sort_values(
            ["present_prop", "present_n"], ascending=[False, False]
        )
        top = sub.head(int(max_top_tags))
        for _, r in top.iterrows():
            top_tag_rows.append(
                {
                    "matched_cluster": int(cl),
                    "tag": str(r["tag"]),
                    "present_prop_pooled": float(r["present_prop"]),
                    "present_n_pooled": int(r["present_n"]),
                    "cluster_n_pooled": int(r["n_cases"]),
                }
            )
    top_tags_csv = tables_out / "cluster_profile_top_tags_pooled.csv"
    pd.DataFrame(top_tag_rows).to_csv(top_tags_csv, index=False)

    # Optional fields prevalence (imaging/genetics/pathology + initial dx/misdx), split-wise + pooled
    modality_sets = [set(x.lower() for x in _parse_json_list(v)) for v in df.get("imaging_modalities_json", []).tolist()]
    region_sets = [set(x.lower() for x in _parse_json_list(v)) for v in df.get("imaging_regions_json", []).tolist()]
    pathology_sets = [set(x.lower() for x in _parse_json_list(v)) for v in df.get("pathology_types_json", []).tolist()]
    gene_sets = []
    for v in df.get("genes_reported_json", pd.Series([None] * len(df))).tolist():
        genes = set()
        for g in _parse_json_list(v):
            gg = str(g).strip().upper()
            if not gg:
                continue
            if gg.replace("-", "").replace("_", "") in {"C9ORF72", "C9ORF"}:
                gg = "C9ORF72"
            genes.add(gg)
        gene_sets.append(genes)

    lat = df.get("imaging_laterality", pd.Series([None] * len(df))).astype("object").tolist()
    genetics_status = df.get("genetics_status", pd.Series([None] * len(df))).astype("object").tolist()
    neuropath_status = df.get("neuropath_status", pd.Series([None] * len(df))).astype("object").tolist()
    initial_dx = df.get("initial_dx_category", pd.Series([None] * len(df))).astype("object").tolist()
    misdx = df.get("misdiagnosed_prior_to_ftld", pd.Series([None] * len(df))).astype("object").tolist()

    def _cat(x: object) -> str:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "missing"
        s = str(x).strip()
        return s if s else "missing"

    lat_cat = [(_cat(x)).lower() for x in lat]
    genetics_cat = [(_cat(x)).lower() for x in genetics_status]
    neuropath_cat = [(_cat(x)).lower() for x in neuropath_status]
    initial_dx_cat = [(_cat(x)).lower() for x in initial_dx]
    misdx_cat = [(_cat(x)).lower() for x in misdx]

    indicators: list[tuple[str, str, list[bool]]] = []
    for mod in IMAGING_MODALITIES_V1:
        indicators.append(("imaging_modality", mod, [mod in s for s in modality_sets]))
    for reg in IMAGING_REGIONS_V1:
        indicators.append(("imaging_region", reg, [reg in s for s in region_sets]))
    for val in IMAGING_LATERALITY_V1:
        indicators.append(("imaging_laterality", val, [x == val for x in lat_cat]))
    for val in GENETICS_STATUS_V1:
        indicators.append(("genetics_status", val, [x == val for x in genetics_cat]))
    for val in NEUROPATH_STATUS_V1:
        indicators.append(("neuropath_status", val, [x == val for x in neuropath_cat]))
    for pt in PATHOLOGY_TYPES_V1:
        indicators.append(("pathology_type", pt, [pt in s for s in pathology_sets]))
    for gene in KEY_GENES_V1:
        indicators.append(("gene_reported", gene, [gene in s for s in gene_sets]))
    for val in INITIAL_DX_V1:
        indicators.append(("initial_dx_category", val, [x == val for x in initial_dx_cat]))
    for val in MISDX_V1:
        indicators.append(("misdiagnosed_prior_to_ftld", val, [x == val for x in misdx_cat]))

    opt_rows: list[dict[str, object]] = []
    df_idx = np.arange(len(df), dtype=int)
    split_arr = df["split"].astype(str).to_numpy()
    cluster_arr = df["matched_cluster"].astype(int).to_numpy()

    for family, value, has_list in indicators:
        has = np.asarray(has_list, dtype=bool)
        if has.shape[0] != len(df_idx):
            raise RuntimeError("Internal error: indicator length mismatch.")
        for split_name in ["discovery", "confirmation", "pooled"]:
            if split_name == "pooled":
                split_mask = np.ones(len(df), dtype=bool)
            else:
                split_mask = split_arr == split_name
            n_split = int(split_mask.sum())
            if n_split == 0:
                continue
            split_has_n = int(has[split_mask].sum())
            split_prop = float(split_has_n / n_split) if n_split else float("nan")
            split_lo, split_hi = _wilson_ci(split_has_n, n_split)
            for cl in range(k_fixed):
                mask = split_mask & (cluster_arr == int(cl))
                n_cluster = int(mask.sum())
                if n_cluster == 0:
                    continue
                value_n = int(has[mask].sum())
                prop = float(value_n / n_cluster) if n_cluster else float("nan")
                lo, hi = _wilson_ci(value_n, n_cluster)
                delta = prop - split_prop
                opt_rows.append(
                    {
                        "split": split_name,
                        "matched_cluster": int(cl),
                        "family": family,
                        "value": value,
                        "n_cases": n_cluster,
                        "value_n": value_n,
                        "value_prop": prop,
                        "value_ci_low": lo,
                        "value_ci_high": hi,
                        "split_n": n_split,
                        "value_n_split": split_has_n,
                        "value_prop_split": split_prop,
                        "value_prop_split_ci_low": split_lo,
                        "value_prop_split_ci_high": split_hi,
                        "direction_vs_split": _direction(delta, eps=float(direction_eps)),
                    }
                )

    opt_df = pd.DataFrame(opt_rows)
    disc_dir = opt_df[opt_df["split"] == "discovery"][
        ["matched_cluster", "family", "value", "direction_vs_split"]
    ].rename(columns={"direction_vs_split": "direction_discovery"})
    conf_dir = opt_df[opt_df["split"] == "confirmation"][
        ["matched_cluster", "family", "value", "direction_vs_split"]
    ].rename(columns={"direction_vs_split": "direction_confirmation"})
    opt_df = opt_df.merge(disc_dir, on=["matched_cluster", "family", "value"], how="left")
    opt_df = opt_df.merge(conf_dir, on=["matched_cluster", "family", "value"], how="left")
    opt_df["direction_consistent"] = opt_df["direction_discovery"] == opt_df["direction_confirmation"]
    optional_fields_csv = tables_out / "optional_fields_by_matched_cluster_split_vs_pooled.csv"
    opt_df.sort_values(["family", "value", "split", "matched_cluster"]).to_csv(optional_fields_csv, index=False)

    # Secondary precision: only endpoints that pass BH-FDR in Confirmation (post-replication precision estimates)
    synd_rep = pd.read_csv(synd_rep_path)
    synd_rep["confirmation_reject_bh_fdr"] = synd_rep["confirmation_reject_bh_fdr"].astype(bool)
    keep_endpoints = synd_rep[synd_rep["confirmation_reject_bh_fdr"]].copy()
    secondary_rows: list[dict[str, object]] = []
    for row in keep_endpoints.itertuples(index=False):
        endpoint = str(getattr(row, "endpoint"))
        target_cluster = int(getattr(row, "discovery_cluster_max_share"))
        for split_name, sdf in [("discovery", df[df["split"] == "discovery"]), ("confirmation", conf), ("pooled", df)]:
            in_cluster = sdf["matched_cluster"].astype(int).to_numpy() == int(target_cluster)
            y = sdf["ftld_syndrome_reported"].astype(str).to_numpy() == endpoint
            eff, counts = _effect_sizes_for_binary(y=y, in_cluster=in_cluster)
            lo, hi = _wilson_ci(int(counts["pos_cluster"]), int(counts["n_cluster"]))
            secondary_rows.append(
                {
                    "endpoint": endpoint,
                    "split": split_name,
                    "target_matched_cluster": int(target_cluster),
                    "n_total": counts["n_total"],
                    "n_cluster": counts["n_cluster"],
                    "pos_total": counts["pos_total"],
                    "pos_cluster": counts["pos_cluster"],
                    "share_cluster": eff["share_cluster"],
                    "share_cluster_ci_low": lo,
                    "share_cluster_ci_high": hi,
                    "share_split": eff["share_total"],
                    "rr": eff["rr"],
                    "rr_ci95_low": eff["rr_ci95_low"],
                    "rr_ci95_high": eff["rr_ci95_high"],
                    "or": eff["or"],
                    "or_ci95_low": eff["or_ci95_low"],
                    "or_ci95_high": eff["or_ci95_high"],
                    "confirmation_p": float(getattr(row, "confirmation_perm_p_value_fixed_cluster")),
                    "confirmation_q_bh_fdr": float(getattr(row, "confirmation_q_bh_fdr")),
                }
            )
    secondary_precision_csv = tables_out / "secondary_precision_split_vs_pooled.csv"
    pd.DataFrame(secondary_rows).to_csv(secondary_precision_csv, index=False)

    # Figure: PPA share by matched cluster (Discovery vs Confirmation vs Pooled)
    fig_rows = []
    for split_name, sdf in [("discovery", df[df["split"] == "discovery"]), ("confirmation", conf), ("pooled", df)]:
        for cl in range(k_fixed):
            cdf = sdf[sdf["matched_cluster"] == int(cl)]
            if cdf.empty:
                continue
            fig_rows.append(
                {
                    "split": split_name,
                    "matched_cluster": int(cl),
                    "n_cases": int(len(cdf)),
                    "ppa_share": float(cdf["is_ppa"].mean()),
                }
            )
    fig_df = pd.DataFrame(fig_rows)
    fig_ppa_share_png = figures_out / "fig_ppa_share_by_matched_cluster_split_vs_pooled.png"
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    x = np.arange(k_fixed)
    width = 0.25
    colors = {"discovery": "#4C78A8", "confirmation": "#F58518", "pooled": "#54A24B"}
    for i, split_name in enumerate(["discovery", "confirmation", "pooled"]):
        sub = fig_df[fig_df["split"] == split_name].sort_values("matched_cluster")
        y = sub["ppa_share"].to_numpy(dtype=float)
        ax.bar(x + (i - 1) * width, y, width=width, label=split_name.title(), color=colors[split_name])
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(k_fixed)])
    ax.set_xlabel("Matched cluster (discovery ID space)")
    ax.set_ylabel("PPA share")
    ax.set_title("PPA share by matched cluster (split vs pooled)")
    ax.axhline(conf_baseline, color="#999999", linestyle=":", linewidth=1, label="Confirmation baseline")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6)
    ax.legend(frameon=False, ncol=4, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(fig_ppa_share_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Report
    lines: list[str] = []
    lines.append("# Phase E (Post-Replication Pooling) Report")
    lines.append("")
    lines.append(f"Phase D input: `{phase_d_run_dir}`")
    lines.append(f"- Gate (replication): p_conf={p_conf:.6f}, share_lang={conf_share_lang:.3f}, baseline={conf_baseline:.3f}, pass={replicate_ok}")
    lines.append("- Pooling is descriptive/precision only; no new pooled confirmatory p-values.")
    lines.append("")
    lines.append("## Outputs")
    lines.append(f"- Primary precision table: `{primary_precision_csv}`")
    lines.append(f"- Syndrome composition: `{syndrome_composition_csv}`")
    lines.append(f"- Symptom tags: `{symptom_tags_csv}`")
    lines.append(f"- Optional fields: `{optional_fields_csv}`")
    lines.append(f"- Secondary precision (BH-FDR positive only): `{secondary_precision_csv}`")
    lines.append(f"- Figure: `{fig_ppa_share_png}`")
    lines.append("")
    report_md = run_dir / "phase_e_report.md"
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Packet markdown (manuscript-ready bundle index)
    packet_lines: list[str] = []
    packet_lines.append("# Phase E Results Packet")
    packet_lines.append("")
    packet_lines.append("Post-replication pooled characterization (estimation only; inference remains split-based).")
    packet_lines.append("")
    packet_lines.append(f"Phase D input: `{phase_d_run_dir}`")
    packet_lines.append("")
    packet_lines.append("## Key Files")
    for p in [
        primary_precision_csv,
        syndrome_composition_csv,
        symptom_tags_csv,
        optional_fields_csv,
        secondary_precision_csv,
        top_syndromes_csv,
        top_tags_csv,
        fig_ppa_share_png,
        report_md,
    ]:
        packet_lines.append(f"- `{p.relative_to(run_dir)}`")
    results_packet_md = run_dir / "results_packet.md"
    results_packet_md.write_text("\n".join(packet_lines) + "\n", encoding="utf-8")

    run_config = {
        "phase": "E",
        "created_utc": ts,
        "phase_d_run_dir": str(phase_d_run_dir),
        "gate": {
            "p_conf": float(p_conf),
            "conf_share_lang": float(conf_share_lang),
            "conf_baseline": float(conf_baseline),
            "pass": bool(replicate_ok),
            "force": bool(force),
        },
        "k_fixed": int(k_fixed),
        "language_cluster_discovery": int(lang_cluster),
        "alpha": float(alpha),
        "direction_eps": float(direction_eps),
        "max_top_syndromes": int(max_top_syndromes),
        "max_top_tags": int(max_top_tags),
        "notes": "Pooled analyses are post-replication descriptive/precision only; no pooled p-values.",
    }
    run_config_json = run_dir / "run_config.json"
    run_config_json.write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    return PhaseEOutputs(
        run_dir=run_dir,
        run_config_json=run_config_json,
        report_md=report_md,
        results_packet_md=results_packet_md,
        primary_precision_csv=primary_precision_csv,
        syndrome_composition_csv=syndrome_composition_csv,
        symptom_tags_csv=symptom_tags_csv,
        optional_fields_csv=optional_fields_csv,
        secondary_precision_csv=secondary_precision_csv,
        top_syndromes_csv=top_syndromes_csv,
        top_tags_csv=top_tags_csv,
        fig_ppa_share_png=fig_ppa_share_png,
    )
