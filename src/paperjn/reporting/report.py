from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class PaperRunSummary:
    paper_run_id: str
    primary_tag: str
    stage_results_path: Path
    combined_results_path: Path | None
    report_path: Path


def _fmt_float(x: float, ndigits: int = 3) -> str:
    return f"{x:.{ndigits}f}"


def _fmt_pct(x: float, ndigits: int = 1) -> str:
    return f"{100 * x:.{ndigits}f}%"


def write_paper_markdown_report(
    *,
    paper_run_id: str,
    primary_stage_results: pd.DataFrame,
    combined_results: pd.DataFrame | None,
    stage_order: list[str] | None = None,
    early_diagnostics: pd.DataFrame | None,
    early_centroid_summary: pd.DataFrame | None,
    bootstrap_stability: pd.DataFrame | None,
    seed_stability: pd.DataFrame | None,
    literature_summary: pd.DataFrame | None,
    literature_note: str | None = None,
    out_path: Path,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append(f"# paperJN run report ({paper_run_id})")
    lines.append("")
    lines.append("## Primary endpoint (combined PPA; selection-aware permutation + BH-FDR)")
    lines.append("")

    df = primary_stage_results.copy()
    if stage_order:
        df["Temporal_Stage"] = pd.Categorical(df["Temporal_Stage"], categories=stage_order, ordered=True)
    df = df.sort_values("Temporal_Stage")
    for _, r in df.iterrows():
        lines.append(
            f"- {r['Temporal_Stage']}: language cluster={int(r['language_cluster'])} "
            f"(n={int(r['n_language_cluster'])}/{int(r['n_stage'])}), "
            f"PPA share={_fmt_pct(float(r['ppa_share_language_cluster']))} vs baseline {_fmt_pct(float(r['ppa_share_stage']))}, "
            f"RR={_fmt_float(float(r['rr']))} (95% CI {_fmt_float(float(r['rr_ci95_low']))}–{_fmt_float(float(r['rr_ci95_high']))}), "
            f"p={_fmt_float(float(r['perm_p_value']), 4)}, q={_fmt_float(float(r['perm_q_value_bh']), 4)}"
        )

    if early_diagnostics is not None and not early_diagnostics.empty:
        lines.append("")
        lines.append("## Early diagnostics (power and null distribution context)")
        lines.append("")
        try:
            clusters = early_diagnostics.sort_values("Cluster")
            max_share = float(clusters["observed_max_ppa_share"].iloc[0])
            p = float(clusters["perm_p_value_max_share"].iloc[0])
            p95 = float(clusters["perm_p95_max_share"].iloc[0])
            lines.append(
                f"- Early max PPA share = {_fmt_float(max_share, 3)}; permutation p={_fmt_float(p, 4)}; null 95th percentile={_fmt_float(p95, 3)}"
            )
            for _, r in clusters.iterrows():
                lines.append(
                    f"  - Cluster {int(r['Cluster'])}: n={int(r['n'])}, PPA share={_fmt_pct(float(r['ppa_share']))}"
                )
        except Exception:
            lines.append("- Early diagnostics available (see early diagnostics table).")

    if early_centroid_summary is not None and not early_centroid_summary.empty:
        lines.append("")
        lines.append("## Early supportive analysis (fixed Middle/Late centroid assignment)")
        lines.append("")
        r = early_centroid_summary.iloc[0]
        lines.append(
            f"- Assigned-language n={int(r['n_assigned_language'])}/{int(r['n_stage'])}, "
            f"PPA share={_fmt_pct(float(r['ppa_share_assigned_language']))} vs baseline {_fmt_pct(float(r['ppa_share_stage']))}, "
            f"RR={_fmt_float(float(r['rr']))} (95% CI {_fmt_float(float(r['rr_ci95_low']))}–{_fmt_float(float(r['rr_ci95_high']))}), "
            f"p={_fmt_float(float(r['perm_p_value']), 4)}, Jaccard vs within-stage language cluster={_fmt_float(float(r['jaccard_vs_within_stage_language_cluster']), 3)}"
        )

    if bootstrap_stability is not None and not bootstrap_stability.empty:
        lines.append("")
        lines.append("## Stability (subsample; fixed PCA representation)")
        lines.append("")
        try:
            df_stab = bootstrap_stability.copy()
            if stage_order:
                df_stab["Temporal_Stage"] = pd.Categorical(
                    df_stab["Temporal_Stage"], categories=stage_order, ordered=True
                )
            df_stab = df_stab.sort_values("Temporal_Stage")
            for stage, g in df_stab.groupby("Temporal_Stage", sort=False, observed=False):
                ari_med = float(g["ari_vs_baseline"].median())
                jac_med = float(g["jaccard_language_membership"].median())
                lines.append(f"- {stage}: median ARI={_fmt_float(ari_med, 3)}, median Jaccard={_fmt_float(jac_med, 3)}")
        except Exception:
            lines.append("- Bootstrap stability results available (see stability table).")

    if seed_stability is not None and not seed_stability.empty:
        lines.append("")
        lines.append("## Seed sensitivity (k-means initialization)")
        lines.append("")
        try:
            df_seed = seed_stability.copy()
            if stage_order:
                df_seed["Temporal_Stage"] = pd.Categorical(
                    df_seed["Temporal_Stage"], categories=stage_order, ordered=True
                )
            df_seed = df_seed.sort_values("Temporal_Stage")
            for stage, g in df_seed.groupby("Temporal_Stage", sort=False, observed=False):
                ari_min = float(g["ari_vs_baseline"].min())
                ari_med = float(g["ari_vs_baseline"].median())
                lines.append(f"- {stage}: ARI range min={_fmt_float(ari_min, 3)}, median={_fmt_float(ari_med, 3)}")
        except Exception:
            lines.append("- Seed sensitivity results available (see seed table).")

    if literature_summary is not None and not literature_summary.empty:
        lines.append("")
        lines.append("## External replication (literature; nearest-centroid assignment)")
        lines.append("")
        r = literature_summary.iloc[0]
        lines.append(
            f"- n={int(r['n_docs'])}, assigned-language n={int(r['n_assigned_language'])}, "
            f"PPA share={_fmt_pct(float(r['ppa_share_assigned_language']))} vs overall {_fmt_pct(float(r['ppa_share_overall']))}, "
            f"RR={_fmt_float(float(r['rr']))} (95% CI {_fmt_float(float(r['rr_ci95_low']))}–{_fmt_float(float(r['rr_ci95_high']))}), "
            f"p={_fmt_float(float(r['perm_p_value']), 4)}"
        )
    elif literature_note:
        lines.append("")
        lines.append("## External replication (literature; nearest-centroid assignment)")
        lines.append("")
        lines.append(f"- {literature_note}")

    if combined_results is not None and not combined_results.empty:
        lines.append("")
        lines.append("## Sensitivity summary (directional check)")
        lines.append("")
        # Simple summary: per run tag, count stages with RR>1, and min q.
        grp = combined_results.groupby("run_tag", dropna=False)
        for run_tag, g in grp:
            g = g.copy()
            n_rr_gt1 = int((g["rr"].astype(float) > 1).sum())
            min_q = float(g["perm_q_value_bh"].astype(float).min())
            lines.append(f"- {run_tag}: stages RR>1 = {n_rr_gt1}/3, min q = {_fmt_float(min_q, 4)}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
