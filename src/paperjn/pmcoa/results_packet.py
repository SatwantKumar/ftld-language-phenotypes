from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paperjn.utils.paths import ensure_dir


def _utc_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _copy_if_exists(src: Path, dst: Path) -> Path | None:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


@dataclass(frozen=True)
class PhaseDResultsPacket:
    out_dir: Path
    packet_md: Path
    tables_dir: Path
    figures_dir: Path
    copied_files: list[Path]
    generated_figures: list[Path]


def _forest_plot_rr(
    *,
    rows: list[dict[str, float | str]],
    title: str,
    out_path: Path,
) -> Path:
    labels = [str(r["label"]) for r in rows]
    rr = np.array([float(r["rr"]) for r in rows], dtype=float)
    lo = np.array([float(r["rr_lo"]) for r in rows], dtype=float)
    hi = np.array([float(r["rr_hi"]) for r in rows], dtype=float)

    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(7.2, max(2.6, 0.45 * len(labels) + 1.2)))
    ax.errorbar(rr, y, xerr=[rr - lo, hi - rr], fmt="o", color="black", ecolor="black", capsize=3)
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Risk ratio (RR) with 95% CI")
    ax.set_title(title)
    ax.grid(True, axis="x", linestyle=":", linewidth=0.6)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _cluster_similarity_heatmap(*, sim_csv: Path, out_path: Path) -> Path:
    df = pd.read_csv(sim_csv)
    mat = df.pivot(index="discovery_cluster", columns="confirmation_cluster", values="cosine_similarity").to_numpy()
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xlabel("Confirmation cluster")
    ax.set_ylabel("Discovery cluster")
    ax.set_title("Cluster cosine similarity (centroids)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_phase_d_results_packet(
    *,
    phase_d_run_dir: Path,
    out_dir: Path | None = None,
) -> PhaseDResultsPacket:
    """Assemble manuscript-ready Phase D outputs into a single packet directory."""
    phase_d_run_dir = Path(phase_d_run_dir).resolve()
    tables_in = phase_d_run_dir / "tables"
    if not tables_in.exists():
        raise FileNotFoundError(f"Missing tables/: {tables_in}")

    if out_dir is None:
        out_dir = (phase_d_run_dir.parent / f"results_packet__{_utc_slug()}").resolve()
    out_dir = ensure_dir(Path(out_dir).resolve())
    tables_out = ensure_dir(out_dir / "tables")
    figures_out = ensure_dir(out_dir / "figures")

    copied: list[Path] = []

    for rel in [
        "flow_table.csv",
        "replication_summary.csv",
        "syndrome_replication_summary.csv",
        "cluster_similarity_matrix.csv",
        "syndrome_composition_by_matched_cluster.csv",
        "symptom_tags_by_matched_cluster.csv",
        "cluster_characterization_by_matched_cluster.csv",
        "numeric_characterization_by_matched_cluster.csv",
    ]:
        src = tables_in / rel
        dst = tables_out / rel
        p = _copy_if_exists(src, dst)
        if p is not None:
            copied.append(p)

    # Optional stability artifacts (if present)
    stability_dir = phase_d_run_dir / "stability"
    for rel in [
        "stability_report.md",
        "fig_stability_ari.png",
        "fig_stability_jaccard.png",
        "stability_bootstrap.csv",
        "stability_seed.csv",
    ]:
        src = stability_dir / rel
        dst = out_dir / "stability" / rel
        p = _copy_if_exists(src, dst)
        if p is not None:
            copied.append(p)

    generated: list[Path] = []

    # Forest plot (primary PPA) using RR from replication_summary (Discovery + Confirmation)
    rep_path = tables_in / "replication_summary.csv"
    if rep_path.exists():
        rep = pd.read_csv(rep_path).iloc[0].to_dict()
        rows = [
            {
                "label": "Discovery (max cluster)",
                "rr": float(rep["discovery_rr"]),
                "rr_lo": float(rep["discovery_rr_ci95_low"]),
                "rr_hi": float(rep["discovery_rr_ci95_high"]),
            },
            {
                "label": "Confirmation (matched cluster)",
                "rr": float(rep["confirmation_rr"]),
                "rr_lo": float(rep["confirmation_rr_ci95_low"]),
                "rr_hi": float(rep["confirmation_rr_ci95_high"]),
            },
        ]
        fig_path = _forest_plot_rr(
            rows=rows,
            title="Primary endpoint: PPA enrichment (RR)",
            out_path=figures_out / "fig_forest_primary_rr.png",
        )
        generated.append(fig_path)

    # Forest plot (secondary syndromes) for Confirmation only
    synd_path = tables_in / "syndrome_replication_summary.csv"
    if synd_path.exists():
        synd = pd.read_csv(synd_path).sort_values("confirmation_perm_p_value_fixed_cluster")
        rows = []
        for _, r in synd.iterrows():
            q = float(r.get("confirmation_q_bh_fdr", float("nan")))
            rows.append(
                {
                    "label": f"{r['endpoint']} (q={q:.3f})",
                    "rr": float(r["confirmation_rr"]),
                    "rr_lo": float(r["confirmation_rr_ci95_low"]),
                    "rr_hi": float(r["confirmation_rr_ci95_high"]),
                }
            )
        fig_path = _forest_plot_rr(
            rows=rows,
            title="Secondary endpoints: syndrome enrichment (Confirmation RR)",
            out_path=figures_out / "fig_forest_secondary_confirmation_rr.png",
        )
        generated.append(fig_path)

    # Cluster similarity heatmap
    sim_csv = tables_in / "cluster_similarity_matrix.csv"
    if sim_csv.exists():
        fig_path = _cluster_similarity_heatmap(sim_csv=sim_csv, out_path=figures_out / "fig_cluster_similarity.png")
        generated.append(fig_path)

    # Packet markdown
    lines: list[str] = []
    lines.append("# Phase D Results Packet")
    lines.append("")
    lines.append(f"Phase D run: `{phase_d_run_dir}`")
    lines.append("")
    lines.append("## Contents")
    lines.append("- Tables: `tables/` (flow + replication summaries + characterization)")
    lines.append("- Figures: `figures/` (forest plots + similarity heatmap)")
    if (out_dir / "stability").exists():
        lines.append("- Stability: `stability/` (bootstrap + seed sensitivity; copied from Phase D run)")
    lines.append("")
    lines.append("## Key Files")
    for p in sorted(copied):
        rel = p.relative_to(out_dir)
        lines.append(f"- `{rel}`")
    for p in sorted(generated):
        rel = p.relative_to(out_dir)
        lines.append(f"- `{rel}`")
    lines.append("")

    packet_md = out_dir / "results_packet.md"
    packet_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return PhaseDResultsPacket(
        out_dir=out_dir,
        packet_md=packet_md,
        tables_dir=tables_out,
        figures_dir=figures_out,
        copied_files=copied,
        generated_figures=generated,
    )
