from __future__ import annotations

from pathlib import Path

import pandas as pd


def _require_matplotlib() -> None:
    try:
        import matplotlib  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing plotting dependencies. Install with matplotlib."
        ) from exc


def plot_ppa_enrichment_forest(
    stage_results: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
) -> Path:
    _require_matplotlib()
    import matplotlib.pyplot as plt

    df = stage_results.copy()
    df = df.sort_values("Temporal_Stage")

    y = list(range(len(df)))
    rr = df["rr"].astype(float).to_numpy()
    lo = df["rr_ci95_low"].astype(float).to_numpy()
    hi = df["rr_ci95_high"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.errorbar(rr, y, xerr=[rr - lo, hi - rr], fmt="o", color="black", ecolor="black", capsize=3)
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["Temporal_Stage"].tolist())
    ax.set_xlabel("Risk ratio (PPA in language cluster vs stage baseline)")
    ax.set_title(title)
    ax.set_xscale("log")
    ax.grid(True, axis="x", linestyle=":", linewidth=0.6)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_ppa_share_bars(
    stage_results: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
) -> Path:
    _require_matplotlib()
    import matplotlib.pyplot as plt

    df = stage_results.copy()
    df = df.sort_values("Temporal_Stage")
    stages = df["Temporal_Stage"].tolist()
    x = list(range(len(stages)))
    width = 0.35

    y_cluster = df["ppa_share_language_cluster"].astype(float).to_numpy()
    y_stage = df["ppa_share_stage"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar([v - width / 2 for v in x], y_cluster, width=width, label="Language cluster")
    ax.bar([v + width / 2 for v in x], y_stage, width=width, label="Stage baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylim(0, 1)
    ax.set_ylabel("PPA share")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_ppa_share_by_cluster(
    assignments: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
) -> Path:
    _require_matplotlib()
    import matplotlib.pyplot as plt

    df = assignments.copy()
    # Expect columns: Temporal_Stage, Cluster, is_ppa
    agg = (
        df.groupby(["Temporal_Stage", "Cluster"], as_index=False)["is_ppa"]
        .mean()
        .rename(columns={"is_ppa": "PPA_share"})
    )
    agg = agg.sort_values(["Temporal_Stage", "Cluster"])

    stages = sorted(agg["Temporal_Stage"].unique().tolist())
    clusters = sorted(agg["Cluster"].unique().tolist())

    # Matrix of shares indexed by (stage, cluster)
    share = {(r["Temporal_Stage"], int(r["Cluster"])): float(r["PPA_share"]) for _, r in agg.iterrows()}

    x = list(range(len(clusters)))
    n_stages = len(stages)
    width = 0.8 / max(n_stages, 1)

    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for i, stage in enumerate(stages):
        offset = (i - (n_stages - 1) / 2) * width
        heights = [share.get((stage, int(cl)), 0.0) for cl in clusters]
        ax.bar([v + offset for v in x], heights, width=width, label=stage)

    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in clusters])
    ax.set_xlabel("Cluster")
    ax.set_ylim(0, 1)
    ax.set_ylabel("PPA share")
    ax.set_title(title)
    ax.legend(title="Stage", frameon=False)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path
