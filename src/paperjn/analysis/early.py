from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from paperjn.stats.enrichment import TwoByTwo, compute_effect_sizes
from paperjn.stats.permutation import permutation_p_value_ge


def _cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cosine similarity between rows of A (n,d) and rows of B (m,d) -> (n,m)."""
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    A_norm = np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
    B_norm = np.linalg.norm(B, axis=1, keepdims=True) + 1e-12
    return (A / A_norm) @ (B / B_norm).T


@dataclass(frozen=True)
class EarlyDiagnosticsResult:
    table_path: Path
    figure_path: Path


def compute_stage_cluster_summary(assignments: pd.DataFrame) -> pd.DataFrame:
    required = {"Temporal_Stage", "Cluster", "is_ppa"}
    missing = required - set(assignments.columns)
    if missing:
        raise ValueError(f"Missing required columns in assignments: {sorted(missing)}")

    df = assignments.copy()
    df["Cluster"] = df["Cluster"].astype(int)
    df["is_ppa"] = df["is_ppa"].astype(bool)

    out = (
        df.groupby(["Temporal_Stage", "Cluster"], as_index=False)
        .agg(n=("is_ppa", "size"), ppa_n=("is_ppa", "sum"), ppa_share=("is_ppa", "mean"))
        .sort_values(["Temporal_Stage", "Cluster"])
    )
    return out


def run_early_diagnostics(
    *,
    assignments_primary: pd.DataFrame,
    stage_results_primary: pd.DataFrame,
    permutations: int,
    random_seed: int,
    out_table: Path,
    out_figure: Path,
) -> EarlyDiagnosticsResult:
    """Early-stage diagnostics: cluster sizes/shares + null distribution for max PPA share."""
    import matplotlib.pyplot as plt

    early = assignments_primary[assignments_primary["Temporal_Stage"] == "Early"].copy()
    early["Cluster"] = early["Cluster"].astype(int)
    early["is_ppa"] = early["is_ppa"].astype(bool)

    summary = compute_stage_cluster_summary(early)
    # observed max PPA share (tie -> smallest cluster id)
    max_share = float(summary["ppa_share"].max())
    language_cluster = int(summary[summary["ppa_share"] == max_share]["Cluster"].min())

    # permutation null (labels fixed; permute PPA; take max share)
    labels = early["Cluster"].to_numpy()
    is_ppa = early["is_ppa"].to_numpy(dtype=bool)
    rng = np.random.default_rng(random_seed)
    perm = np.zeros(permutations, dtype=np.float32)
    clusters = sorted(np.unique(labels).tolist())
    for i in range(permutations):
        perm_is = rng.permutation(is_ppa)
        shares = []
        for cl in clusters:
            m = labels == cl
            shares.append(float(np.mean(perm_is[m])) if np.any(m) else 0.0)
        perm[i] = float(np.max(shares))
    p_perm = permutation_p_value_ge(max_share, perm)

    # Merge in primary stage-level p/q for context.
    early_stage_row = stage_results_primary[stage_results_primary["Temporal_Stage"] == "Early"]
    if len(early_stage_row) == 1:
        early_stage_row = early_stage_row.iloc[0].to_dict()
    else:
        early_stage_row = {}

    out_df = summary.copy()
    out_df["observed_language_cluster"] = language_cluster
    out_df["observed_max_ppa_share"] = max_share
    out_df["perm_p_value_max_share"] = p_perm
    out_df["perm_mean_max_share"] = float(np.mean(perm))
    out_df["perm_p95_max_share"] = float(np.quantile(perm, 0.95))
    out_df["perm_B"] = int(permutations)
    if early_stage_row:
        out_df["primary_perm_p_value"] = float(early_stage_row.get("perm_p_value", np.nan))
        out_df["primary_perm_q_value_bh"] = float(early_stage_row.get("perm_q_value_bh", np.nan))

    out_table.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_table, index=False)

    # Plot null distribution
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.hist(perm, bins=40, color="#cccccc", edgecolor="white")
    ax.axvline(max_share, color="red", linewidth=2, label=f"Observed max share = {max_share:.3f}")
    ax.set_title("Early: permutation null of max PPA share (k=4 clusters)")
    ax.set_xlabel("Max PPA share across clusters")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6)
    fig.tight_layout()
    out_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_figure, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return EarlyDiagnosticsResult(table_path=out_table, figure_path=out_figure)


@dataclass(frozen=True)
class EarlyCentroidAssignmentResult:
    summary_path: Path
    assignments_path: Path
    figure_path: Path


def run_early_centroid_assignment(
    *,
    curated_embeddings: np.ndarray,
    curated_table: pd.DataFrame,
    assignments_primary: pd.DataFrame,
    stage_results_primary: pd.DataFrame,
    ppa_markers: list[str],
    permutations: int,
    random_seed: int,
    out_summary: Path,
    out_assignments: Path,
    out_figure: Path,
) -> EarlyCentroidAssignmentResult:
    """Supportive Early analysis using Middle/Late prototypes (fixed nearest-centroid assignment)."""
    import matplotlib.pyplot as plt

    required_assign = {"row", "Temporal_Stage", "Cluster", "Subtype", "is_ppa"}
    missing = required_assign - set(assignments_primary.columns)
    if missing:
        raise ValueError(f"Missing required columns in assignments_primary: {sorted(missing)}")

    # Identify language clusters for Middle and Late from primary stage results.
    lang = (
        stage_results_primary.set_index("Temporal_Stage")["language_cluster"].astype(int).to_dict()
        if "language_cluster" in stage_results_primary.columns
        else {}
    )
    if "Middle" not in lang or "Late" not in lang:
        raise ValueError("Need language_cluster for Middle and Late in stage results.")

    # Compute centroids for Middle+Late clusters in full embedding space.
    df_a = assignments_primary.copy()
    df_a["Cluster"] = df_a["Cluster"].astype(int)

    ref_rows = df_a[df_a["Temporal_Stage"].isin(["Middle", "Late"])].copy()
    ref_centroids = []
    ref_labels = []
    for stage in ["Middle", "Late"]:
        for cl in sorted(ref_rows[ref_rows["Temporal_Stage"] == stage]["Cluster"].unique().tolist()):
            rows = ref_rows[(ref_rows["Temporal_Stage"] == stage) & (ref_rows["Cluster"] == cl)]["row"].to_numpy()
            centroid = curated_embeddings[rows].mean(axis=0)
            ref_centroids.append(centroid)
            ref_labels.append({"ref_stage": stage, "ref_cluster": int(cl), "is_language_ref": int(cl) == int(lang[stage])})

    ref_centroids = np.vstack(ref_centroids).astype(np.float32)
    ref_meta = pd.DataFrame(ref_labels)

    # Assign Early units to nearest reference centroid (cosine similarity).
    early_rows = df_a[df_a["Temporal_Stage"] == "Early"].copy()
    early_idx = early_rows["row"].to_numpy()
    early_emb = curated_embeddings[early_idx]
    sim = _cosine_sim_matrix(early_emb, ref_centroids)
    best = np.argmax(sim, axis=1)

    assigned = early_rows[["row", "Subtype", "Cluster", "is_ppa"]].copy()
    assigned = assigned.rename(columns={"Cluster": "early_cluster"})
    assigned["assigned_ref_idx"] = best.astype(int)
    assigned["assigned_ref_stage"] = ref_meta.loc[best, "ref_stage"].to_numpy()
    assigned["assigned_ref_cluster"] = ref_meta.loc[best, "ref_cluster"].to_numpy()
    assigned["assigned_language"] = ref_meta.loc[best, "is_language_ref"].to_numpy().astype(bool)
    assigned["assigned_similarity"] = sim[np.arange(sim.shape[0]), best]

    # Effect size + permutation p-value (fixed assignment; permute PPA labels within Early).
    is_ppa = assigned["is_ppa"].astype(bool).to_numpy()
    is_lang = assigned["assigned_language"].astype(bool).to_numpy()

    n_stage = int(len(assigned))
    ppa_stage = int(is_ppa.sum())
    n_lang = int(is_lang.sum())
    ppa_lang = int(is_ppa[is_lang].sum())

    a = ppa_lang
    b = n_lang - ppa_lang
    c = ppa_stage - ppa_lang
    d = (n_stage - n_lang) - c
    effects = compute_effect_sizes(TwoByTwo(a=a, b=b, c=c, d=d))

    rng = np.random.default_rng(random_seed)
    perm = np.zeros(permutations, dtype=np.float32)
    for i in range(permutations):
        perm_is = rng.permutation(is_ppa)
        perm[i] = float(np.mean(perm_is[is_lang])) if n_lang > 0 else 0.0
    p_perm = permutation_p_value_ge(float(effects.ppa_share_cluster), perm)

    # Overlap with baseline Early language cluster from within-stage clustering (primary).
    early_language_cluster = int(stage_results_primary.set_index("Temporal_Stage").loc["Early", "language_cluster"])
    baseline_lang = assigned["early_cluster"].astype(int).to_numpy() == early_language_cluster
    jaccard = float(
        (np.logical_and(baseline_lang, is_lang).sum())
        / max(np.logical_or(baseline_lang, is_lang).sum(), 1)
    )

    summary = pd.DataFrame(
        [
            {
                "stage": "Early",
                "method": "nearest_centroid_to_middle_late",
                "n_stage": n_stage,
                "n_assigned_language": n_lang,
                "ppa_stage": ppa_stage,
                "ppa_assigned_language": ppa_lang,
                "ppa_share_assigned_language": effects.ppa_share_cluster,
                "ppa_share_stage": effects.ppa_share_stage,
                "rr": effects.rr,
                "rr_ci95_low": effects.rr_ci95_low,
                "rr_ci95_high": effects.rr_ci95_high,
                "or": effects.or_,
                "or_ci95_low": effects.or_ci95_low,
                "or_ci95_high": effects.or_ci95_high,
                "perm_p_value": p_perm,
                "perm_B": int(permutations),
                "jaccard_vs_within_stage_language_cluster": jaccard,
                "middle_language_cluster": int(lang["Middle"]),
                "late_language_cluster": int(lang["Late"]),
                "early_language_cluster_within_stage": early_language_cluster,
            }
        ]
    )

    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_assignments.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_summary, index=False)
    assigned.to_csv(out_assignments, index=False)

    # Simple bar plot: PPA share in assigned language vs baseline.
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.bar(
        ["Assigned language", "Early baseline"],
        [effects.ppa_share_cluster, effects.ppa_share_stage],
        color=["#4C72B0", "#DD8452"],
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("PPA share")
    ax.set_title("Early: PPA share under fixed Middle/Late centroid assignment")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6)
    fig.tight_layout()
    out_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_figure, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return EarlyCentroidAssignmentResult(
        summary_path=out_summary, assignments_path=out_assignments, figure_path=out_figure
    )

