from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from paperjn.clustering.kmeans import fit_kmeans
from paperjn.stats.enrichment import TwoByTwo, compute_effect_sizes


def _cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    A_norm = np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
    B_norm = np.linalg.norm(B, axis=1, keepdims=True) + 1e-12
    return (A / A_norm) @ (B / B_norm).T


def _hungarian_map_centroids(baseline_centers: np.ndarray, centers: np.ndarray) -> dict[int, int]:
    sim = _cosine_sim_matrix(centers, baseline_centers)  # (k,k)
    cost = 1.0 - sim
    row_ind, col_ind = linear_sum_assignment(cost)
    return {int(r): int(c) for r, c in zip(row_ind, col_ind)}


def _assign_to_centers(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    # Euclidean nearest center
    X = np.asarray(X, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    # (n,k)
    d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(d2, axis=1).astype(int)


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 1.0


@dataclass(frozen=True)
class BootstrapStabilityResult:
    table_path: Path
    fig_ari_path: Path
    fig_jaccard_path: Path


def run_bootstrap_stability(
    *,
    X_reduced: np.ndarray,
    assignments_primary: pd.DataFrame,
    stage_results_primary: pd.DataFrame,
    n_bootstrap: int,
    subsample_fraction: float,
    k_fixed: int,
    random_seed: int,
    out_table: Path,
    out_fig_ari: Path,
    out_fig_jaccard: Path,
) -> BootstrapStabilityResult:
    """Subsample-based stability (without replacement) on the fixed PCA representation."""
    import matplotlib.pyplot as plt

    df_a = assignments_primary.copy()
    df_a["Cluster"] = df_a["Cluster"].astype(int)
    df_a["is_ppa"] = df_a["is_ppa"].astype(bool)
    df_a = df_a.sort_values(["Temporal_Stage", "row"]).reset_index(drop=True)

    lang = stage_results_primary.set_index("Temporal_Stage")["language_cluster"].astype(int).to_dict()

    rng = np.random.default_rng(random_seed)
    rows = []

    for stage in sorted(df_a["Temporal_Stage"].unique().tolist()):
        stage_df = df_a[df_a["Temporal_Stage"] == stage].copy()
        idx = stage_df["row"].to_numpy()
        X_stage = X_reduced[idx]
        baseline_labels = stage_df["Cluster"].to_numpy(dtype=int)
        baseline_lang_mask = baseline_labels == int(lang.get(stage, -1))

        # baseline centers in reduced space
        baseline_centers = np.vstack([X_stage[baseline_labels == c].mean(axis=0) for c in range(k_fixed)])

        n = len(idx)
        n_sub = max(int(round(n * subsample_fraction)), k_fixed)

        for b in range(n_bootstrap):
            sub_idx = rng.choice(np.arange(n), size=n_sub, replace=False)
            X_sub = X_stage[sub_idx]

            fit = fit_kmeans(X_sub, k=k_fixed, random_seed=random_seed)
            centers = fit.centers
            mapping = _hungarian_map_centroids(baseline_centers, centers)

            # Assign all points to this fit's centers
            pred = _assign_to_centers(X_stage, centers)

            # Map cluster ids to baseline cluster ids for ARI comparison
            pred_mapped = np.array([mapping[int(c)] for c in pred], dtype=int)
            ari = adjusted_rand_score(baseline_labels, pred_mapped)

            # Identify replicate language cluster (max PPA share in replicate clusters)
            is_ppa = stage_df["is_ppa"].to_numpy(dtype=bool)
            ppa_share_by_cluster = []
            for c in range(k_fixed):
                m = pred == c
                ppa_share_by_cluster.append(float(np.mean(is_ppa[m])) if np.any(m) else 0.0)
            max_share = float(np.max(ppa_share_by_cluster))
            lang_cluster = int(np.where(np.isclose(ppa_share_by_cluster, max_share))[0].min())
            lang_mask = pred == lang_cluster
            jacc = _jaccard(baseline_lang_mask, lang_mask)

            # Effect size
            n_stage = int(n)
            ppa_stage = int(is_ppa.sum())
            n_lang = int(lang_mask.sum())
            ppa_lang = int(is_ppa[lang_mask].sum())
            a = ppa_lang
            bb = n_lang - ppa_lang
            c = ppa_stage - ppa_lang
            d = (n_stage - n_lang) - c
            eff = compute_effect_sizes(TwoByTwo(a=a, b=bb, c=c, d=d))

            rows.append(
                {
                    "Temporal_Stage": stage,
                    "replicate": b,
                    "ari_vs_baseline": float(ari),
                    "jaccard_language_membership": float(jacc),
                    "language_cluster": int(lang_cluster),
                    "language_cluster_size": int(n_lang),
                    "ppa_share_language_cluster": float(eff.ppa_share_cluster),
                    "rr": float(eff.rr),
                }
            )

    out_df = pd.DataFrame(rows)
    out_table.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_table, index=False)

    # ARI plot
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for i, stage in enumerate(sorted(out_df["Temporal_Stage"].unique().tolist())):
        vals = out_df[out_df["Temporal_Stage"] == stage]["ari_vs_baseline"].astype(float).to_numpy()
        ax.boxplot(vals, positions=[i], widths=0.6)
    ax.set_xticks(list(range(len(sorted(out_df["Temporal_Stage"].unique().tolist())))))
    ax.set_xticklabels(sorted(out_df["Temporal_Stage"].unique().tolist()))
    ax.set_ylabel("ARI vs baseline")
    ax.set_title("Stability (subsample): ARI vs baseline clustering")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6)
    fig.tight_layout()
    out_fig_ari.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig_ari, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Jaccard plot
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for i, stage in enumerate(sorted(out_df["Temporal_Stage"].unique().tolist())):
        vals = (
            out_df[out_df["Temporal_Stage"] == stage]["jaccard_language_membership"].astype(float).to_numpy()
        )
        ax.boxplot(vals, positions=[i], widths=0.6)
    ax.set_xticks(list(range(len(sorted(out_df["Temporal_Stage"].unique().tolist())))))
    ax.set_xticklabels(sorted(out_df["Temporal_Stage"].unique().tolist()))
    ax.set_ylabel("Jaccard (language membership)")
    ax.set_title("Stability (subsample): language-cluster membership overlap")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6)
    fig.tight_layout()
    out_fig_jaccard.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig_jaccard, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return BootstrapStabilityResult(table_path=out_table, fig_ari_path=out_fig_ari, fig_jaccard_path=out_fig_jaccard)


@dataclass(frozen=True)
class SeedStabilityResult:
    table_path: Path


def run_seed_sensitivity(
    *,
    X_reduced: np.ndarray,
    assignments_primary: pd.DataFrame,
    stage_results_primary: pd.DataFrame,
    seed_values: list[int],
    k_fixed: int,
    baseline_seed: int,
    out_table: Path,
) -> SeedStabilityResult:
    df_a = assignments_primary.copy()
    df_a["Cluster"] = df_a["Cluster"].astype(int)
    df_a["is_ppa"] = df_a["is_ppa"].astype(bool)
    df_a = df_a.sort_values(["Temporal_Stage", "row"]).reset_index(drop=True)

    lang = stage_results_primary.set_index("Temporal_Stage")["language_cluster"].astype(int).to_dict()

    rows = []

    for stage in sorted(df_a["Temporal_Stage"].unique().tolist()):
        stage_df = df_a[df_a["Temporal_Stage"] == stage].copy()
        idx = stage_df["row"].to_numpy()
        X_stage = X_reduced[idx]
        baseline_labels = stage_df["Cluster"].to_numpy(dtype=int)

        # baseline centers in reduced space for mapping
        baseline_centers = np.vstack([X_stage[baseline_labels == c].mean(axis=0) for c in range(k_fixed)])

        for seed in seed_values:
            fit = fit_kmeans(X_stage, k=k_fixed, random_seed=seed)
            mapping = _hungarian_map_centroids(baseline_centers, fit.centers)
            labels = fit.labels
            labels_mapped = np.array([mapping[int(c)] for c in labels], dtype=int)
            ari = adjusted_rand_score(baseline_labels, labels_mapped)

            # language cluster under this seed (max PPA share)
            is_ppa = stage_df["is_ppa"].to_numpy(dtype=bool)
            shares = []
            for c in range(k_fixed):
                m = labels == c
                shares.append(float(np.mean(is_ppa[m])) if np.any(m) else 0.0)
            max_share = float(np.max(shares))
            lang_cluster = int(np.where(np.isclose(shares, max_share))[0].min())

            # Effect size
            lang_mask = labels == lang_cluster
            n_stage = int(len(stage_df))
            ppa_stage = int(is_ppa.sum())
            n_lang = int(lang_mask.sum())
            ppa_lang = int(is_ppa[lang_mask].sum())
            a = ppa_lang
            bb = n_lang - ppa_lang
            c = ppa_stage - ppa_lang
            d = (n_stage - n_lang) - c
            eff = compute_effect_sizes(TwoByTwo(a=a, b=bb, c=c, d=d))

            rows.append(
                {
                    "Temporal_Stage": stage,
                    "seed": int(seed),
                    "ari_vs_baseline": float(ari),
                    "language_cluster": int(lang_cluster),
                    "language_cluster_size": int(n_lang),
                    "ppa_share_language_cluster": float(eff.ppa_share_cluster),
                    "rr": float(eff.rr),
                    "baseline_language_cluster": int(lang.get(stage, -1)),
                    "baseline_seed": int(baseline_seed),
                }
            )

    out_df = pd.DataFrame(rows)
    out_table.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_table, index=False)
    return SeedStabilityResult(table_path=out_table)

