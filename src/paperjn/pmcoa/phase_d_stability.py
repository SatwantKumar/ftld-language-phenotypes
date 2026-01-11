from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from paperjn.clustering.kmeans import fit_kmeans
from paperjn.clustering.pca import fit_pca_with_cap
from paperjn.config import ProjectConfig
from paperjn.stats.enrichment import TwoByTwo, compute_effect_sizes
from paperjn.utils.paths import ensure_dir


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
    X = np.asarray(X, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(d2, axis=1).astype(int)


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 1.0


@dataclass(frozen=True)
class PhaseDStabilityOutputs:
    out_dir: Path
    bootstrap_csv: Path
    seed_csv: Path
    fig_ari_path: Path
    fig_jaccard_path: Path
    meta_json: Path
    report_md: Path


def run_phase_d_stability(
    *,
    config: ProjectConfig,
    phase_d_run_dir: Path,
    n_bootstrap: int | None = None,
    subsample_fraction: float | None = None,
    seed_values: list[int] | None = None,
) -> PhaseDStabilityOutputs:
    """Stability for Phase D split clustering (ARI + language-cluster membership overlap)."""
    import matplotlib.pyplot as plt

    phase_d_run_dir = Path(phase_d_run_dir).resolve()
    run_config_path = phase_d_run_dir / "run_config.json"
    if not run_config_path.exists():
        raise FileNotFoundError(f"Missing Phase D run_config.json: {run_config_path}")

    tables_dir = phase_d_run_dir / "tables"
    case_table_path = tables_dir / "case_table_with_clusters.csv"
    if not case_table_path.exists():
        raise FileNotFoundError(f"Missing Phase D case table: {case_table_path}")

    # We rely on the saved case embeddings (full space) to re-fit PCA for the stability runs.
    emb_path = phase_d_run_dir / "case_embeddings.npz"
    if not emb_path.exists():
        raise FileNotFoundError(f"Missing Phase D case embeddings: {emb_path}")

    rep_path = tables_dir / "replication_summary.csv"
    if not rep_path.exists():
        raise FileNotFoundError(f"Missing replication_summary.csv: {rep_path}")

    rep = pd.read_csv(rep_path).iloc[0].to_dict()
    k_fixed = int(rep["k_fixed"])
    lang_disc = int(rep["language_cluster_discovery"])
    lang_conf = int(rep["language_cluster_confirmation_matched"])

    try:
        with np.load(emb_path, allow_pickle=False) as npz:
            doc_id = np.asarray(npz["doc_id"]).astype(str)
            X_full = np.asarray(npz["embeddings"], dtype=np.float32)
    except ValueError:
        # Back-compat: older runs may have stored string arrays as dtype=object, which requires pickling.
        with np.load(emb_path, allow_pickle=True) as npz:
            doc_id = np.asarray(npz["doc_id"]).astype(str)
            X_full = np.asarray(npz["embeddings"], dtype=np.float32)

    df = pd.read_csv(case_table_path)
    df["doc_id"] = df["doc_id"].astype(str)
    df = df.drop_duplicates(subset=["doc_id"], keep="last").reset_index(drop=True)
    df = df[df["status"].astype(str) == "ok"].copy()
    df["is_ppa"] = df["is_ppa"].astype(bool)
    df["cluster_split"] = df["cluster_split"].astype(int)

    # Align embeddings to df row order.
    emb_idx = {d: i for i, d in enumerate(doc_id)}
    order = [emb_idx[d] for d in df["doc_id"].tolist()]
    X = X_full[order]

    # PCA re-fit using the same rule as Phase D.
    run_cfg = json.loads(run_config_path.read_text(encoding="utf-8"))
    pca_fit_on = str(run_cfg.get("pca_fit_on", "discovery"))
    if pca_fit_on not in {"discovery", "all"}:
        raise ValueError(f"Unexpected pca_fit_on in run_config: {pca_fit_on}")

    split_series = df["split"].astype(str)
    disc_mask = split_series == "discovery"
    conf_mask = split_series == "confirmation"
    if not disc_mask.any() or not conf_mask.any():
        raise RuntimeError("Need both discovery and confirmation rows for stability.")

    X_fit = X[disc_mask.to_numpy()] if pca_fit_on == "discovery" else X
    _, pca_fit = fit_pca_with_cap(
        X_fit,
        variance_threshold=config.dimensionality_reduction.pca_variance_threshold,
        max_components=config.dimensionality_reduction.pca_max_components,
    )
    X_reduced = pca_fit.pca.transform(X)[:, : pca_fit.n_components_used]

    n_bootstrap = int(n_bootstrap) if n_bootstrap is not None else int(config.stability.n_bootstrap)
    subsample_fraction = float(subsample_fraction) if subsample_fraction is not None else float(config.stability.subsample_fraction)
    seed_values = seed_values if seed_values is not None else list(config.stability.seed_values)

    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be >= 1.")
    if not (0.1 <= subsample_fraction <= 1.0):
        raise ValueError("subsample_fraction must be in [0.1, 1.0].")
    if not seed_values:
        raise ValueError("seed_values must be non-empty.")

    out_dir = ensure_dir(phase_d_run_dir / "stability")

    rng = np.random.default_rng(int(config.random_seed))
    rows_boot: list[dict[str, Any]] = []

    def _run_bootstrap_for_split(split_name: str, mask: np.ndarray, baseline_lang_cluster: int) -> None:
        idx = np.where(mask)[0]
        Xs = X_reduced[idx]
        baseline_labels = df.loc[mask, "cluster_split"].to_numpy(dtype=int)
        is_ppa = df.loc[mask, "is_ppa"].to_numpy(dtype=bool)

        baseline_centers = np.vstack([Xs[baseline_labels == c].mean(axis=0) for c in range(k_fixed)])
        baseline_lang_mask = baseline_labels == int(baseline_lang_cluster)

        n = int(len(idx))
        n_sub = max(int(round(n * subsample_fraction)), k_fixed)

        for b in range(int(n_bootstrap)):
            sub_idx = rng.choice(np.arange(n), size=n_sub, replace=False)
            X_sub = Xs[sub_idx]
            fit = fit_kmeans(X_sub, k=k_fixed, random_seed=int(config.random_seed))
            centers = fit.centers
            mapping = _hungarian_map_centroids(baseline_centers, centers)
            pred = _assign_to_centers(Xs, centers)
            pred_mapped = np.array([mapping[int(c)] for c in pred], dtype=int)
            ari = adjusted_rand_score(baseline_labels, pred_mapped)

            # language-cluster membership overlap for the baseline language cluster
            pred_lang_mask = pred_mapped == int(baseline_lang_cluster)
            jacc = _jaccard(baseline_lang_mask, pred_lang_mask)

            n_total = int(n)
            ppa_total = int(is_ppa.sum())
            n_lang = int(pred_lang_mask.sum())
            ppa_lang = int(is_ppa[pred_lang_mask].sum())
            a = ppa_lang
            bb = n_lang - ppa_lang
            c = ppa_total - ppa_lang
            d = (n_total - n_lang) - c
            eff = compute_effect_sizes(TwoByTwo(a=a, b=bb, c=c, d=d))

            rows_boot.append(
                {
                    "split": split_name,
                    "replicate": int(b),
                    "ari_vs_baseline": float(ari),
                    "jaccard_language_membership": float(jacc),
                    "language_cluster": int(baseline_lang_cluster),
                    "language_cluster_size": int(n_lang),
                    "ppa_share_language_cluster": float(eff.ppa_share_cluster),
                    "rr": float(eff.rr),
                }
            )

    _run_bootstrap_for_split("discovery", disc_mask.to_numpy(), lang_disc)
    _run_bootstrap_for_split("confirmation", conf_mask.to_numpy(), lang_conf)

    boot_df = pd.DataFrame(rows_boot)
    boot_csv = out_dir / "stability_bootstrap.csv"
    boot_df.to_csv(boot_csv, index=False)

    # Seed sensitivity
    rows_seed: list[dict[str, Any]] = []

    def _run_seed_for_split(split_name: str, mask: np.ndarray, baseline_lang_cluster: int) -> None:
        idx = np.where(mask)[0]
        Xs = X_reduced[idx]
        baseline_labels = df.loc[mask, "cluster_split"].to_numpy(dtype=int)
        is_ppa = df.loc[mask, "is_ppa"].to_numpy(dtype=bool)

        baseline_centers = np.vstack([Xs[baseline_labels == c].mean(axis=0) for c in range(k_fixed)])
        baseline_lang_mask = baseline_labels == int(baseline_lang_cluster)

        for seed in seed_values:
            fit = fit_kmeans(Xs, k=k_fixed, random_seed=int(seed))
            mapping = _hungarian_map_centroids(baseline_centers, fit.centers)
            labels_seed = fit.labels.astype(int)
            labels_mapped = np.array([mapping[int(c)] for c in labels_seed], dtype=int)
            ari = adjusted_rand_score(baseline_labels, labels_mapped)

            lang_mask = labels_mapped == int(baseline_lang_cluster)
            jacc = _jaccard(baseline_lang_mask, lang_mask)

            n_total = int(len(is_ppa))
            ppa_total = int(is_ppa.sum())
            n_lang = int(lang_mask.sum())
            ppa_lang = int(is_ppa[lang_mask].sum())
            a = ppa_lang
            bb = n_lang - ppa_lang
            c = ppa_total - ppa_lang
            d = (n_total - n_lang) - c
            eff = compute_effect_sizes(TwoByTwo(a=a, b=bb, c=c, d=d))

            rows_seed.append(
                {
                    "split": split_name,
                    "seed": int(seed),
                    "ari_vs_baseline": float(ari),
                    "jaccard_language_membership": float(jacc),
                    "language_cluster": int(baseline_lang_cluster),
                    "language_cluster_size": int(n_lang),
                    "ppa_share_language_cluster": float(eff.ppa_share_cluster),
                    "rr": float(eff.rr),
                }
            )

    _run_seed_for_split("discovery", disc_mask.to_numpy(), lang_disc)
    _run_seed_for_split("confirmation", conf_mask.to_numpy(), lang_conf)

    seed_df = pd.DataFrame(rows_seed)
    seed_csv = out_dir / "stability_seed.csv"
    seed_df.to_csv(seed_csv, index=False)

    # Figures
    fig_ari_path = out_dir / "fig_stability_ari.png"
    fig_jaccard_path = out_dir / "fig_stability_jaccard.png"

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for i, split_name in enumerate(["discovery", "confirmation"]):
        vals = boot_df[boot_df["split"] == split_name]["ari_vs_baseline"].astype(float).to_numpy()
        ax.boxplot(vals, positions=[i], widths=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Discovery", "Confirmation"])
    ax.set_ylabel("ARI vs baseline")
    ax.set_title("Stability (subsample): ARI vs baseline clustering")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(fig_ari_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for i, split_name in enumerate(["discovery", "confirmation"]):
        vals = boot_df[boot_df["split"] == split_name]["jaccard_language_membership"].astype(float).to_numpy()
        ax.boxplot(vals, positions=[i], widths=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Discovery", "Confirmation"])
    ax.set_ylabel("Jaccard (language membership)")
    ax.set_title("Stability (subsample): matched language-cluster membership overlap")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(fig_jaccard_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    meta = {
        "phase_d_run_dir": str(phase_d_run_dir),
        "k_fixed": int(k_fixed),
        "pca_fit_on": str(pca_fit_on),
        "n_bootstrap": int(n_bootstrap),
        "subsample_fraction": float(subsample_fraction),
        "seed_values": [int(x) for x in seed_values],
        "baseline_language_cluster_discovery": int(lang_disc),
        "baseline_language_cluster_confirmation": int(lang_conf),
    }
    meta_json = out_dir / "stability_meta.json"
    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Markdown report (manuscript-ready stability summary)
    lines: list[str] = []
    lines.append("# Phase D Stability Report")
    lines.append("")
    lines.append(f"Phase D run: `{phase_d_run_dir}`")
    lines.append(f"- k_fixed: {k_fixed}")
    lines.append(f"- pca_fit_on: {pca_fit_on}")
    lines.append(f"- bootstrap replicates per split: {n_bootstrap}")
    lines.append(f"- subsample fraction: {subsample_fraction:.2f}")
    lines.append(f"- seed sensitivity seeds: {', '.join(map(str, seed_values))}")
    lines.append("")

    def _summarize(df: pd.DataFrame, col: str) -> dict[str, float]:
        s = df[col].astype(float)
        return {
            "median": float(s.median()),
            "p10": float(s.quantile(0.10)),
            "p90": float(s.quantile(0.90)),
            "min": float(s.min()),
            "max": float(s.max()),
        }

    lines.append("## Subsample Stability (bootstrap)")
    for split_name in ["discovery", "confirmation"]:
        sdf = boot_df[boot_df["split"] == split_name].copy()
        ari = _summarize(sdf, "ari_vs_baseline")
        jac = _summarize(sdf, "jaccard_language_membership")
        lines.append(f"### {split_name.title()}")
        lines.append(f"- ARI vs baseline: median={ari['median']:.3f} (p10={ari['p10']:.3f}, p90={ari['p90']:.3f})")
        lines.append(
            f"- Language membership Jaccard: median={jac['median']:.3f} (p10={jac['p10']:.3f}, p90={jac['p90']:.3f})"
        )
        lines.append("")

    lines.append("## Seed Sensitivity")
    for split_name in ["discovery", "confirmation"]:
        sdf = seed_df[seed_df["split"] == split_name].copy()
        ari = _summarize(sdf, "ari_vs_baseline")
        jac = _summarize(sdf, "jaccard_language_membership")
        lines.append(f"### {split_name.title()}")
        lines.append(f"- ARI vs baseline: median={ari['median']:.3f} (min={ari['min']:.3f})")
        lines.append(f"- Language membership Jaccard: median={jac['median']:.3f} (min={jac['min']:.3f})")
        lines.append("")

    lines.append("## Files")
    lines.append(f"- Bootstrap CSV: `{boot_csv}`")
    lines.append(f"- Seed CSV: `{seed_csv}`")
    lines.append(f"- Figure (ARI): `{fig_ari_path}`")
    lines.append(f"- Figure (Jaccard): `{fig_jaccard_path}`")
    lines.append(f"- Meta JSON: `{meta_json}`")
    report_md = out_dir / "stability_report.md"
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return PhaseDStabilityOutputs(
        out_dir=out_dir,
        bootstrap_csv=boot_csv,
        seed_csv=seed_csv,
        fig_ari_path=fig_ari_path,
        fig_jaccard_path=fig_jaccard_path,
        meta_json=meta_json,
        report_md=report_md,
    )
