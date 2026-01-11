#!/usr/bin/env python3
"""
Phase G ("Prestige Suite")

Generates high-ROI robustness and interpretability figures/tables without:
  - changing the locked primary analysis, or
  - making new LLM calls.

Outputs (timestamped):
  results/phase_g_prestige/prestige_suite__<UTC_TIMESTAMP>/
    - REPORT.md
    - run_config.json
    - figures/
    - tables/
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


PROJECT_ROOT = Path(__file__).resolve().parents[1]  # repo root
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from paperjn.clustering.kmeans import fit_kmeans  # noqa: E402
from paperjn.clustering.pca import fit_pca_with_cap  # noqa: E402
from paperjn.stats.permutation import permutation_p_value_ge  # noqa: E402
from paperjn.utils.paths import ensure_dir  # noqa: E402


def _utc_stamp() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _latest_dir(parent: Path, prefix: str) -> Path:
    parent = Path(parent)
    candidates = sorted([p for p in parent.iterdir() if p.is_dir() and p.name.startswith(prefix)])
    if not candidates:
        raise FileNotFoundError(f"No directories matching {prefix}* under {parent}")
    return candidates[-1]


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _safe_float(x: object) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _odds_ratio_ci_from_2x2(a: int, b: int, c: int, d: int) -> tuple[float, float, float]:
    """OR and Wald 95% CI from 2×2 table with zero-cell continuity correction."""
    aa, bb, cc, dd = float(a), float(b), float(c), float(d)
    if min(aa, bb, cc, dd) == 0.0:
        aa += 0.5
        bb += 0.5
        cc += 0.5
        dd += 0.5
    or_ = (aa * dd) / (bb * cc)
    se = math.sqrt(1.0 / aa + 1.0 / bb + 1.0 / cc + 1.0 / dd)
    lo = math.exp(math.log(or_) - 1.96 * se)
    hi = math.exp(math.log(or_) + 1.96 * se)
    return float(or_), float(lo), float(hi)


def _cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    A_norm = np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
    B_norm = np.linalg.norm(B, axis=1, keepdims=True) + 1e-12
    return (A / A_norm) @ (B / B_norm).T


def _hungarian_map_centroids(baseline_centers: np.ndarray, centers: np.ndarray) -> dict[int, int]:
    sim = _cosine_sim_matrix(centers, baseline_centers)  # (k,k): rows=replicate, cols=baseline
    cost = 1.0 - sim
    row_ind, col_ind = linear_sum_assignment(cost)
    return {int(r): int(c) for r, c in zip(row_ind, col_ind)}


def _assign_to_centers(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(d2, axis=1).astype(int)


@dataclass(frozen=True)
class PrestigeOutputs:
    out_dir: Path
    report_md: Path


def _plot_specification_curve(df: pd.DataFrame, *, out_png: Path) -> None:
    import matplotlib.pyplot as plt

    # Order: keep primary first, then the main prespecified sensitivities.
    preferred = [
        "primary_strict_pca_discovery_k4_bge",
        "sens_include_broad_pca_all_k4_bge",
        "sens_k3_strict_pca_discovery_bge",
        "sens_k5_strict_pca_discovery_bge",
        "sens_k6_strict_pca_discovery_bge",
        "sens_embed_e5_strict_pca_discovery_k4",
    ]
    if set(preferred).issubset(set(df["scenario"].tolist())):
        df = df.set_index("scenario").loc[preferred].reset_index()
    else:
        df = df.sort_values("confirmation_perm_p_value_fixed_cluster", ascending=True).reset_index(drop=True)

    # Human-friendly labels for the y-axis.
    def label_for(row: pd.Series) -> str:
        emb = "e5" if "e5" in str(row["scenario"]) else "bge"
        broad = "broad" if bool(row["include_broad"]) else "strict"
        return f"{row['scenario']}\n(k={int(row['k_fixed'])}, PCA={row['pca_fit_on']}, {emb}, {broad})"

    y = np.arange(len(df))
    or_ = df["confirmation_or"].astype(float).to_numpy()
    lo = df["confirmation_or_ci95_low"].astype(float).to_numpy()
    hi = df["confirmation_or_ci95_high"].astype(float).to_numpy()
    p = df["confirmation_perm_p_value_fixed_cluster"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.errorbar(or_, y, xerr=[or_ - lo, hi - or_], fmt="o", color="black", ecolor="black", capsize=3)
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([label_for(r) for _, r in df.iterrows()], fontsize=8)
    ax.set_xlabel("Odds ratio (PPA enrichment in matched cluster; Confirmation)")
    ax.set_xscale("log")
    ax.grid(True, axis="x", linestyle=":", linewidth=0.6)
    ax.set_title("Specification curve (prespecified sensitivity set; Confirmation effect size)")

    # Annotate p-values on the right.
    x_text = float(np.nanmax(hi)) * 1.15
    for yi, pi in zip(y, p):
        if math.isnan(pi):
            txt = "p=NA"
        elif pi < 0.001:
            txt = "p<.001"
        elif pi < 0.01:
            txt = f"p={pi:.3f}".lstrip("0")
        else:
            txt = f"p={pi:.2f}".lstrip("0")
        ax.text(x_text, yi, txt, va="center", fontsize=8)

    ax.set_xlim(max(0.2, float(np.nanmin(lo)) * 0.8), x_text * 1.15)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_perm_hist(permuted: np.ndarray, *, observed: float, title: str, xlabel: str, out_png: Path) -> None:
    import matplotlib.pyplot as plt

    perm = np.asarray(permuted, dtype=float)
    p = permutation_p_value_ge(float(observed), perm)

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.hist(perm, bins=40, color="#4C78A8", alpha=0.85, edgecolor="white")
    ax.axvline(float(observed), color="#F58518", linewidth=2, label=f"Observed = {observed:.3f}")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Permutation count")
    ax.legend(frameon=False)
    ax.text(
        0.98,
        0.95,
        f"One-sided permutation p={p:.6f}\nB={int(len(perm))}",
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmap(
    M: np.ndarray,
    *,
    title: str,
    out_png: Path,
    cluster_boundaries: list[int] | None = None,
    cmap: str = "viridis",
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    im = ax.imshow(M, vmin=0.0, vmax=1.0, cmap=cmap, interpolation="nearest", aspect="auto")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    if cluster_boundaries:
        for b in cluster_boundaries:
            ax.axhline(b - 0.5, color="white", linewidth=0.8, alpha=0.8)
            ax.axvline(b - 0.5, color="white", linewidth=0.8, alpha=0.8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Co-assignment probability")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_prestige_suite() -> PrestigeOutputs:
    # Canonical inputs
    phase_d_root = PROJECT_ROOT / "results" / "phase_d"
    phase_d_run_dir = _latest_dir(phase_d_root, "pmcoa_split__v2__")
    results_packet_dir = _latest_dir(phase_d_root, "results_packet__")
    robustness_dir = _latest_dir(phase_d_root, "robustness__")

    out_dir = ensure_dir(PROJECT_ROOT / "results" / "phase_g_prestige" / f"prestige_suite__{_utc_stamp()}")
    figs_dir = ensure_dir(out_dir / "figures")
    tables_dir = ensure_dir(out_dir / "tables")

    # Load baseline run config + tables
    run_cfg = _read_json(phase_d_run_dir / "run_config.json")
    rep = pd.read_csv(phase_d_run_dir / "tables" / "replication_summary.csv").iloc[0].to_dict()
    k_fixed = int(rep["k_fixed"])
    lang_disc = int(rep["language_cluster_discovery"])
    lang_conf = int(rep["language_cluster_confirmation_matched"])
    permutations = int(run_cfg.get("permutations", 5000))
    random_seed = int(run_cfg.get("random_seed", 0))

    # --- 1) Specification curve (restricted multiverse) ---
    rob_summary = pd.read_csv(robustness_dir / "robustness_suite_summary.csv")
    rob_summary = rob_summary[rob_summary["status"].astype(str) == "ok"].copy()
    if rob_summary.empty:
        raise RuntimeError("No ok rows in robustness_suite_summary.csv")

    # Compute OR CIs from underlying 2x2 tables (Confirmation) for each scenario.
    rows = []
    for _, r in rob_summary.iterrows():
        scenario = str(r["scenario"])
        run_dir = Path(str(r["run_dir"]))
        case_path = run_dir / "tables" / "case_table_with_clusters.csv"
        df = pd.read_csv(case_path)
        df = df[df["status"].astype(str) == "ok"].copy()
        df["split"] = df["split"].astype(str)
        df["is_ppa"] = df["is_ppa"].astype(bool)
        df["cluster_matched_to_discovery"] = df["cluster_matched_to_discovery"].astype(int)

        lang = int(r["language_cluster_discovery"])
        conf = df[df["split"] == "confirmation"].copy()
        in_mask = conf["cluster_matched_to_discovery"].astype(int) == lang
        n_total = int(len(conf))
        n_cluster = int(in_mask.sum())
        ppa_total = int(conf["is_ppa"].sum())
        ppa_cluster = int(conf.loc[in_mask, "is_ppa"].sum())
        a = ppa_cluster
        b = n_cluster - ppa_cluster
        c = ppa_total - ppa_cluster
        d = (n_total - n_cluster) - c
        or_, lo, hi = _odds_ratio_ci_from_2x2(a, b, c, d)

        rows.append(
            {
                "scenario": scenario,
                "run_dir": str(run_dir),
                "k_fixed": int(r["k_fixed"]),
                "pca_fit_on": str(r["pca_fit_on"]),
                "include_broad": bool("include_broad" in scenario),
                "embedding_family": "e5" if "e5" in scenario else "bge",
                "n_discovery": int(r["n_discovery"]),
                "n_confirmation": int(r["n_confirmation"]),
                "confirmation_perm_p_value_fixed_cluster": _safe_float(r["confirmation_perm_p_value_fixed_cluster"]),
                "confirmation_or": _safe_float(r["confirmation_or"]),
                "confirmation_or_ci95_low": lo,
                "confirmation_or_ci95_high": hi,
                "confirmation_ppa_share_matched_cluster": _safe_float(r["confirmation_ppa_share_matched_cluster"]),
            }
        )

    spec_df = pd.DataFrame(rows)
    spec_csv = tables_dir / "spec_curve.csv"
    spec_df.to_csv(spec_csv, index=False)
    fig_spec = figs_dir / "fig_specification_curve.png"
    _plot_specification_curve(spec_df, out_png=fig_spec)

    # --- 2) Permutation null distributions (primary endpoint) ---
    case_table = pd.read_csv(phase_d_run_dir / "tables" / "case_table_with_clusters.csv")
    case_table = case_table[case_table["status"].astype(str) == "ok"].copy()
    case_table["split"] = case_table["split"].astype(str)
    case_table["is_ppa"] = case_table["is_ppa"].astype(bool)
    case_table["cluster_split"] = case_table["cluster_split"].astype(int)
    case_table["cluster_matched_to_discovery"] = case_table["cluster_matched_to_discovery"].astype(int)

    disc = case_table[case_table["split"] == "discovery"].copy()
    conf = case_table[case_table["split"] == "confirmation"].copy()
    disc_labels = disc["cluster_split"].to_numpy(dtype=int)
    is_ppa_disc = disc["is_ppa"].to_numpy(dtype=bool)
    is_ppa_conf = conf["is_ppa"].to_numpy(dtype=bool)

    # Discovery selection-aware null (max share across clusters)
    rng = np.random.default_rng(int(random_seed))
    cluster_idx = [np.where(disc_labels == cl)[0] for cl in range(int(k_fixed))]
    perm_max = np.zeros(int(permutations), dtype=np.float32)
    for b in range(int(permutations)):
        perm_is = rng.permutation(is_ppa_disc)
        shares = [float(np.mean(perm_is[idx])) if idx.size else 0.0 for idx in cluster_idx]
        perm_max[b] = float(np.max(shares))

    # Observed max share is taken from replication summary.
    obs_max = float(rep["discovery_max_ppa_share"]) if "discovery_max_ppa_share" in rep else float(np.nan)
    perm_disc_csv = tables_dir / "perm_stats_discovery.csv"
    pd.DataFrame({"perm_max_ppa_share": perm_max.astype(float)}).to_csv(perm_disc_csv, index=False)
    fig_perm_disc = figs_dir / "fig_perm_null_discovery.png"
    _plot_perm_hist(
        perm_max,
        observed=obs_max,
        title="Discovery: selection-aware permutation null (max PPA share across k clusters)",
        xlabel="Max within-cluster PPA share under permuted labels",
        out_png=fig_perm_disc,
    )

    # Confirmation fixed-cluster null (share in matched cluster)
    target_mask = conf["cluster_matched_to_discovery"].to_numpy(dtype=int) == int(lang_disc)
    obs_share = float(is_ppa_conf[target_mask].mean()) if target_mask.any() else 0.0
    rng = np.random.default_rng(int(random_seed))
    perm_share = np.zeros(int(permutations), dtype=np.float32)
    for b in range(int(permutations)):
        perm_is = rng.permutation(is_ppa_conf)
        perm_share[b] = float(np.mean(perm_is[target_mask])) if target_mask.any() else 0.0

    perm_conf_csv = tables_dir / "perm_stats_confirmation.csv"
    pd.DataFrame({"perm_ppa_share_matched_cluster": perm_share.astype(float)}).to_csv(perm_conf_csv, index=False)
    fig_perm_conf = figs_dir / "fig_perm_null_confirmation.png"
    _plot_perm_hist(
        perm_share,
        observed=obs_share,
        title="Confirmation: fixed-cluster permutation null (PPA share in prespecified matched cluster)",
        xlabel="Matched-cluster PPA share under permuted labels",
        out_png=fig_perm_conf,
    )

    # --- 3) Consensus co-assignment (bootstrap) ---
    stability_meta = _read_json(phase_d_run_dir / "stability" / "stability_meta.json")
    n_bootstrap = int(stability_meta.get("n_bootstrap", 200))
    subsample_fraction = float(stability_meta.get("subsample_fraction", 0.8))

    # Load full embeddings aligned to case_table rows (by doc_id).
    with np.load(phase_d_run_dir / "case_embeddings.npz", allow_pickle=False) as npz:
        doc_id_all = np.asarray(npz["doc_id"]).astype(str)
        X_full = np.asarray(npz["embeddings"], dtype=np.float32)

    emb_idx = {d: i for i, d in enumerate(doc_id_all)}
    order = [emb_idx[d] for d in case_table["doc_id"].astype(str).tolist()]
    X = X_full[order]
    split_series = case_table["split"].astype(str).to_numpy()

    # Refit PCA using the same rule as Phase D (from run_config).
    pca_fit_on = str(run_cfg.get("pca_fit_on", "discovery")).strip().lower()
    pca_var = float(run_cfg.get("pca", {}).get("variance_threshold", 0.95))
    pca_max = int(run_cfg.get("pca", {}).get("max_components", 50))

    disc_mask_full = split_series == "discovery"
    conf_mask_full = split_series == "confirmation"
    X_fit = X[disc_mask_full] if pca_fit_on == "discovery" else X
    _, pca_fit = fit_pca_with_cap(X_fit, variance_threshold=pca_var, max_components=pca_max)
    X_reduced = pca_fit.pca.transform(X)[:, : pca_fit.n_components_used]

    def _coassign_for_split(split_name: str, mask: np.ndarray, baseline_lang_cluster: int) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
        idx = np.where(mask)[0]
        Xs = X_reduced[idx]
        baseline_labels = case_table.loc[mask, "cluster_split"].to_numpy(dtype=int)
        ids = case_table.loc[mask, "doc_id"].astype(str).tolist()

        n = int(len(idx))
        n_sub = max(int(round(n * subsample_fraction)), k_fixed)
        baseline_centers = np.vstack([Xs[baseline_labels == c].mean(axis=0) for c in range(k_fixed)])

        rng_local = np.random.default_rng(int(random_seed))
        coassign = np.zeros((n, n), dtype=np.float32)
        lang_counts = np.zeros(n, dtype=np.int32)

        for b in range(int(n_bootstrap)):
            sub_idx = rng_local.choice(np.arange(n), size=n_sub, replace=False)
            X_sub = Xs[sub_idx]
            fit = fit_kmeans(X_sub, k=k_fixed, random_seed=int(random_seed))
            mapping = _hungarian_map_centroids(baseline_centers, fit.centers)
            pred = _assign_to_centers(Xs, fit.centers)
            pred_mapped = np.asarray([mapping[int(c)] for c in pred], dtype=int)

            H = np.zeros((n, k_fixed), dtype=np.float32)
            H[np.arange(n), pred_mapped] = 1.0
            coassign += H @ H.T
            lang_counts += (pred_mapped == int(baseline_lang_cluster)).astype(np.int32)

        coassign_prob = coassign / float(n_bootstrap)
        lang_prob = lang_counts.astype(np.float32) / float(n_bootstrap)

        # Order for plotting: baseline cluster, then language-membership probability (descending).
        order_idx = np.lexsort((-lang_prob, baseline_labels))
        return coassign_prob, lang_prob, ids, order_idx

    # Discovery co-assignment (baseline language cluster is lang_disc in discovery label space).
    co_disc, langp_disc, ids_disc, ord_disc = _coassign_for_split("discovery", disc_mask_full, lang_disc)
    # Confirmation co-assignment (baseline language cluster is lang_conf in confirmation label space).
    co_conf, langp_conf, ids_conf, ord_conf = _coassign_for_split("confirmation", conf_mask_full, lang_conf)

    # Save matrices + membership probabilities.
    co_disc_npz = tables_dir / "coassign_discovery.npz"
    np.savez_compressed(
        co_disc_npz,
        doc_id=np.asarray(ids_disc, dtype=str),
        coassign_prob=co_disc.astype(np.float32),
        language_membership_prob=langp_disc.astype(np.float32),
    )
    co_conf_npz = tables_dir / "coassign_confirmation.npz"
    np.savez_compressed(
        co_conf_npz,
        doc_id=np.asarray(ids_conf, dtype=str),
        coassign_prob=co_conf.astype(np.float32),
        language_membership_prob=langp_conf.astype(np.float32),
    )

    # Core language cases list (probability threshold).
    core_thresh = 0.8
    core_rows = []
    for split_name, ids, probs in [
        ("discovery", ids_disc, langp_disc),
        ("confirmation", ids_conf, langp_conf),
    ]:
        for doc_id, pr in zip(ids, probs):
            if float(pr) >= core_thresh:
                pmcid, case_id = (doc_id.split("__", 1) + [""])[:2]
                core_rows.append(
                    {
                        "split": split_name,
                        "doc_id": doc_id,
                        "pmcid": pmcid,
                        "case_id": case_id,
                        "language_membership_prob": float(pr),
                        "threshold": core_thresh,
                    }
                )
    core_csv = tables_dir / "core_language_cases.csv"
    pd.DataFrame(core_rows).sort_values(["split", "language_membership_prob"], ascending=[True, False]).to_csv(
        core_csv, index=False
    )

    # Heatmap plotting with cluster boundaries in baseline order.
    def _cluster_boundaries(mask: np.ndarray) -> list[int]:
        labels = case_table.loc[mask, "cluster_split"].to_numpy(dtype=int)
        sizes = [int((labels == c).sum()) for c in range(k_fixed)]
        bounds = []
        s = 0
        for sz in sizes:
            s += sz
            bounds.append(s)
        return bounds[:-1]

    fig_co_disc = figs_dir / "fig_coassign_discovery.png"
    _plot_heatmap(
        co_disc[np.ix_(ord_disc, ord_disc)],
        title=f"Discovery: consensus co-assignment (bootstrap; B={n_bootstrap}, frac={subsample_fraction:.2f})",
        out_png=fig_co_disc,
        cluster_boundaries=_cluster_boundaries(disc_mask_full),
    )
    fig_co_conf = figs_dir / "fig_coassign_confirmation.png"
    _plot_heatmap(
        co_conf[np.ix_(ord_conf, ord_conf)],
        title=f"Confirmation: consensus co-assignment (bootstrap; B={n_bootstrap}, frac={subsample_fraction:.2f})",
        out_png=fig_co_conf,
        cluster_boundaries=_cluster_boundaries(conf_mask_full),
    )

    # --- 4) Split-separated cluster clinical profile heatmaps (descriptive) ---
    tags_path = phase_d_run_dir / "tables" / "symptom_tags_by_matched_cluster.csv"
    synd_path = phase_d_run_dir / "tables" / "syndrome_composition_by_matched_cluster.csv"
    if not tags_path.exists() or not synd_path.exists():
        raise FileNotFoundError("Missing matched-cluster profile tables in Phase D run.")

    tags = pd.read_csv(tags_path)
    synd = pd.read_csv(synd_path)

    tag_order = [
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
    tags["tag"] = tags["tag"].astype(str)
    tags["tag"] = pd.Categorical(tags["tag"], categories=tag_order, ordered=True)
    tags_pivot = (
        tags.pivot_table(index=["split", "tag"], columns="matched_cluster", values="present_prop", fill_value=0.0)
        .reset_index()
        .sort_values(["split", "tag"])
    )
    tags_pivot.to_csv(tables_dir / "cluster_profiles_tags.csv", index=False)

    synd_order = [
        "PPA_unspecified",
        "svPPA",
        "nfvPPA",
        "lvPPA",
        "bvFTD",
        "PSP",
        "CBS",
        "FTD_MND",
        "FTLD_unspecified",
    ]
    synd["ftld_syndrome_reported"] = synd["ftld_syndrome_reported"].astype(str)
    synd = synd[synd["ftld_syndrome_reported"].isin(synd_order)].copy()
    synd["ftld_syndrome_reported"] = pd.Categorical(synd["ftld_syndrome_reported"], categories=synd_order, ordered=True)
    synd_pivot = (
        synd.pivot_table(
            index=["split", "ftld_syndrome_reported"],
            columns="matched_cluster",
            values="prop_within_cluster",
            fill_value=0.0,
        )
        .reset_index()
        .sort_values(["split", "ftld_syndrome_reported"])
    )
    synd_pivot.to_csv(tables_dir / "cluster_profiles_syndromes.csv", index=False)

    def _two_panel_heatmap(pivot: pd.DataFrame, index_col: str, value_title: str, out_png: Path) -> None:
        import matplotlib.pyplot as plt

        # Use an explicit colorbar axis to avoid overlap/clipping with tight layouts.
        fig = plt.figure(figsize=(11.2, 4.6))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.055], wspace=0.18)
        ax_left = fig.add_subplot(gs[0, 0])
        ax_right = fig.add_subplot(gs[0, 1], sharey=ax_left)
        cax = fig.add_subplot(gs[0, 2])

        im = None
        for ax, split_name in [(ax_left, "discovery"), (ax_right, "confirmation")]:
            sdf = pivot[pivot["split"] == split_name].copy()
            sdf = sdf.set_index(index_col)
            cols = [c for c in sdf.columns if c not in {"split"}]
            M = sdf[cols].to_numpy(dtype=float)
            im = ax.imshow(M, vmin=0.0, vmax=1.0, cmap="magma", aspect="auto", interpolation="nearest")
            ax.set_title(split_name.capitalize())
            ax.set_xticks(range(len(cols)))
            ax.set_xticklabels([f"C{c}" for c in cols], rotation=0)
            ax.set_yticks(range(len(sdf.index)))
            ax.set_yticklabels([str(x) for x in sdf.index], fontsize=8)

        # Only show y tick labels on the left panel for readability.
        ax_right.tick_params(axis="y", labelleft=False)

        if im is not None:
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_label(value_title)

        fig.suptitle(value_title + " by matched cluster (descriptive)", y=0.98)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)

    fig_tags = figs_dir / "fig_cluster_profiles_tags.png"
    _two_panel_heatmap(tags_pivot, index_col="tag", value_title="Symptom tag prevalence (present proportion)", out_png=fig_tags)

    fig_synd = figs_dir / "fig_cluster_profiles_syndromes.png"
    _two_panel_heatmap(
        synd_pivot,
        index_col="ftld_syndrome_reported",
        value_title="Syndrome composition (proportion within cluster)",
        out_png=fig_synd,
    )

    # --- Run config and report ---
    out_run_cfg = {
        "created_utc": out_dir.name.replace("prestige_suite__", ""),
        "inputs": {
            "phase_d_run_dir": str(phase_d_run_dir),
            "results_packet_dir": str(results_packet_dir),
            "robustness_dir": str(robustness_dir),
        },
        "locked_primary": {
            "k_fixed": k_fixed,
            "pca_fit_on": str(run_cfg.get("pca_fit_on")),
            "permutations": permutations,
            "random_seed": random_seed,
            "language_cluster_discovery": lang_disc,
            "language_cluster_confirmation_matched": lang_conf,
        },
        "prestige_suite": {
            "specification_curve_n": int(len(spec_df)),
            "bootstrap": {"n_bootstrap": n_bootstrap, "subsample_fraction": subsample_fraction, "core_threshold": core_thresh},
        },
    }
    _write_json(out_dir / "run_config.json", out_run_cfg)

    # Quick summary stats for REPORT.md.
    p_disc = permutation_p_value_ge(float(obs_max), perm_max.astype(float))
    p_conf = permutation_p_value_ge(float(obs_share), perm_share.astype(float))
    core_disc_n = int(sum((langp_disc >= core_thresh).tolist()))
    core_conf_n = int(sum((langp_conf >= core_thresh).tolist()))

    report_lines: list[str] = []
    report_lines.append("# Phase G — Prestige Suite Report")
    report_lines.append("")
    report_lines.append(f"Run: `{out_dir.name}`")
    report_lines.append("")
    report_lines.append("## Inputs (frozen)")
    report_lines.append(f"- Phase D run: `{phase_d_run_dir}`")
    report_lines.append(f"- Robustness suite: `{robustness_dir}`")
    report_lines.append(f"- Results packet (for context): `{results_packet_dir}`")
    report_lines.append("")
    report_lines.append("## Outputs")
    report_lines.append(f"- Figures: `{figs_dir}`")
    report_lines.append(f"- Tables: `{tables_dir}`")
    report_lines.append("")
    report_lines.append("## Key checks (should match manuscript)")
    report_lines.append(f"- Discovery selection-aware permutation p (primary): {p_disc:.6f} (B={permutations})")
    report_lines.append(f"- Confirmation fixed-cluster permutation p (primary): {p_conf:.6f} (B={permutations})")
    report_lines.append("")
    report_lines.append("## New artifacts (high ROI)")
    report_lines.append(f"- Specification curve: `{fig_spec}` (data: `{spec_csv}`)")
    report_lines.append(f"- Permutation nulls: `{fig_perm_disc}`, `{fig_perm_conf}` (data: `{perm_disc_csv}`, `{perm_conf_csv}`)")
    report_lines.append(f"- Consensus co-assignment (bootstrap): `{fig_co_disc}`, `{fig_co_conf}`")
    report_lines.append(f"  - Matrices: `{co_disc_npz}`, `{co_conf_npz}`")
    report_lines.append(f"  - Core language cases (prob ≥ {core_thresh}): Discovery={core_disc_n}, Confirmation={core_conf_n} (`{core_csv}`)")
    report_lines.append(f"- Cluster profiles (descriptive): `{fig_tags}`, `{fig_synd}`")
    report_lines.append(f"  - Tag matrix: `{tables_dir / 'cluster_profiles_tags.csv'}`")
    report_lines.append(f"  - Syndrome matrix: `{tables_dir / 'cluster_profiles_syndromes.csv'}`")
    report_lines.append("")
    report_lines.append("## Notes for manuscript/poster")
    report_lines.append("- These figures are robustness/stability/characterization artifacts and do not introduce new confirmatory endpoints.")
    report_lines.append("- Use the specification curve to transparently show boundary conditions (k and embedding-family dependence).")
    report_lines.append("- Use the permutation-null plots to make the inference scheme legible to general clinical audiences.")
    report_lines.append("- Use the co-assignment heatmaps to show a stable “core” of the language construct and where instability lies.")
    report_lines.append("")
    report_md = out_dir / "REPORT.md"
    report_md.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return PrestigeOutputs(out_dir=out_dir, report_md=report_md)


def main() -> None:
    out = run_prestige_suite()
    print(f"Wrote: {out.report_md}")


if __name__ == "__main__":
    main()
