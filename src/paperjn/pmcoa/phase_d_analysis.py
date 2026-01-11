from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from paperjn.clustering.kmeans import fit_kmeans
from paperjn.clustering.pca import fit_pca_with_cap
from paperjn.config import ProjectConfig
from paperjn.embeddings.sentence_transformers_backend import compute_sentence_transformer_embeddings
from paperjn.nlp.text import normalize_text, normalize_whitespace
from paperjn.stats.enrichment import TwoByTwo, compute_effect_sizes
from paperjn.stats.fdr import bh_fdr
from paperjn.stats.permutation import permutation_p_value_ge
from paperjn.utils.paths import ensure_dir
from paperjn.utils.seed import set_global_seed


PPA_SYNDROMES = {"svPPA", "nfvPPA", "lvPPA", "PPA_unspecified"}
SYNDROME_REPLICATION_ENDPOINTS_V1 = ["PSP", "CBS", "FTD_MND", "svPPA", "nfvPPA", "lvPPA"]

IMAGING_MODALITIES_V1 = ["mri", "ct", "fdg_pet", "spect", "other"]
IMAGING_LATERALITY_V1 = ["left", "right", "bilateral", "diffuse", "unknown"]
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
GENETICS_STATUS_V1 = ["confirmed_pathogenic", "reported_uncertain", "tested_negative", "not_reported", "unclear"]
NEUROPATH_STATUS_V1 = ["confirmed", "not_reported", "unclear"]
PATHOLOGY_TYPES_V1 = ["tau", "tdp43", "fus", "mixed", "other", "unknown"]
KEY_GENES_V1 = ["MAPT", "GRN", "C9ORF72"]


def _cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    A_norm = np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
    B_norm = np.linalg.norm(B, axis=1, keepdims=True) + 1e-12
    return (A / A_norm) @ (B / B_norm).T


def _best_cluster_match(sim: np.ndarray) -> dict[int, int]:
    """Return mapping discovery_cluster -> confirmation_cluster that maximizes total similarity."""
    if sim.ndim != 2 or sim.shape[0] != sim.shape[1]:
        raise ValueError("sim must be square (k x k)")
    k = int(sim.shape[0])
    best_score = -1e9
    best_perm: tuple[int, ...] | None = None
    for perm in itertools.permutations(range(k)):
        score = float(sum(sim[i, perm[i]] for i in range(k)))
        if score > best_score:
            best_score = score
            best_perm = perm
    if best_perm is None:
        raise RuntimeError("Failed to compute best cluster matching.")
    return {int(i): int(best_perm[i]) for i in range(k)}


def _language_cluster(labels: np.ndarray, is_ppa: np.ndarray, *, k: int) -> tuple[int, float]:
    """Return (cluster_id, ppa_share) for the max-PPA cluster, tie -> min id."""
    shares = []
    for cl in range(int(k)):
        mask = labels == cl
        shares.append(float(is_ppa[mask].mean()) if np.any(mask) else 0.0)
    max_share = float(np.max(shares))
    cl_id = int(min(i for i, s in enumerate(shares) if float(s) == max_share))
    return cl_id, max_share


def _enrichment_table(*, labels: np.ndarray, is_ppa: np.ndarray, target_cluster: int) -> tuple[TwoByTwo, float]:
    in_cluster = labels == int(target_cluster)
    n_total = int(len(labels))
    n_cluster = int(in_cluster.sum())
    ppa_total = int(is_ppa.sum())
    ppa_cluster = int(is_ppa[in_cluster].sum())

    a = float(ppa_cluster)
    b = float(n_cluster - ppa_cluster)
    c = float(ppa_total - ppa_cluster)
    d = float((n_total - n_cluster) - (ppa_total - ppa_cluster))
    share = float(ppa_cluster / n_cluster) if n_cluster > 0 else 0.0
    return TwoByTwo(a=a, b=b, c=c, d=d), share


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


def _text_for_embedding(text_clean: str) -> str:
    # text_clean may include "[REDACTED]" markers; remove them to avoid proxy leakage.
    t = normalize_text(str(text_clean))
    t = t.replace("[redacted]", " ")
    return normalize_whitespace(t)


@dataclass(frozen=True)
class PhaseDOutputs:
    run_dir: Path
    run_config_json: Path
    flow_table_csv: Path
    case_table_csv: Path
    replication_summary_csv: Path
    syndrome_replication_summary_csv: Path
    cluster_characterization_csv: Path
    numeric_characterization_csv: Path
    syndrome_composition_csv: Path
    cluster_similarity_csv: Path
    discovery_cluster_summary_csv: Path
    confirmation_cluster_summary_csv: Path
    symptom_tags_discovery_csv: Path
    symptom_tags_confirmation_csv: Path
    symptom_tags_matched_csv: Path
    performance_report_md: Path
    case_embeddings_npz: Path


def run_phase_d_pmcoa_split_analysis(
    *,
    config: ProjectConfig,
    segments_csv: Path,
    case_labels_csv: Path,
    out_dir: Path | None,
    include_broad: bool,
    pca_fit_on: str,
    min_segments_per_case: int,
    one_case_per_pmcid: bool,
    write_latest_outputs: bool,
) -> PhaseDOutputs:
    """Phase D: case-level clustering in Discovery and replication in Confirmation."""
    set_global_seed(config.random_seed)

    segments_csv = Path(segments_csv).resolve()
    case_labels_csv = Path(case_labels_csv).resolve()
    if not segments_csv.exists():
        raise FileNotFoundError(f"segments.csv not found: {segments_csv}")
    if not case_labels_csv.exists():
        raise FileNotFoundError(f"case_labels_long.csv not found: {case_labels_csv}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if out_dir is None:
        out_dir = segments_csv.parent / "phase_d" / f"analysis__{run_id}"
    run_dir = ensure_dir(Path(out_dir).resolve())
    tables_dir = ensure_dir(run_dir / "tables")

    # Load labels (one row per case unit expected).
    labels_all = pd.read_csv(case_labels_csv)
    labels_all["pmcid"] = labels_all["pmcid"].astype(str)
    labels_all["case_id"] = labels_all["case_id"].astype(str)
    labels_all = labels_all.drop_duplicates(subset=["pmcid", "case_id"], keep="last").reset_index(drop=True)

    labels = labels_all.copy()

    min_segments_per_case = int(min_segments_per_case)
    if min_segments_per_case < 1:
        raise ValueError("min_segments_per_case must be >= 1.")

    tiers = ["ftld_strict"] + (["ftld_broad"] if include_broad else [])
    labels = labels[labels["ftld_inclusion_tier"].isin(tiers)].copy()
    if labels.empty:
        raise RuntimeError("No cases remaining after ftld_inclusion_tier filter.")

    labels["is_ppa"] = labels["ftld_syndrome_reported"].astype(str).isin(sorted(PPA_SYNDROMES))

    # Load segments and restrict to labeled cases.
    seg_usecols = [
        "segment_uid",
        "pmcid",
        "case_id",
        "segment_type",
        "include_for_embedding",
        "leakage_n_matches_clean",
        "text_clean",
    ]
    seg = pd.read_csv(segments_csv, usecols=seg_usecols)
    seg["pmcid"] = seg["pmcid"].astype(str)
    seg["case_id"] = seg["case_id"].astype(str)
    seg["include_for_embedding"] = seg["include_for_embedding"].astype(bool)

    key = labels[["pmcid", "case_id"]].drop_duplicates()
    seg = seg.merge(key, on=["pmcid", "case_id"], how="inner")
    seg = seg[seg["include_for_embedding"]].copy()
    seg = seg[seg["text_clean"].notna()].copy()
    seg["text_embed"] = seg["text_clean"].astype(str).map(_text_for_embedding)

    # Build a flow table (manuscript-ready cohort accounting)
    flow_rows: list[dict[str, Any]] = []

    def _count_cases(df_labels: pd.DataFrame) -> int:
        return int(df_labels[["pmcid", "case_id"]].drop_duplicates().shape[0])

    def _count_by_split(df_labels: pd.DataFrame, split: str) -> int:
        return int(df_labels[df_labels["split"].astype(str) == split][["pmcid", "case_id"]].drop_duplicates().shape[0])

    # Phase C totals (all case units available)
    flow_rows.append(
        {
            "step": "phase_c_total_cases",
            "n_total": _count_cases(labels_all),
            "n_discovery": _count_by_split(labels_all, "discovery") if "split" in labels_all.columns else None,
            "n_confirmation": _count_by_split(labels_all, "confirmation") if "split" in labels_all.columns else None,
        }
    )
    flow_rows.append(
        {
            "step": "phase_c_ftld_strict",
            "n_total": _count_cases(labels_all[labels_all["ftld_inclusion_tier"] == "ftld_strict"]),
            "n_discovery": _count_by_split(labels_all[labels_all["ftld_inclusion_tier"] == "ftld_strict"], "discovery")
            if "split" in labels_all.columns
            else None,
            "n_confirmation": _count_by_split(labels_all[labels_all["ftld_inclusion_tier"] == "ftld_strict"], "confirmation")
            if "split" in labels_all.columns
            else None,
        }
    )
    flow_rows.append(
        {
            "step": "phase_c_ftld_broad",
            "n_total": _count_cases(labels_all[labels_all["ftld_inclusion_tier"] == "ftld_broad"]),
            "n_discovery": _count_by_split(labels_all[labels_all["ftld_inclusion_tier"] == "ftld_broad"], "discovery")
            if "split" in labels_all.columns
            else None,
            "n_confirmation": _count_by_split(labels_all[labels_all["ftld_inclusion_tier"] == "ftld_broad"], "confirmation")
            if "split" in labels_all.columns
            else None,
        }
    )
    flow_rows.append(
        {
            "step": "phase_d_after_tier_filter",
            "n_total": _count_cases(labels),
            "n_discovery": _count_by_split(labels, "discovery") if "split" in labels.columns else None,
            "n_confirmation": _count_by_split(labels, "confirmation") if "split" in labels.columns else None,
        }
    )

    # Leakage must be zero for included segments.
    leak_sum = int(pd.to_numeric(seg["leakage_n_matches_clean"], errors="coerce").fillna(0).sum())
    if leak_sum != 0:
        raise RuntimeError(f"Leakage audit failed for embedded segments: total matches={leak_sum}")

    # Cases must have at least one embedding segment.
    counts = seg.groupby(["pmcid", "case_id"], as_index=False).size().rename(columns={"size": "n_segments"})
    labels = labels.merge(counts, on=["pmcid", "case_id"], how="left")
    labels["n_segments"] = pd.to_numeric(labels["n_segments"], errors="coerce").fillna(0).astype(int)
    labels = labels[labels["n_segments"] >= int(min_segments_per_case)].copy()
    if labels.empty:
        raise RuntimeError("No cases have embedding-eligible segments after filtering.")

    flow_rows.append(
        {
            "step": f"phase_d_min_segments_ge_{int(min_segments_per_case)}",
            "n_total": _count_cases(labels),
            "n_discovery": _count_by_split(labels, "discovery") if "split" in labels.columns else None,
            "n_confirmation": _count_by_split(labels, "confirmation") if "split" in labels.columns else None,
        }
    )

    # Canonical row order for analysis + embedding alignment.
    labels = labels.sort_values(["pmcid", "case_id"]).reset_index(drop=True)

    if bool(one_case_per_pmcid):
        # Deterministic dedupe to reduce within-paper clustering artifacts:
        # keep the lexicographically smallest case_id per PMCID (independent of labels/embeddings).
        labels = labels.sort_values(["pmcid", "case_id"]).groupby("pmcid", as_index=False).head(1).reset_index(drop=True)

    flow_rows.append(
        {
            "step": f"phase_d_one_case_per_pmcid_{bool(one_case_per_pmcid)}",
            "n_total": _count_cases(labels),
            "n_discovery": _count_by_split(labels, "discovery") if "split" in labels.columns else None,
            "n_confirmation": _count_by_split(labels, "confirmation") if "split" in labels.columns else None,
        }
    )

    # Restrict segments again to the kept cases.
    keep_key = labels[["pmcid", "case_id"]].drop_duplicates()
    seg = seg.merge(keep_key, on=["pmcid", "case_id"], how="inner")

    # Embed segments (primary model).
    seg_embeddings, embed_info = compute_sentence_transformer_embeddings(
        seg["text_embed"].tolist(),
        config.embeddings.model_name,
        normalize_embeddings=config.embeddings.normalize_embeddings,
        batch_size=config.embeddings.batch_size,
        device=config.embeddings.device,
    )

    # Aggregate segment embeddings -> case embeddings (mean; then L2 normalize).
    key_to_row = {
        (r.pmcid, r.case_id): int(i)
        for i, r in enumerate(labels[["pmcid", "case_id"]].itertuples(index=False))
    }
    dim = int(seg_embeddings.shape[1])
    case_emb = np.zeros((len(labels), dim), dtype=np.float32)
    case_counts = np.zeros(len(labels), dtype=np.int32)

    seg_case = list(zip(seg["pmcid"].astype(str).tolist(), seg["case_id"].astype(str).tolist()))
    for i, (pmcid, case_id) in enumerate(seg_case):
        j = key_to_row.get((pmcid, case_id))
        if j is None:
            continue
        case_emb[j] += seg_embeddings[i]
        case_counts[j] += 1
    if int((case_counts == 0).sum()) != 0:
        raise RuntimeError("Internal error: some cases have zero segments after filtering.")
    case_emb = case_emb / case_counts[:, None].astype(np.float32)
    norms = np.linalg.norm(case_emb, axis=1, keepdims=True) + 1e-12
    case_emb = case_emb / norms

    # Build analysis table
    doc_id = labels["pmcid"].astype(str) + "__" + labels["case_id"].astype(str)
    labels.insert(0, "doc_id", doc_id)

    # Split
    if "split" not in labels.columns:
        raise RuntimeError("Phase C labels must include a 'split' column (discovery vs confirmation).")
    disc_mask = labels["split"].astype(str) == "discovery"
    conf_mask = labels["split"].astype(str) == "confirmation"
    if not disc_mask.any() or not conf_mask.any():
        raise RuntimeError("Both discovery and confirmation splits must be non-empty.")

    X_full = case_emb
    X_disc_full = X_full[disc_mask.to_numpy()]
    X_conf_full = X_full[conf_mask.to_numpy()]

    # PCA fit rule
    pca_fit_on = str(pca_fit_on).strip().lower()
    if pca_fit_on not in {"discovery", "all"}:
        raise ValueError("pca_fit_on must be one of: discovery|all")
    X_fit = X_disc_full if pca_fit_on == "discovery" else X_full

    X_fit_reduced, pca_fit = fit_pca_with_cap(
        X_fit,
        variance_threshold=config.dimensionality_reduction.pca_variance_threshold,
        max_components=config.dimensionality_reduction.pca_max_components,
    )

    def _transform(X: np.ndarray) -> np.ndarray:
        Xp = pca_fit.pca.transform(X)
        return Xp[:, : pca_fit.n_components_used]

    X_disc = _transform(X_disc_full)
    X_conf = _transform(X_conf_full)

    k = int(config.clustering.k_fixed)
    km_disc = fit_kmeans(X_disc, k=k, random_seed=config.random_seed)
    km_conf = fit_kmeans(X_conf, k=k, random_seed=config.random_seed)

    disc_labels = km_disc.labels.astype(int)
    conf_labels = km_conf.labels.astype(int)

    # Language cluster selection + discovery permutation (selection-aware).
    is_ppa_disc = labels.loc[disc_mask, "is_ppa"].astype(bool).to_numpy()
    is_ppa_conf = labels.loc[conf_mask, "is_ppa"].astype(bool).to_numpy()

    lang_disc, max_share_disc = _language_cluster(disc_labels, is_ppa_disc, k=k)
    p_disc = _selection_aware_max_share_p(
        labels=disc_labels,
        is_ppa=is_ppa_disc,
        k=k,
        permutations=config.inference.permutations,
        random_seed=config.random_seed,
        observed_max=max_share_disc,
    )

    disc_table, disc_lang_share = _enrichment_table(
        labels=disc_labels, is_ppa=is_ppa_disc, target_cluster=lang_disc
    )
    disc_effects = compute_effect_sizes(disc_table)

    # Match clusters by centroid cosine similarity (full embedding space).
    disc_centroids = np.vstack([X_disc_full[disc_labels == cl].mean(axis=0) for cl in range(k)]).astype(np.float32)
    conf_centroids = np.vstack([X_conf_full[conf_labels == cl].mean(axis=0) for cl in range(k)]).astype(np.float32)
    sim = _cosine_sim_matrix(disc_centroids, conf_centroids)
    disc_to_conf = _best_cluster_match(sim)
    conf_to_disc = {v: k for k, v in disc_to_conf.items()}

    lang_conf = int(disc_to_conf[lang_disc])
    target_mask_conf = conf_labels == lang_conf
    conf_lang_share = float(is_ppa_conf[target_mask_conf].mean()) if target_mask_conf.any() else 0.0
    conf_table, _ = _enrichment_table(labels=conf_labels, is_ppa=is_ppa_conf, target_cluster=lang_conf)
    conf_effects = compute_effect_sizes(conf_table)
    p_conf = _fixed_cluster_share_p(
        target_mask=target_mask_conf,
        is_ppa=is_ppa_conf,
        permutations=config.inference.permutations,
        random_seed=config.random_seed,
        observed=conf_lang_share,
    )

    # Write outputs
    case_table = labels.copy()
    case_table["cluster_split"] = np.nan
    case_table.loc[disc_mask, "cluster_split"] = disc_labels
    case_table.loc[conf_mask, "cluster_split"] = conf_labels
    case_table["cluster_split"] = case_table["cluster_split"].astype(int)
    case_table["cluster_matched_to_discovery"] = case_table["cluster_split"].map(lambda c: conf_to_disc.get(int(c), int(c)))  # for confirmation, remap; discovery stays identity
    case_table.loc[disc_mask, "cluster_matched_to_discovery"] = case_table.loc[disc_mask, "cluster_split"]

    # Case embeddings artifact
    case_embeddings_npz = run_dir / "case_embeddings.npz"
    np.savez_compressed(
        case_embeddings_npz,
        doc_id=np.asarray(case_table["doc_id"].astype(str).tolist(), dtype=str),
        pmcid=np.asarray(case_table["pmcid"].astype(str).tolist(), dtype=str),
        case_id=np.asarray(case_table["case_id"].astype(str).tolist(), dtype=str),
        split=np.asarray(case_table["split"].astype(str).tolist(), dtype=str),
        embeddings=X_full.astype(np.float32),
    )

    # Cluster similarity table (k x k)
    sim_rows = []
    for i in range(k):
        for j in range(k):
            sim_rows.append(
                {
                    "discovery_cluster": int(i),
                    "confirmation_cluster": int(j),
                    "cosine_similarity": float(sim[i, j]),
                    "matched": bool(disc_to_conf.get(int(i)) == int(j)),
                }
            )
    df_sim = pd.DataFrame(sim_rows)
    cluster_similarity_csv = tables_dir / "cluster_similarity_matrix.csv"
    df_sim.to_csv(cluster_similarity_csv, index=False)

    # Cluster summaries
    def _cluster_summary(split_name: str, cluster_ids: np.ndarray, is_ppa: np.ndarray) -> pd.DataFrame:
        rows = []
        n_total = int(len(cluster_ids))
        ppa_total = int(is_ppa.sum())
        base_share = float(ppa_total / n_total) if n_total else 0.0
        for cl in range(k):
            mask = cluster_ids == cl
            n = int(mask.sum())
            ppa = int(is_ppa[mask].sum())
            share = float(ppa / n) if n else 0.0
            rows.append(
                {
                    "split": split_name,
                    "cluster": int(cl),
                    "n_cases": n,
                    "ppa_cases": ppa,
                    "ppa_share": share,
                    "ppa_share_split": base_share,
                }
            )
        return pd.DataFrame(rows).sort_values(["split", "cluster"])

    df_disc_sum = _cluster_summary("discovery", disc_labels, is_ppa_disc)
    df_conf_sum = _cluster_summary("confirmation", conf_labels, is_ppa_conf)
    discovery_cluster_summary_csv = tables_dir / "cluster_summary_discovery.csv"
    confirmation_cluster_summary_csv = tables_dir / "cluster_summary_confirmation.csv"
    df_disc_sum.to_csv(discovery_cluster_summary_csv, index=False)
    df_conf_sum.to_csv(confirmation_cluster_summary_csv, index=False)

    # Symptom tag characterization (exploratory)
    tag_cols = [c for c in labels.columns if c.startswith("tag__") and c.endswith("__status")]
    def _tag_summary(split_mask: pd.Series, clusters: np.ndarray, split_name: str) -> pd.DataFrame:
        sub = labels.loc[split_mask, ["doc_id"] + tag_cols].copy()
        sub["cluster"] = clusters
        out_rows = []
        for tag_col in tag_cols:
            tag = tag_col.split("__", 2)[1]
            for cl in range(k):
                cl_rows = sub[sub["cluster"] == cl]
                if cl_rows.empty:
                    continue
                vc = cl_rows[tag_col].value_counts(dropna=False)
                n = int(len(cl_rows))
                present = int(vc.get("present", 0))
                out_rows.append(
                    {
                        "split": split_name,
                        "cluster": int(cl),
                        "tag": tag,
                        "n_cases": n,
                        "present_n": present,
                        "present_prop": float(present / n) if n else 0.0,
                        "explicitly_absent_n": int(vc.get("explicitly_absent", 0)),
                        "uncertain_n": int(vc.get("uncertain", 0)),
                        "not_reported_n": int(vc.get("not_reported", 0)),
                    }
                )
        return pd.DataFrame(out_rows).sort_values(["split", "cluster", "tag"])

    df_tags_disc = _tag_summary(disc_mask, disc_labels, "discovery")
    df_tags_conf = _tag_summary(conf_mask, conf_labels, "confirmation")
    symptom_tags_discovery_csv = tables_dir / "symptom_tags_by_cluster_discovery.csv"
    symptom_tags_confirmation_csv = tables_dir / "symptom_tags_by_cluster_confirmation.csv"
    df_tags_disc.to_csv(symptom_tags_discovery_csv, index=False)
    df_tags_conf.to_csv(symptom_tags_confirmation_csv, index=False)

    # Syndrome-level replication (secondary; multiplicity-controlled)
    syndrome_rows: list[dict[str, object]] = []

    def _syndrome_endpoint_row(endpoint: str) -> dict[str, object]:
        y_disc = (labels.loc[disc_mask, "ftld_syndrome_reported"].astype(str).to_numpy() == str(endpoint))
        y_conf = (labels.loc[conf_mask, "ftld_syndrome_reported"].astype(str).to_numpy() == str(endpoint))

        disc_cluster, disc_max_share = _language_cluster(disc_labels, y_disc, k=k)
        disc_p = _selection_aware_max_share_p(
            labels=disc_labels,
            is_ppa=y_disc,
            k=k,
            permutations=config.inference.permutations,
            random_seed=config.random_seed,
            observed_max=disc_max_share,
        )
        disc_table, _ = _enrichment_table(labels=disc_labels, is_ppa=y_disc, target_cluster=disc_cluster)
        disc_eff = compute_effect_sizes(disc_table)

        conf_cluster = int(disc_to_conf[int(disc_cluster)])
        conf_mask_cluster = conf_labels == int(conf_cluster)
        conf_share = float(y_conf[conf_mask_cluster].mean()) if conf_mask_cluster.any() else 0.0
        conf_table, _ = _enrichment_table(labels=conf_labels, is_ppa=y_conf, target_cluster=conf_cluster)
        conf_eff = compute_effect_sizes(conf_table)
        conf_p = _fixed_cluster_share_p(
            target_mask=conf_mask_cluster,
            is_ppa=y_conf,
            permutations=config.inference.permutations,
            random_seed=config.random_seed,
            observed=conf_share,
        )

        n_disc = int(len(y_disc))
        n_conf = int(len(y_conf))
        n_pos_disc = int(y_disc.sum())
        n_pos_conf = int(y_conf.sum())
        n_disc_cluster = int((disc_labels == int(disc_cluster)).sum())
        n_conf_cluster = int(conf_mask_cluster.sum())
        n_pos_disc_cluster = int((y_disc & (disc_labels == int(disc_cluster))).sum())
        n_pos_conf_cluster = int((y_conf & conf_mask_cluster).sum())

        return {
            "endpoint": str(endpoint),
            "k_fixed": int(k),
            "pca_fit_on": str(pca_fit_on),
            "n_discovery": n_disc,
            "n_confirmation": n_conf,
            "n_positive_discovery": n_pos_disc,
            "n_positive_confirmation": n_pos_conf,
            "discovery_cluster_max_share": int(disc_cluster),
            "confirmation_cluster_matched": int(conf_cluster),
            "discovery_positive_share_max_cluster": float(disc_max_share),
            "discovery_perm_p_value_selection_aware": float(disc_p),
            "confirmation_positive_share_matched_cluster": float(conf_share),
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
            "discovery_positive_share_split": float(disc_eff.ppa_share_stage),
            "confirmation_positive_share_split": float(conf_eff.ppa_share_stage),
            "discovery_cluster_n": n_disc_cluster,
            "confirmation_cluster_n": n_conf_cluster,
            "discovery_cluster_positive_n": n_pos_disc_cluster,
            "confirmation_cluster_positive_n": n_pos_conf_cluster,
            "permutations_B": int(config.inference.permutations),
        }

    for endpoint in SYNDROME_REPLICATION_ENDPOINTS_V1:
        syndrome_rows.append(_syndrome_endpoint_row(endpoint))

    syndrome_df = pd.DataFrame(syndrome_rows)
    rejected, q_vals = bh_fdr(
        syndrome_df["confirmation_perm_p_value_fixed_cluster"].astype(float).tolist(),
        alpha=float(config.inference.family_fdr_q),
    )
    syndrome_df["confirmation_q_bh_fdr"] = q_vals
    syndrome_df["confirmation_reject_bh_fdr"] = rejected.astype(bool)
    syndrome_replication_summary_csv = tables_dir / "syndrome_replication_summary.csv"
    syndrome_df.to_csv(syndrome_replication_summary_csv, index=False)

    # Cluster characterization (exploratory; descriptive + effect sizes; aligned to discovery-cluster IDs)
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

    def _two_by_two_effects(a: int, n_cluster: int, split_a: int, split_n: int) -> tuple[float, float]:
        b = int(n_cluster - a)
        c = int(split_a - a)
        d = int((split_n - n_cluster) - c)
        eff = compute_effect_sizes(TwoByTwo(a=float(a), b=float(b), c=float(c), d=float(d)))
        return float(eff.rr), float(eff.or_)

    char_rows: list[dict[str, object]] = []
    case_char = case_table.copy()
    case_char["matched_cluster"] = case_char["cluster_matched_to_discovery"].astype(int)

    for split_name in ["discovery", "confirmation"]:
        sdf = case_char[case_char["split"].astype(str) == split_name].copy()
        split_n = int(len(sdf))
        if split_n == 0:
            continue

        matched = sdf["matched_cluster"].astype(int).to_numpy()

        modality_sets = [set(_parse_json_list(x)) for x in sdf.get("imaging_modalities_json", pd.Series([None] * split_n)).tolist()]
        region_sets = [set(_parse_json_list(x)) for x in sdf.get("imaging_regions_json", pd.Series([None] * split_n)).tolist()]
        pathology_sets = [set(_parse_json_list(x)) for x in sdf.get("pathology_types_json", pd.Series([None] * split_n)).tolist()]
        gene_sets_raw = [set(_parse_json_list(x)) for x in sdf.get("genes_reported_json", pd.Series([None] * split_n)).tolist()]
        gene_sets = []
        for genes in gene_sets_raw:
            norm = set()
            for g in genes:
                gg = str(g).strip()
                if not gg:
                    continue
                up = gg.upper()
                # normalize common gene spellings
                if up.replace("-", "").replace("_", "") in {"C9ORF72", "C9ORF"}:
                    up = "C9ORF72"
                norm.add(up)
            gene_sets.append(norm)

        laterality = sdf.get("imaging_laterality", pd.Series([None] * split_n)).fillna("unknown").astype(str).tolist()
        genetics_status = sdf.get("genetics_status", pd.Series([None] * split_n)).fillna("not_reported").astype(str).tolist()
        neuropath_status = sdf.get("neuropath_status", pd.Series([None] * split_n)).fillna("not_reported").astype(str).tolist()

        def _add_family_value_rows(family: str, value: str, has_value: list[bool]) -> None:
            split_a = int(sum(bool(x) for x in has_value))
            split_prop = float(split_a / split_n) if split_n else 0.0
            for cl in range(k):
                cl_mask = matched == int(cl)
                n_cluster = int(cl_mask.sum())
                if n_cluster == 0:
                    continue
                a = int(sum(bool(has_value[i]) for i in range(split_n) if cl_mask[i]))
                prop = float(a / n_cluster) if n_cluster else 0.0
                rr, or_ = _two_by_two_effects(a, n_cluster, split_a, split_n)
                char_rows.append(
                    {
                        "split": split_name,
                        "matched_cluster": int(cl),
                        "family": family,
                        "value": str(value),
                        "n_cases": n_cluster,
                        "value_n": int(a),
                        "value_prop": float(prop),
                        "value_prop_split": float(split_prop),
                        "rr_vs_rest": float(rr),
                        "or_vs_rest": float(or_),
                    }
                )

        for mod in IMAGING_MODALITIES_V1:
            _add_family_value_rows("imaging_modality", mod, [mod in s for s in modality_sets])
        for reg in IMAGING_REGIONS_V1:
            _add_family_value_rows("imaging_region", reg, [reg in s for s in region_sets])
        for lat in IMAGING_LATERALITY_V1:
            _add_family_value_rows("imaging_laterality", lat, [str(x) == lat for x in laterality])
        for gs in GENETICS_STATUS_V1:
            _add_family_value_rows("genetics_status", gs, [str(x) == gs for x in genetics_status])
        for ns in NEUROPATH_STATUS_V1:
            _add_family_value_rows("neuropath_status", ns, [str(x) == ns for x in neuropath_status])
        for pt in PATHOLOGY_TYPES_V1:
            _add_family_value_rows("pathology_type", pt, [pt in s for s in pathology_sets])
        for gene in KEY_GENES_V1:
            _add_family_value_rows("gene_reported", gene, [gene in s for s in gene_sets])

    cluster_characterization_csv = tables_dir / "cluster_characterization_by_matched_cluster.csv"
    pd.DataFrame(char_rows).to_csv(cluster_characterization_csv, index=False)

    syndrome_comp = (
        case_char.groupby(["split", "matched_cluster", "ftld_syndrome_reported"], dropna=False)
        .size()
        .reset_index(name="n_cases")
    )
    syndrome_comp["cluster_n"] = syndrome_comp.groupby(["split", "matched_cluster"])["n_cases"].transform("sum")
    syndrome_comp["prop_within_cluster"] = syndrome_comp["n_cases"] / syndrome_comp["cluster_n"]
    syndrome_comp = syndrome_comp.sort_values(["split", "matched_cluster", "n_cases"], ascending=[True, True, False])
    syndrome_composition_csv = tables_dir / "syndrome_composition_by_matched_cluster.csv"
    syndrome_comp.to_csv(syndrome_composition_csv, index=False)

    tag_rows: list[dict[str, object]] = []
    for split_name in ["discovery", "confirmation"]:
        sdf = case_char[case_char["split"].astype(str) == split_name].copy()
        for cl in range(k):
            cdf = sdf[sdf["matched_cluster"] == int(cl)]
            if cdf.empty:
                continue
            for tag_col in tag_cols:
                tag = tag_col.split("__", 2)[1]
                vc = cdf[tag_col].value_counts(dropna=False)
                n = int(len(cdf))
                tag_rows.append(
                    {
                        "split": split_name,
                        "matched_cluster": int(cl),
                        "tag": tag,
                        "n_cases": n,
                        "present_n": int(vc.get("present", 0)),
                        "present_prop": float(int(vc.get("present", 0)) / n) if n else 0.0,
                        "explicitly_absent_n": int(vc.get("explicitly_absent", 0)),
                        "uncertain_n": int(vc.get("uncertain", 0)),
                        "not_reported_n": int(vc.get("not_reported", 0)),
                    }
                )
    symptom_tags_matched_csv = tables_dir / "symptom_tags_by_matched_cluster.csv"
    pd.DataFrame(tag_rows).sort_values(["split", "matched_cluster", "tag"]).to_csv(
        symptom_tags_matched_csv, index=False
    )

    # Numeric summaries (descriptive only)
    numeric_rows: list[dict[str, object]] = []
    for split_name in ["discovery", "confirmation"]:
        sdf = case_char[case_char["split"].astype(str) == split_name].copy()
        if sdf.empty:
            continue
        for cl in range(k):
            cdf = sdf[sdf["matched_cluster"] == int(cl)]
            for col in ["symptom_duration_months", "age_at_onset_years", "age_at_presentation_years"]:
                if col not in cdf.columns:
                    continue
                vals = pd.to_numeric(cdf[col], errors="coerce").dropna().astype(float)
                numeric_rows.append(
                    {
                        "split": split_name,
                        "matched_cluster": int(cl),
                        "variable": col,
                        "n_cases": int(len(cdf)),
                        "n_nonmissing": int(len(vals)),
                        "mean": float(vals.mean()) if len(vals) else float("nan"),
                        "sd": float(vals.std(ddof=1)) if len(vals) > 1 else float("nan"),
                        "median": float(vals.median()) if len(vals) else float("nan"),
                        "p25": float(vals.quantile(0.25)) if len(vals) else float("nan"),
                        "p75": float(vals.quantile(0.75)) if len(vals) else float("nan"),
                    }
                )

    numeric_characterization_csv = tables_dir / "numeric_characterization_by_matched_cluster.csv"
    pd.DataFrame(numeric_rows).to_csv(numeric_characterization_csv, index=False)

    # Replication summary table
    replication_summary = pd.DataFrame(
        [
            {
                "k_fixed": k,
                "pca_fit_on": pca_fit_on,
                "n_discovery": int(len(is_ppa_disc)),
                "n_confirmation": int(len(is_ppa_conf)),
                "language_cluster_discovery": int(lang_disc),
                "language_cluster_confirmation_matched": int(lang_conf),
                "discovery_max_ppa_share": float(max_share_disc),
                "discovery_perm_p_value_selection_aware": float(p_disc),
                "confirmation_ppa_share_matched_cluster": float(conf_lang_share),
                "confirmation_perm_p_value_fixed_cluster": float(p_conf),
                "discovery_rr": float(disc_effects.rr),
                "discovery_rr_ci95_low": float(disc_effects.rr_ci95_low),
                "discovery_rr_ci95_high": float(disc_effects.rr_ci95_high),
                "discovery_or": float(disc_effects.or_),
                "discovery_or_ci95_low": float(disc_effects.or_ci95_low),
                "discovery_or_ci95_high": float(disc_effects.or_ci95_high),
                "confirmation_rr": float(conf_effects.rr),
                "confirmation_rr_ci95_low": float(conf_effects.rr_ci95_low),
                "confirmation_rr_ci95_high": float(conf_effects.rr_ci95_high),
                "confirmation_or": float(conf_effects.or_),
                "confirmation_or_ci95_low": float(conf_effects.or_ci95_low),
                "confirmation_or_ci95_high": float(conf_effects.or_ci95_high),
                "permutations_B": int(config.inference.permutations),
            }
        ]
    )
    replication_summary_csv = tables_dir / "replication_summary.csv"
    replication_summary.to_csv(replication_summary_csv, index=False)

    # Full case table (analysis-ready)
    case_table_csv = tables_dir / "case_table_with_clusters.csv"
    case_table.to_csv(case_table_csv, index=False)

    # Cohort flow table
    flow_table_csv = tables_dir / "flow_table.csv"
    pd.DataFrame(flow_rows).to_csv(flow_table_csv, index=False)

    # Run config
    run_config = {
        "run_id": run_id,
        "created_utc": run_id,
        "segments_csv": str(segments_csv),
        "case_labels_csv": str(case_labels_csv),
        "include_broad": bool(include_broad),
        "pca_fit_on": pca_fit_on,
        "min_segments_per_case": int(min_segments_per_case),
        "one_case_per_pmcid": bool(one_case_per_pmcid),
        "k_fixed": k,
        "permutations": int(config.inference.permutations),
        "random_seed": int(config.random_seed),
        "embedding": {
            "model_name": embed_info.model_name,
            "embedding_dim": embed_info.embedding_dim,
            "device": embed_info.device,
            "sentence_transformers_version": embed_info.sentence_transformers_version,
            "transformers_version": embed_info.transformers_version,
            "torch_version": embed_info.torch_version,
            "model_commit_hash": embed_info.model_commit_hash,
        },
        "pca": {
            "variance_threshold": float(config.dimensionality_reduction.pca_variance_threshold),
            "max_components": int(config.dimensionality_reduction.pca_max_components),
            "n_components_used": int(pca_fit.n_components_used),
            "explained_variance_ratio": [float(x) for x in pca_fit.explained_variance_ratio.tolist()],
        },
        "counts": {
            "n_cases_total": int(len(case_table)),
            "n_discovery": int(disc_mask.sum()),
            "n_confirmation": int(conf_mask.sum()),
        },
        "matching": {"discovery_to_confirmation": {str(k): int(v) for k, v in disc_to_conf.items()}},
    }
    run_config_json = run_dir / "run_config.json"
    run_config_json.write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    # Markdown performance report
    perf_lines: list[str] = []
    perf_lines.append("# Phase D (PMC OA) — Split Analysis Report")
    perf_lines.append("")
    perf_lines.append(f"Run: {run_id}")
    perf_lines.append("")
    perf_lines.append("## Cohort")
    perf_lines.append(f"- Cases analyzed: {int(len(case_table))}")
    perf_lines.append(f"- Discovery: {int(disc_mask.sum())}")
    perf_lines.append(f"- Confirmation: {int(conf_mask.sum())}")
    perf_lines.append(f"- Min segments per case: {int(min_segments_per_case)}")
    perf_lines.append(f"- One case per PMCID: {bool(one_case_per_pmcid)}")
    perf_lines.append("")
    perf_lines.append("## Discovery (selection-aware)")
    perf_lines.append(f"- Language cluster (max PPA share): {lang_disc} (share={max_share_disc:.3f})")
    perf_lines.append(f"- Permutation p-value (max-stat): {p_disc:.6f} (B={int(config.inference.permutations)})")
    perf_lines.append(f"- RR={disc_effects.rr:.3f}, OR={disc_effects.or_:.3f}")
    perf_lines.append("")
    perf_lines.append("## Confirmation (matched cluster)")
    perf_lines.append(f"- Matched cluster: {lang_conf} (share={conf_lang_share:.3f})")
    perf_lines.append(f"- Permutation p-value (fixed cluster): {p_conf:.6f} (B={int(config.inference.permutations)})")
    perf_lines.append(f"- RR={conf_effects.rr:.3f}, OR={conf_effects.or_:.3f}")
    perf_lines.append("")
    perf_lines.append("## Syndrome-level Replication (secondary)")
    perf_lines.append(
        f"- Endpoints: {', '.join(SYNDROME_REPLICATION_ENDPOINTS_V1)} (BH-FDR q={float(config.inference.family_fdr_q):.2f})"
    )
    perf_lines.append(
        "- Table columns are selection-aware p in Discovery and fixed-cluster p in Confirmation (cluster matched by centroid similarity)."
    )
    for _, r in syndrome_df.sort_values("confirmation_perm_p_value_fixed_cluster").iterrows():
        perf_lines.append(
            f"- {r['endpoint']}: disc share={float(r['discovery_positive_share_max_cluster']):.3f}, "
            f"p={float(r['discovery_perm_p_value_selection_aware']):.4f}; "
            f"conf share={float(r['confirmation_positive_share_matched_cluster']):.3f}, "
            f"p={float(r['confirmation_perm_p_value_fixed_cluster']):.4f}, "
            f"q={float(r['confirmation_q_bh_fdr']):.4f}, "
            f"reject={bool(r['confirmation_reject_bh_fdr'])}"
        )
    perf_lines.append("")
    perf_lines.append("## Outputs")
    perf_lines.append(f"- Flow table: {flow_table_csv}")
    perf_lines.append(f"- Case table: {case_table_csv}")
    perf_lines.append(f"- Replication summary: {replication_summary_csv}")
    perf_lines.append(f"- Syndrome replication (secondary): {syndrome_replication_summary_csv}")
    perf_lines.append(f"- Cluster characterization (matched clusters): {cluster_characterization_csv}")
    perf_lines.append(f"- Numeric characterization (matched clusters): {numeric_characterization_csv}")
    perf_lines.append(f"- Syndrome composition (matched clusters): {syndrome_composition_csv}")
    perf_lines.append(f"- Cluster similarity matrix: {cluster_similarity_csv}")
    perf_lines.append(f"- Symptom tags (discovery): {symptom_tags_discovery_csv}")
    perf_lines.append(f"- Symptom tags (confirmation): {symptom_tags_confirmation_csv}")
    perf_lines.append(f"- Symptom tags (matched clusters): {symptom_tags_matched_csv}")
    performance_report_md = run_dir / "performance_report.md"
    performance_report_md.write_text("\n".join(perf_lines) + "\n", encoding="utf-8")

    # Optional convenience copy
    if write_latest_outputs:
        latest_dir = ensure_dir(run_dir.parent / "latest")
        (latest_dir / "performance_report.md").write_text(
            performance_report_md.read_text(encoding="utf-8"), encoding="utf-8"
        )

    return PhaseDOutputs(
        run_dir=run_dir,
        run_config_json=run_config_json,
        flow_table_csv=flow_table_csv,
        case_table_csv=case_table_csv,
        replication_summary_csv=replication_summary_csv,
        syndrome_replication_summary_csv=syndrome_replication_summary_csv,
        cluster_characterization_csv=cluster_characterization_csv,
        numeric_characterization_csv=numeric_characterization_csv,
        syndrome_composition_csv=syndrome_composition_csv,
        cluster_similarity_csv=cluster_similarity_csv,
        discovery_cluster_summary_csv=discovery_cluster_summary_csv,
        confirmation_cluster_summary_csv=confirmation_cluster_summary_csv,
        symptom_tags_discovery_csv=symptom_tags_discovery_csv,
        symptom_tags_confirmation_csv=symptom_tags_confirmation_csv,
        symptom_tags_matched_csv=symptom_tags_matched_csv,
        performance_report_md=performance_report_md,
        case_embeddings_npz=case_embeddings_npz,
    )
