from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from paperjn.clustering.kmeans import fit_kmeans
from paperjn.clustering.pca import fit_pca_with_cap
from paperjn.config import ProjectConfig
from paperjn.embeddings.sentence_transformers_backend import compute_sentence_transformer_embeddings
from paperjn.io.curated import CuratedTableSpec, load_curated_table
from paperjn.nlp.leakage import audit_text_for_leakage, remove_blacklisted_terms
from paperjn.nlp.text import normalize_text, normalize_whitespace
from paperjn.stats.enrichment import TwoByTwo, compute_effect_sizes, is_ppa_subtype
from paperjn.stats.fdr import bh_fdr
from paperjn.stats.permutation import permutation_p_value_ge
from paperjn.utils.paths import ensure_dir, resolve_path
from paperjn.utils.seed import set_global_seed


def _project_root_from_config_path(config_path: Path) -> Path:
    if config_path.parent.name == "configs":
        return config_path.parent.parent.resolve()
    return config_path.parent.resolve()

def _sanitize_tag(tag: str) -> str:
    safe = []
    for ch in tag:
        if ch.isalnum() or ch in {"-", "_", "."}:
            safe.append(ch)
        else:
            safe.append("_")
    out = "".join(safe).strip("._")
    return out or "run"


def run_primary(
    config: ProjectConfig,
    *,
    config_path: Path,
    run_id: str | None = None,
    output_tag: str = "primary",
    write_latest_outputs: bool = True,
) -> dict[str, object]:
    project_root = _project_root_from_config_path(config_path)
    set_global_seed(config.random_seed)

    curated_csv = resolve_path(project_root, config.paths.curated_input_csv)
    results_dir = ensure_dir(resolve_path(project_root, config.paths.results_dir))
    tables_dir = ensure_dir(results_dir / "tables")
    logs_dir = ensure_dir(results_dir / "logs")
    snapshots_dir = ensure_dir(results_dir / "snapshots")

    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = _sanitize_tag(output_tag)
    run_dir = ensure_dir(snapshots_dir / run_id / tag)
    run_tables_dir = ensure_dir(run_dir / "tables")
    run_figures_dir = ensure_dir(run_dir / "figures")

    spec = CuratedTableSpec(
        stage_col=config.columns.stage,
        subtype_col=config.columns.subtype,
        text_col=config.columns.text,
        stages=config.stages,
    )
    df = load_curated_table(str(curated_csv), spec)

    # Normalize and redact text.
    text_original = df[spec.text_col].astype(str)
    text_norm = text_original.map(normalize_text)
    text_redacted = text_norm.map(
        lambda t: normalize_whitespace(remove_blacklisted_terms(t, config.leakage_blacklist, replacement=" "))
    )

    leakage = text_redacted.map(lambda t: audit_text_for_leakage(t, config.leakage_blacklist))
    leakage_n = leakage.map(lambda r: r.n_matches)
    leakage_total = int(leakage_n.sum())

    leakage_report = pd.DataFrame(
        {
            "row": np.arange(len(df)),
            "temporal_stage": df[spec.stage_col].astype(str),
            "subtype": df[spec.subtype_col].astype(str),
            "n_matches": leakage_n,
        }
    )
    leakage_report_path = logs_dir / f"leakage_audit_{run_id}__{tag}.csv"
    leakage_report.to_csv(leakage_report_path, index=False)

    if leakage_total != 0:
        # Fail fast: primary analysis requires zero leakage.
        raise RuntimeError(
            f"Leakage audit failed: {leakage_total} blacklist matches remain after cleaning. "
            f"See {leakage_report_path}"
        )

    # Persist the cleaned curated table (snapshot artifact).
    cleaned_path = run_dir / "curated_clean.csv"
    df_clean = df.copy()
    df_clean["Description_original"] = text_original
    df_clean["Description_clean"] = text_redacted
    df_clean.to_csv(cleaned_path, index=False)

    # Embeddings (primary model only; sensitivity handled via separate runs).
    embeddings, embed_info = compute_sentence_transformer_embeddings(
        df_clean["Description_clean"].tolist(),
        config.embeddings.model_name,
        normalize_embeddings=config.embeddings.normalize_embeddings,
        batch_size=config.embeddings.batch_size,
        device=config.embeddings.device,
    )

    if embeddings.shape[0] != len(df_clean):
        raise RuntimeError("Embedding row count mismatch.")

    embeddings_path = run_dir / "embeddings.npz"
    np.savez_compressed(
        embeddings_path,
        embeddings=embeddings,
        row=np.arange(len(df_clean), dtype=np.int32),
    )

    # PCA reduction (fit once on all rows; applied within-stage).
    if config.dimensionality_reduction.method != "pca":
        raise ValueError("Only PCA dimensionality reduction is supported in the primary pipeline.")

    X_reduced, pca_fit = fit_pca_with_cap(
        embeddings,
        variance_threshold=config.dimensionality_reduction.pca_variance_threshold,
        max_components=config.dimensionality_reduction.pca_max_components,
    )

    pca_path = run_dir / "pca_fit.json"
    pca_payload = {
        "n_components_used": pca_fit.n_components_used,
        "explained_variance_ratio": pca_fit.explained_variance_ratio.tolist(),
        "variance_threshold": config.dimensionality_reduction.pca_variance_threshold,
        "max_components": config.dimensionality_reduction.pca_max_components,
    }
    pca_path.write_text(json.dumps(pca_payload, indent=2), encoding="utf-8")

    # Identify PPA subtypes from the Subtype column.
    subtype_series = df_clean[spec.subtype_col].astype(str)
    is_ppa = subtype_series.map(lambda s: is_ppa_subtype(s, config.phenotypes.ppa_markers)).to_numpy()

    # Stage-wise clustering + selection-aware permutation (combined PPA).
    stage_rows = []
    assignment_rows = []
    composition_rows = []

    for stage in config.stages:
        mask = df_clean[spec.stage_col].astype(str).to_numpy() == stage
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue

        X_stage = X_reduced[idx]
        subtypes_stage = subtype_series.iloc[idx].tolist()
        is_ppa_stage = is_ppa[idx]

        km = fit_kmeans(X_stage, k=config.clustering.k_fixed, random_seed=config.random_seed)
        labels = km.labels

        # cluster composition
        stage_df = pd.DataFrame(
            {
                "row": idx,
                "Temporal_Stage": stage,
                "Subtype": subtypes_stage,
                "Cluster": labels.astype(int),
                "is_ppa": is_ppa_stage.astype(bool),
            }
        )
        assignment_rows.append(stage_df)

        comp = (
            stage_df.groupby(["Temporal_Stage", "Cluster", "Subtype"], as_index=False)
            .size()
            .rename(columns={"size": "Count"})
        )
        comp["Cluster_Size"] = comp.groupby(["Temporal_Stage", "Cluster"])["Count"].transform("sum")
        comp["Proportion"] = comp["Count"] / comp["Cluster_Size"]
        composition_rows.append(comp)

        # language-dominant cluster = argmax PPA_share (tie -> smallest cluster id).
        ppa_by_cluster = (
            stage_df.groupby("Cluster")["is_ppa"].mean().reset_index().rename(columns={"is_ppa": "PPA_share"})
        )
        max_share = float(ppa_by_cluster["PPA_share"].max())
        language_cluster = int(ppa_by_cluster[ppa_by_cluster["PPA_share"] == max_share]["Cluster"].min())

        n_stage = int(len(stage_df))
        n_cluster = int((stage_df["Cluster"] == language_cluster).sum())
        ppa_stage = int(stage_df["is_ppa"].sum())
        ppa_cluster = int(stage_df.loc[stage_df["Cluster"] == language_cluster, "is_ppa"].sum())

        # 2x2 table vs outside-cluster
        a = ppa_cluster
        b = n_cluster - ppa_cluster
        c = ppa_stage - ppa_cluster
        d = (n_stage - n_cluster) - c

        effects = compute_effect_sizes(TwoByTwo(a=a, b=b, c=c, d=d))

        # Selection-aware permutation: permute subtype labels (equivalently permute is_ppa), select max.
        rng = np.random.default_rng(config.random_seed)
        perm_stats = np.zeros(config.inference.permutations, dtype=np.float32)
        for b_idx in range(config.inference.permutations):
            perm_is_ppa = rng.permutation(is_ppa_stage)
            # max PPA_share across clusters under permuted labels
            shares = []
            for cl in range(config.clustering.k_fixed):
                cl_mask = labels == cl
                if not np.any(cl_mask):
                    shares.append(0.0)
                else:
                    shares.append(float(np.mean(perm_is_ppa[cl_mask])))
            perm_stats[b_idx] = float(np.max(shares))

        p_value = permutation_p_value_ge(max_share, perm_stats)

        stage_rows.append(
            {
                "Temporal_Stage": stage,
                "k_fixed": config.clustering.k_fixed,
                "language_cluster": language_cluster,
                "n_stage": n_stage,
                "n_language_cluster": n_cluster,
                "ppa_stage": ppa_stage,
                "ppa_language_cluster": ppa_cluster,
                "ppa_share_language_cluster": float(effects.ppa_share_cluster),
                "ppa_share_stage": float(effects.ppa_share_stage),
                "rr": effects.rr,
                "rr_ci95_low": effects.rr_ci95_low,
                "rr_ci95_high": effects.rr_ci95_high,
                "or": effects.or_,
                "or_ci95_low": effects.or_ci95_low,
                "or_ci95_high": effects.or_ci95_high,
                "perm_p_value": p_value,
                "perm_B": int(config.inference.permutations),
            }
        )

    if not stage_rows:
        raise RuntimeError("No stages produced results. Check stage labels and input data.")

    df_stage = pd.DataFrame(stage_rows).sort_values("Temporal_Stage")
    rejected, q_values = bh_fdr(df_stage["perm_p_value"].tolist(), alpha=config.inference.family_fdr_q)
    df_stage["perm_q_value_bh"] = q_values
    df_stage["perm_rejected_bh"] = rejected

    df_assign = pd.concat(assignment_rows, ignore_index=True)
    df_comp = pd.concat(composition_rows, ignore_index=True)

    # Always write run-scoped outputs.
    stage_out_run = run_tables_dir / "ppa_enrichment_stage.csv"
    assign_out_run = run_tables_dir / "cluster_assignments.csv"
    comp_out_run = run_tables_dir / "cluster_composition_by_stage.csv"
    df_stage.to_csv(stage_out_run, index=False)
    df_assign.to_csv(assign_out_run, index=False)
    df_comp.to_csv(comp_out_run, index=False)

    # Also write tagged outputs into results/ for convenience.
    stage_out_tagged = tables_dir / f"ppa_enrichment_stage__{tag}.csv"
    assign_out_tagged = tables_dir / f"cluster_assignments__{tag}.csv"
    comp_out_tagged = tables_dir / f"cluster_composition_by_stage__{tag}.csv"
    df_stage.to_csv(stage_out_tagged, index=False)
    df_assign.to_csv(assign_out_tagged, index=False)
    df_comp.to_csv(comp_out_tagged, index=False)

    # Maintain legacy filenames pointing to the latest primary run.
    stage_out = tables_dir / "ppa_enrichment_stage.csv"
    assign_out = tables_dir / "cluster_assignments.csv"
    comp_out = tables_dir / "cluster_composition_by_stage.csv"
    if tag == "primary" and write_latest_outputs:
        df_stage.to_csv(stage_out, index=False)
        df_assign.to_csv(assign_out, index=False)
        df_comp.to_csv(comp_out, index=False)

    # Snapshot key artifacts for reviewer/audit trail.
    (run_dir / "config_source.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    (run_dir / "config_effective.json").write_text(
        json.dumps(config.model_dump(mode="json"), indent=2), encoding="utf-8"
    )
    meta = {
        "run_id": run_id,
        "output_tag": tag,
        "created_utc": run_id,
        "curated_input_csv": str(curated_csv),
        "cleaned_table_csv": str(cleaned_path),
        "embeddings_npz": str(embeddings_path),
        "primary_embedding": asdict(embed_info),
        "pca": pca_payload,
        "outputs": {
            "run_dir": str(run_dir),
            "run_tables_dir": str(run_tables_dir),
            "run_figures_dir": str(run_figures_dir),
            "ppa_enrichment_stage_run": str(stage_out_run),
            "cluster_assignments_run": str(assign_out_run),
            "cluster_composition_by_stage_run": str(comp_out_run),
            "ppa_enrichment_stage_tagged": str(stage_out_tagged),
            "cluster_assignments_tagged": str(assign_out_tagged),
            "cluster_composition_by_stage_tagged": str(comp_out_tagged),
            "ppa_enrichment_stage_latest": str(stage_out),
            "cluster_assignments_latest": str(assign_out),
            "cluster_composition_by_stage_latest": str(comp_out),
            "leakage_audit": str(leakage_report_path),
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return meta


def run_audit_only(config: ProjectConfig, *, config_path: Path) -> Path:
    project_root = _project_root_from_config_path(config_path)
    curated_csv = resolve_path(project_root, config.paths.curated_input_csv)
    results_dir = ensure_dir(resolve_path(project_root, config.paths.results_dir))
    logs_dir = ensure_dir(results_dir / "logs")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    spec = CuratedTableSpec(
        stage_col=config.columns.stage,
        subtype_col=config.columns.subtype,
        text_col=config.columns.text,
        stages=config.stages,
    )
    df = load_curated_table(str(curated_csv), spec)

    text_norm = df[spec.text_col].astype(str).map(normalize_text)
    text_redacted = text_norm.map(
        lambda t: normalize_whitespace(remove_blacklisted_terms(t, config.leakage_blacklist, replacement=" "))
    )
    leakage = text_redacted.map(lambda t: audit_text_for_leakage(t, config.leakage_blacklist))
    leakage_n = leakage.map(lambda r: r.n_matches)

    leakage_report = pd.DataFrame(
        {
            "row": np.arange(len(df)),
            "temporal_stage": df[spec.stage_col].astype(str),
            "subtype": df[spec.subtype_col].astype(str),
            "n_matches": leakage_n,
        }
    )
    out = logs_dir / f"leakage_audit_{run_id}.csv"
    leakage_report.to_csv(out, index=False)
    return out
