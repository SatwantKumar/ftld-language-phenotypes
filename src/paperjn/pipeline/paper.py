from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from paperjn.analysis.early import run_early_centroid_assignment, run_early_diagnostics
from paperjn.analysis.literature import run_literature_replication
from paperjn.analysis.stability import run_bootstrap_stability, run_seed_sensitivity
from paperjn.clustering.pca import fit_pca_with_cap
from paperjn.config import ProjectConfig
from paperjn.embeddings.sentence_transformers_backend import compute_sentence_transformer_embeddings
from paperjn.nlp.leakage import remove_blacklisted_terms
from paperjn.nlp.text import normalize_text, normalize_whitespace
from paperjn.pipeline.make_curated import make_curated_table
from paperjn.pipeline.primary import run_audit_only, run_primary
from paperjn.reporting.plots import (
    plot_ppa_enrichment_forest,
    plot_ppa_share_bars,
    plot_ppa_share_by_cluster,
)
from paperjn.reporting.report import write_paper_markdown_report
from paperjn.utils.paths import ensure_dir, resolve_path


@dataclass(frozen=True)
class PaperPipelineOutputs:
    paper_run_id: str
    primary_meta: dict[str, object]
    sensitivity_metas: list[dict[str, object]]
    combined_table_path: Path
    report_path: Path
    figures: list[Path]


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


def run_paper_pipeline(
    config: ProjectConfig,
    *,
    config_path: Path,
    make_curated: bool = True,
    curated_source_csv: str = "../prepared_ftd_dataset.csv",
    overwrite_curated: bool = True,
) -> PaperPipelineOutputs:
    project_root = _project_root_from_config_path(config_path)
    results_dir = ensure_dir(resolve_path(project_root, config.paths.results_dir))
    tables_dir = ensure_dir(results_dir / "tables")
    figures_dir = ensure_dir(results_dir / "figures")

    paper_run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if make_curated:
        make_curated_table(
            config,
            config_path=config_path,
            input_csv=curated_source_csv,
            overwrite=overwrite_curated,
        )

    # Audits (fail fast if leakage persists after cleaning).
    run_audit_only(config, config_path=config_path)

    # Primary run (writes legacy + tagged outputs).
    primary_meta = run_primary(
        config,
        config_path=config_path,
        run_id=paper_run_id,
        output_tag="primary",
        write_latest_outputs=True,
    )

    sensitivity_metas: list[dict[str, object]] = []

    # Sensitivity: embedding family
    for model_name in config.embeddings.sensitivity_model_names:
        tag = _sanitize_tag(f"embed_{model_name}")
        cfg2 = config.model_copy(deep=True)
        cfg2.embeddings.model_name = model_name
        sensitivity_metas.append(
            run_primary(
                cfg2,
                config_path=config_path,
                run_id=paper_run_id,
                output_tag=tag,
                write_latest_outputs=False,
            )
        )

    # Sensitivity: k robustness (primary embedding)
    for k in config.clustering.k_sensitivity_values:
        if k == config.clustering.k_fixed:
            continue
        tag = _sanitize_tag(f"k{k}")
        cfg2 = config.model_copy(deep=True)
        cfg2.clustering.k_fixed = int(k)
        sensitivity_metas.append(
            run_primary(
                cfg2,
                config_path=config_path,
                run_id=paper_run_id,
                output_tag=tag,
                write_latest_outputs=False,
            )
        )

    # Combine stage-level results across runs.
    rows = []
    metas = [primary_meta] + sensitivity_metas
    for meta in metas:
        tag = str(meta.get("output_tag", "run"))
        out = meta.get("outputs", {})
        if not isinstance(out, dict):
            continue
        stage_path = out.get("ppa_enrichment_stage_tagged")
        if not stage_path:
            continue
        df_stage = pd.read_csv(stage_path)
        df_stage.insert(0, "run_tag", tag)
        rows.append(df_stage)

    combined = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    combined_path = tables_dir / f"ppa_enrichment_stage__paper_{paper_run_id}.csv"
    combined.to_csv(combined_path, index=False)

    # Figures (primary only)
    figures: list[Path] = []
    primary_stage_path = Path(primary_meta["outputs"]["ppa_enrichment_stage_tagged"])
    primary_assign_path = Path(primary_meta["outputs"]["cluster_assignments_tagged"])
    primary_run_dir = Path(primary_meta["outputs"]["run_dir"])
    df_stage_primary = pd.read_csv(primary_stage_path)
    df_assign_primary = pd.read_csv(primary_assign_path)

    figures.append(
        plot_ppa_enrichment_forest(
            df_stage_primary,
            out_path=figures_dir / f"fig_forest_rr__{paper_run_id}.png",
            title="PPA enrichment (RR; selection-aware permutation)",
        )
    )
    figures.append(
        plot_ppa_share_bars(
            df_stage_primary,
            out_path=figures_dir / f"fig_ppa_share__{paper_run_id}.png",
            title="PPA share: language cluster vs stage baseline",
        )
    )
    figures.append(
        plot_ppa_share_by_cluster(
            df_assign_primary,
            out_path=figures_dir / f"fig_ppa_share_by_cluster__{paper_run_id}.png",
            title="PPA share by cluster (k=4)",
        )
    )

    # Early diagnostics (null distribution context)
    early_diag = run_early_diagnostics(
        assignments_primary=df_assign_primary,
        stage_results_primary=df_stage_primary,
        permutations=config.inference.permutations,
        random_seed=config.random_seed,
        out_table=tables_dir / f"early_diagnostics__{paper_run_id}.csv",
        out_figure=figures_dir / f"fig_early_null_max_share__{paper_run_id}.png",
    )
    figures.append(early_diag.figure_path)

    # Early centroid assignment (supportive)
    curated_clean_path = primary_run_dir / "curated_clean.csv"
    embeddings_path = primary_run_dir / "embeddings.npz"
    curated_df = pd.read_csv(curated_clean_path)
    with np.load(embeddings_path, allow_pickle=False) as npz:
        curated_embeddings = np.asarray(npz["embeddings"], dtype=np.float32)

    early_centroid = run_early_centroid_assignment(
        curated_embeddings=curated_embeddings,
        curated_table=curated_df,
        assignments_primary=df_assign_primary,
        stage_results_primary=df_stage_primary,
        ppa_markers=config.phenotypes.ppa_markers,
        permutations=config.inference.permutations,
        random_seed=config.random_seed,
        out_summary=tables_dir / f"early_centroid_assignment_summary__{paper_run_id}.csv",
        out_assignments=tables_dir / f"early_centroid_assignment_assignments__{paper_run_id}.csv",
        out_figure=figures_dir / f"fig_early_centroid_assignment__{paper_run_id}.png",
    )
    figures.append(early_centroid.figure_path)

    # Stability analyses (subsample + seed sensitivity) on primary PCA representation
    X_reduced, _ = fit_pca_with_cap(
        curated_embeddings,
        variance_threshold=config.dimensionality_reduction.pca_variance_threshold,
        max_components=config.dimensionality_reduction.pca_max_components,
    )
    bootstrap = run_bootstrap_stability(
        X_reduced=X_reduced,
        assignments_primary=df_assign_primary,
        stage_results_primary=df_stage_primary,
        n_bootstrap=config.stability.n_bootstrap,
        subsample_fraction=config.stability.subsample_fraction,
        k_fixed=config.clustering.k_fixed,
        random_seed=config.random_seed,
        out_table=tables_dir / f"stability_bootstrap__{paper_run_id}.csv",
        out_fig_ari=figures_dir / f"fig_stability_ari__{paper_run_id}.png",
        out_fig_jaccard=figures_dir / f"fig_stability_jaccard__{paper_run_id}.png",
    )
    figures.extend([bootstrap.fig_ari_path, bootstrap.fig_jaccard_path])

    seed = run_seed_sensitivity(
        X_reduced=X_reduced,
        assignments_primary=df_assign_primary,
        stage_results_primary=df_stage_primary,
        seed_values=config.stability.seed_values,
        k_fixed=config.clustering.k_fixed,
        baseline_seed=config.random_seed,
        out_table=tables_dir / f"stability_seed__{paper_run_id}.csv",
    )

    df_boot = pd.read_csv(bootstrap.table_path)
    df_seed = pd.read_csv(seed.table_path)

    # External replication (literature) if corpus exists
    literature_summary_df = None
    literature_note = None
    corpus_path = resolve_path(project_root, config.literature.corpus_csv)
    if corpus_path.exists():
        corpus = pd.read_csv(corpus_path)

        # Reference centroids: Middle+Late clusters from curated primary run
        ref_rows = df_assign_primary[df_assign_primary["Temporal_Stage"].isin(["Middle", "Late"])].copy()
        ref_centroids = []
        ref_is_language = []
        lang_map = df_stage_primary.set_index("Temporal_Stage")["language_cluster"].astype(int).to_dict()
        for stage in ["Middle", "Late"]:
            for cl in sorted(ref_rows[ref_rows["Temporal_Stage"] == stage]["Cluster"].unique().tolist()):
                rows_idx = ref_rows[(ref_rows["Temporal_Stage"] == stage) & (ref_rows["Cluster"] == cl)][
                    "row"
                ].to_numpy()
                ref_centroids.append(curated_embeddings[rows_idx].mean(axis=0))
                ref_is_language.append(int(cl) == int(lang_map[stage]))
        ref_centroids = np.vstack(ref_centroids).astype(np.float32)
        ref_is_language = np.asarray(ref_is_language, dtype=bool)

        # Embed literature texts after applying the same normalization + blacklist redaction rules
        # used for the curated table, to prevent leakage into the embedding space.
        lit_text_norm = corpus[config.literature.text_column].astype(str).map(normalize_text)
        lit_text_redacted = lit_text_norm.map(
            lambda t: normalize_whitespace(remove_blacklisted_terms(t, config.leakage_blacklist, replacement=" "))
        )
        lit_embeddings, _ = compute_sentence_transformer_embeddings(
            lit_text_redacted.tolist(),
            config.embeddings.model_name,
            normalize_embeddings=config.embeddings.normalize_embeddings,
            batch_size=config.embeddings.batch_size,
            device=config.embeddings.device,
        )

        lit = run_literature_replication(
            corpus=corpus,
            text_column=config.literature.text_column,
            is_ppa_column=config.literature.is_ppa_column,
            doc_id_column=config.literature.doc_id_column,
            leakage_blacklist=config.leakage_blacklist,
            literature_embeddings=lit_embeddings,
            reference_centroids=ref_centroids,
            reference_is_language=ref_is_language,
            permutations=config.inference.permutations,
            random_seed=config.random_seed,
            out_summary=tables_dir / f"literature_replication_summary__{paper_run_id}.csv",
            out_assignments=tables_dir / f"literature_replication_assignments__{paper_run_id}.csv",
            out_figure=figures_dir / f"fig_literature_replication__{paper_run_id}.png",
        )
        figures.append(lit.figure_path)
        literature_summary_df = pd.read_csv(lit.summary_path)
    else:
        literature_note = f"Skipped (missing corpus): {config.literature.corpus_csv}"

    # Markdown report
    report_path = results_dir / f"report__{paper_run_id}.md"
    write_paper_markdown_report(
        paper_run_id=paper_run_id,
        primary_stage_results=df_stage_primary,
        combined_results=combined,
        stage_order=config.stages,
        early_diagnostics=pd.read_csv(early_diag.table_path),
        early_centroid_summary=pd.read_csv(early_centroid.summary_path),
        bootstrap_stability=df_boot,
        seed_stability=df_seed,
        literature_summary=literature_summary_df,
        literature_note=literature_note,
        out_path=report_path,
    )
    # Convenience "latest" pointer
    (results_dir / "report_latest.md").write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")

    return PaperPipelineOutputs(
        paper_run_id=paper_run_id,
        primary_meta=primary_meta,
        sensitivity_metas=sensitivity_metas,
        combined_table_path=combined_path,
        report_path=report_path,
        figures=figures,
    )
