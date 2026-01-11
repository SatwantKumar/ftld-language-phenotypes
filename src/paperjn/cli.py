from __future__ import annotations

import inspect
import json
from datetime import datetime, timezone
from pathlib import Path

import click
import pandas as pd
import typer
import yaml
from rich.console import Console

from paperjn.config import load_config
from paperjn.citations import (
    PubMedError,
    extract_cite_keys,
    format_ama_journal_reference,
    pubmed_fetch_articles,
    pubmed_search_pmids,
    replace_cite_tags,
    resolve_doi_to_pmid,
)
from paperjn.pmcoa.manifest import build_download_manifest
from paperjn.pmcoa.ncbi_eutils import make_client
from paperjn.pmcoa.jats import fetch_jats_from_registry
from paperjn.pmcoa.blocks_io import parse_jats_dir
from paperjn.pmcoa.extract_pipeline import extract_from_xml_dir
from paperjn.pmcoa.phase_c_labeling import label_phase_c_v1
from paperjn.pmcoa.phase_d_analysis import run_phase_d_pmcoa_split_analysis
from paperjn.pmcoa.phase_d_stability import run_phase_d_stability
from paperjn.pmcoa.insilico_study import run_insilico_rater_study
from paperjn.pmcoa.rater_workbook import make_rater_workbooks
from paperjn.pmcoa.rater_subset import make_rater_subset_n
from paperjn.pmcoa.phase_e_pooling import run_phase_e_pool_after_replication
from paperjn.pmcoa.rater_validation import make_rater_sample, score_two_raters
from paperjn.pmcoa.results_packet import build_phase_d_results_packet
from paperjn.pmcoa.registry import SplitRecommendation, build_registry, load_query_file, recommend_time_split
from paperjn.pipeline.make_curated import make_curated_table
from paperjn.pipeline.paper import run_paper_pipeline
from paperjn.pipeline.primary import run_audit_only, run_primary
from paperjn.utils.dotenv import load_dotenv_candidates

# Typer 0.15.2 + Click 8.2 compatibility: Typer rich help calls `make_metavar()`
# without a Context, but Click 8.2 requires it. Patch to keep `--help` working.
_make_metavar_sig = inspect.signature(click.core.Parameter.make_metavar)
if len(_make_metavar_sig.parameters) == 2:  # (self, ctx) -> str
    _orig_make_metavar = click.core.Parameter.make_metavar

    def _make_metavar_compat(self: click.Parameter, ctx: click.Context | None = None) -> str:  # type: ignore[misc]
        if ctx is None:
            ctx = click.get_current_context(silent=True)
        if ctx is None:
            ctx = click.Context(click.Command("paperjn"))
        return _orig_make_metavar(self, ctx)

    click.core.Parameter.make_metavar = _make_metavar_compat  # type: ignore[assignment]

app = typer.Typer(no_args_is_help=True)
console = Console()
pmcoa_app = typer.Typer(no_args_is_help=True)
app.add_typer(pmcoa_app, name="pmcoa")
citations_app = typer.Typer(no_args_is_help=True)
app.add_typer(citations_app, name="citations")


@app.command()
def audit(
    config: str = typer.Option(..., "--config", "-c", help="Path to configs/project.yaml"),
) -> None:
    """Run schema + leakage audits for the curated table."""
    config_path = Path(config).resolve()
    cfg = load_config(config_path)
    out = run_audit_only(cfg, config_path=config_path)
    console.print(f"[green]OK[/green] audit written to: {out}")


@app.command("make-curated")
def make_curated(
    input_csv: str = typer.Option(
        "../prepared_ftd_dataset.csv",
        "--input",
        "-i",
        help="Source CSV to derive the curated table from (default: ../prepared_ftd_dataset.csv).",
    ),
    config: str = typer.Option(..., "--config", "-c", help="Path to configs/project.yaml"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing curated CSV."),
) -> None:
    """Create `data/raw/curated_ftd_table.csv` from current project data with leakage sanity checks."""
    config_path = Path(config).resolve()
    cfg = load_config(config_path)
    outputs = make_curated_table(cfg, config_path=config_path, input_csv=input_csv, overwrite=overwrite)
    console.print(f"[green]OK[/green] wrote: {outputs.curated_csv}")
    console.print(f"Leakage reports: {outputs.pre_leakage_report_csv}, {outputs.post_leakage_report_csv}")


@app.command("run")
def run_pipeline(
    which: str = typer.Argument(..., help="Which pipeline to run: primary|literature"),
    config: str = typer.Option(..., "--config", "-c", help="Path to configs/project.yaml"),
) -> None:
    """Run primary or literature replication pipeline."""
    config_path = Path(config).resolve()
    cfg = load_config(config_path)

    if which == "primary":
        meta = run_primary(cfg, config_path=config_path)
        console.print(f"[green]OK[/green] primary run complete: {meta['run_id']}")
        console.print(f"Outputs: {meta['outputs']}")
        return

    if which == "paper":
        outputs = run_paper_pipeline(cfg, config_path=config_path)
        console.print(f"[green]OK[/green] paper pipeline complete: {outputs.paper_run_id}")
        console.print(f"Combined table: {outputs.combined_table_path}")
        console.print(f"Report: {outputs.report_path}")
        for fig in outputs.figures:
            console.print(f"Figure: {fig}")
        return

    if which == "literature":
        raise NotImplementedError("Literature pipeline not implemented yet.")

    raise typer.BadParameter("which must be one of: primary|paper|literature")


def _project_root_from_queries_path(path: Path) -> Path:
    # Default: <repo>/queries/*.yaml -> project root is parent of queries/
    if path.parent.name == "queries":
        return path.parent.parent.resolve()
    return path.parent.resolve()


@pmcoa_app.command("search")
def pmcoa_search(
    query_yaml: str = typer.Option(
        "queries/pmcoa_search.yaml", "--query", "-q", help="Path to PMC OA query YAML."
    ),
    out_dir: str | None = typer.Option(
        None, "--out-dir", help="Output directory (default: <project>/data/interim/pmcoa/)."
    ),
    max_records: int | None = typer.Option(
        None, "--max-records", help="Optional cap for debugging (limits records fetched)."
    ),
) -> None:
    """Run PMC OA search and write an auditable registry CSV + split recommendation."""
    qpath = Path(query_yaml).resolve()
    query_file = load_query_file(qpath)
    project_root = _project_root_from_queries_path(qpath)

    if out_dir:
        out_base = Path(out_dir).resolve()
    else:
        out_base = project_root / "data" / "interim" / "pmcoa"
    out_base.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    all_rows = []
    for q in query_file.queries:
        registry_path = out_base / f"registry__{q.id}__{ts}.csv"
        year_counts_path = out_base / f"year_counts__{q.id}__{ts}.csv"
        df = build_registry(
            query_file=query_file,
            query=q,
            out_csv=registry_path,
            year_counts_csv=year_counts_path,
            max_records=max_records,
        )
        all_rows.append(df)

        if query_file.split.cutoff_year is not None:
            cutoff = int(query_file.split.cutoff_year)
            years = pd.to_numeric(df["year"], errors="coerce").dropna().astype(int)
            n_disc = int((years <= cutoff).sum())
            n_conf = int((years > cutoff).sum())
            note = "Fixed cutoff year from query YAML."
            if n_disc < int(query_file.split.min_per_split) or n_conf < int(query_file.split.min_per_split):
                note += " Warning: min_per_split not met."
            rec = SplitRecommendation(
                cutoff_year=cutoff,
                n_discovery=n_disc,
                n_confirmation=n_conf,
                min_per_split=int(query_file.split.min_per_split),
                tie_breaker=str(query_file.split.tie_breaker),
                note=note,
            )
        else:
            rec = recommend_time_split(
                df,
                min_per_split=query_file.split.min_per_split,
                tie_breaker=query_file.split.tie_breaker,
            )
        rec_path = out_base / f"split_recommendation__{q.id}__{ts}.json"
        rec_path.write_text(json.dumps(rec.__dict__, indent=2), encoding="utf-8")

        console.print(f"[green]OK[/green] registry: {registry_path}")
        console.print(f"Year counts: {year_counts_path}")
        console.print(
            f"Recommended split: discovery year<= {rec.cutoff_year} (n={rec.n_discovery}) "
            f"vs confirmation > {rec.cutoff_year} (n={rec.n_confirmation})"
        )
        console.print(f"Split recommendation: {rec_path}")

    if len(all_rows) > 1:
        combined = pd.concat(all_rows, ignore_index=True)
        combined_path = out_base / f"registry__ALL__{ts}.csv"
        combined.to_csv(combined_path, index=False)
        console.print(f"[green]OK[/green] combined registry: {combined_path}")


@pmcoa_app.command("make-manifest")
def pmcoa_make_manifest(
    registry_csv: str = typer.Option(..., "--registry", "-r", help="Path to a registry__*.csv."),
    out_csv: str | None = typer.Option(
        None, "--out", "-o", help="Output manifest CSV (default: alongside registry)."
    ),
    pdf_dir: str | None = typer.Option(
        None,
        "--pdf-dir",
        help="Directory where you will save downloaded PDFs (default: <project>/data/external/pmcoa_pdfs/raw).",
    ),
) -> None:
    """Create a manual-download manifest with stable filenames and URLs."""
    reg_path = Path(registry_csv).resolve()
    if out_csv:
        out_path = Path(out_csv).resolve()
    else:
        out_path = reg_path.with_name(reg_path.name.replace("registry__", "download_manifest__"))

    # Attempt to infer project root from a typical layout: <project>/data/interim/pmcoa/registry__*.csv
    project_root = reg_path
    for _ in range(6):
        if (project_root / "docs").exists() and (project_root / "src").exists():
            break
        project_root = project_root.parent

    if pdf_dir:
        pdf_dir_path = Path(pdf_dir).resolve()
    else:
        pdf_dir_path = project_root / "data" / "external" / "pmcoa_pdfs" / "raw"
    pdf_dir_path.mkdir(parents=True, exist_ok=True)

    outputs = build_download_manifest(registry_csv=reg_path, out_csv=out_path, pdf_dir=pdf_dir_path)
    console.print(f"[green]OK[/green] manifest written: {outputs.manifest_csv}")


@pmcoa_app.command("fetch-jats")
def pmcoa_fetch_jats(
    registry_csv: str = typer.Option(..., "--registry", "-r", help="Path to a registry__*.csv."),
    out_dir: str | None = typer.Option(
        None,
        "--out-dir",
        help="Output dir for JATS XML files (default: <project>/data/interim/pmcoa/jats).",
    ),
    max_papers: int = typer.Option(100, "--n", help="Number of papers to fetch (random sample unless --mode sequential)."),
    fetch_all: bool = typer.Option(False, "--all", help="Fetch all eligible PMCIDs from the registry (sequential)."),
    mode: str = typer.Option("random", "--mode", help="Fetch mode: random|sequential."),
    seed: int = typer.Option(42, "--seed", help="Random seed for sampling."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing XML files."),
    batch_size: int = typer.Option(25, "--batch-size", help="Pause after this many fetches (batch pacing)."),
    batch_sleep_s: float = typer.Option(10.0, "--batch-sleep-s", help="Seconds to sleep between batches."),
    batch_jitter_s: float = typer.Option(2.0, "--batch-jitter-s", help="Random jitter added to batch sleep."),
    env_file: str | None = typer.Option(
        None,
        "--env-file",
        help="Optional .env file to load before running (useful for NCBI_API_KEY / NCBI_EMAIL).",
    ),
) -> None:
    """Fetch PMC full text in JATS/XML via NCBI efetch (streaming, rate-limited)."""
    reg_path = Path(registry_csv).resolve()

    # Infer project root from typical layout: <project>/data/interim/pmcoa/registry__*.csv
    project_root = reg_path
    for _ in range(6):
        if (project_root / "docs").exists() and (project_root / "src").exists():
            break
        project_root = project_root.parent

    if out_dir:
        out_dir_path = Path(out_dir).resolve()
    else:
        out_dir_path = project_root / "data" / "interim" / "pmcoa" / "jats"
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # Best-effort dotenv loading (never prints secrets).
    if env_file:
        load_dotenv_candidates([Path(env_file).resolve()], override=False)
    else:
        load_dotenv_candidates(
            [
                (project_root / ".env").resolve(),
                (project_root.parent / ".env").resolve(),
                (Path.cwd() / ".env").resolve(),
                (Path.cwd().parent / ".env").resolve(),
            ],
            override=False,
        )

    # Read search YAML to re-use the same NCBI rate-limit settings if present.
    # If missing, fall back to safe defaults.
    query_yaml = project_root / "queries" / "pmcoa_search.yaml"
    if query_yaml.exists():
        qfile = load_query_file(query_yaml)
        client = make_client(
            tool=qfile.ncbi.tool,
            tool_env=qfile.ncbi.tool_env,
            email_env=qfile.ncbi.email_env,
            api_key_env=qfile.ncbi.api_key_env,
            request_delay_s=qfile.ncbi.request_delay_s,
        )
    else:
        client = make_client(
            tool="paperjn",
            tool_env="NCBI_TOOL",
            email_env="NCBI_EMAIL",
            api_key_env="NCBI_API_KEY",
            request_delay_s=0.4,
        )

    df_log, log_path = fetch_jats_from_registry(
        client=client,
        registry_csv=reg_path,
        out_dir=out_dir_path,
        max_papers=None if bool(fetch_all) else int(max_papers),
        random_seed=int(seed),
        overwrite=bool(overwrite),
        batch_size=int(batch_size),
        batch_sleep_s=float(batch_sleep_s),
        batch_jitter_s=float(batch_jitter_s),
        mode="sequential" if bool(fetch_all) else ("sequential" if mode == "sequential" else "random"),
    )

    n_ok = int((df_log["status"] == "ok").sum())
    n_skip = int((df_log["status"] == "skipped").sum())
    n_err = int((df_log["status"] == "error").sum())
    n_body = int(df_log["has_body"].fillna(False).astype(bool).sum())
    console.print(f"[green]OK[/green] wrote fetch log: {log_path}")
    console.print(f"Fetched: ok={n_ok}, skipped={n_skip}, error={n_err}")
    console.print(f"Has <body>: {n_body}/{len(df_log)}")


@pmcoa_app.command("parse-jats")
def pmcoa_parse_jats(
    xml_dir: str = typer.Option(
        "data/interim/pmcoa/jats",
        "--xml-dir",
        help="Directory containing fetched PMC JATS/XML files (default: data/interim/pmcoa/jats).",
    ),
    out_dir: str | None = typer.Option(
        None,
        "--out-dir",
        help="Output dir for parsed blocks (default: sibling of xml_dir named 'blocks').",
    ),
    max_papers: int | None = typer.Option(
        None, "--n", help="Optional cap for debugging (samples files from xml_dir)."
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed for sampling."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing parsed outputs."),
) -> None:
    """Parse JATS/XML into paragraph blocks with section provenance (JSONL per paper)."""
    xml_dir_path = Path(xml_dir).resolve()
    if out_dir:
        out_dir_path = Path(out_dir).resolve()
    else:
        out_dir_path = xml_dir_path.parent / "blocks"

    df_log, log_path = parse_jats_dir(
        xml_dir=xml_dir_path,
        out_dir=out_dir_path,
        max_papers=max_papers,
        random_seed=int(seed),
        overwrite=bool(overwrite),
    )

    n_ok = int((df_log["status"] == "ok").sum())
    n_skip = int((df_log["status"] == "skipped").sum())
    n_err = int((df_log["status"] == "error").sum())
    n_body = int(df_log["has_body"].fillna(False).astype(bool).sum())
    console.print(f"[green]OK[/green] wrote parse log: {log_path}")
    console.print(f"Parsed: ok={n_ok}, skipped={n_skip}, error={n_err}")
    console.print(f"Has <body>: {n_body}/{len(df_log)}")


@pmcoa_app.command("extract-segments")
def pmcoa_extract_segments(
    xml_dir: str = typer.Option(
        "data/interim/pmcoa/jats",
        "--xml-dir",
        help="Directory containing fetched PMC JATS/XML files (default: data/interim/pmcoa/jats).",
    ),
    registry_csv: str | None = typer.Option(
        None, "--registry", "-r", help="Optional registry__*.csv (adds PMID/DOI/year metadata)."
    ),
    config: str | None = typer.Option(
        "configs/project.yaml",
        "--config",
        "-c",
        help="Project config (used for leakage blacklist). Set to '' to disable.",
    ),
    out_dir: str | None = typer.Option(
        None,
        "--out-dir",
        help="Output directory (default: <xml_dir>/../extractions/extract__<timestamp>).",
    ),
    extract_all: bool = typer.Option(False, "--all", help="Process all XML files in xml_dir (ignores --n)."),
    max_papers: int | None = typer.Option(
        30,
        "--n",
        help="Optional cap for debugging (samples XML files from xml_dir).",
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed for sampling."),
    dry_run: bool = typer.Option(
        True, "--dry-run/--no-dry-run", help="Dry-run (no API calls; uses heuristics)."
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing outputs in out_dir."),
    resume: bool = typer.Option(
        False, "--resume", help="Resume an existing run in out_dir (append logs; skip completed)."
    ),
    retry_errors: bool = typer.Option(
        False, "--retry-errors", help="When resuming, retry PMCIDs that previously had status=error."
    ),
    watch: bool = typer.Option(
        False,
        "--watch",
        help="Watch xml_dir and process new XML as they appear (useful while fetch-jats is running).",
    ),
    watch_sleep_s: float = typer.Option(10.0, "--watch-sleep-s", help="Seconds between directory scans in watch mode."),
    stop_after_idle_min: float | None = typer.Option(
        None, "--stop-after-idle-min", help="Stop after N minutes with no new processed papers (watch mode)."
    ),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle candidate order (debugging)."),
    save_debug_bundles: bool = typer.Option(
        False,
        "--save-debug-bundles",
        help="Save per-paper LLM inputs/responses (can be large; contains full text snippets).",
    ),
    pass_a_model: str = typer.Option("gpt-5-nano", "--pass-a-model", help="Model for Pass A routing."),
    pass_b_model: str = typer.Option("gpt-5.2", "--pass-b-model", help="Model for Pass B extraction."),
    openai_endpoint: str = typer.Option(
        "responses",
        "--openai-endpoint",
        help="OpenAI endpoint: responses|chat_completions.",
    ),
    openai_request_delay_s: float = typer.Option(
        0.2, "--openai-request-delay-s", help="Seconds to sleep between OpenAI requests."
    ),
    max_blocks: int = typer.Option(140, "--max-blocks", help="Max paragraph blocks sent to Pass B."),
    max_total_chars: int = typer.Option(
        60_000, "--max-total-chars", help="Max total characters sent to Pass B."
    ),
    prompt_a: str | None = typer.Option(None, "--prompt-a", help="Override Pass A prompt file path."),
    prompt_b: str | None = typer.Option(None, "--prompt-b", help="Override Pass B prompt file path."),
    env_file: str | None = typer.Option(
        None,
        "--env-file",
        help="Optional .env file to load before running (useful for OPENAI_API_KEY).",
    ),
) -> None:
    """Extract standardized clinical narrative segments from JATS/XML using a 2-pass LLM pipeline."""
    xml_dir_path = Path(xml_dir).resolve()
    registry_path = Path(registry_csv).resolve() if registry_csv else None

    if bool(resume) and not out_dir:
        raise typer.BadParameter("--resume requires --out-dir (stable run directory).")
    if bool(retry_errors) and not bool(resume):
        raise typer.BadParameter("--retry-errors requires --resume.")
    if bool(extract_all):
        max_papers = None

    # Best-effort dotenv loading (never prints secrets).
    if env_file:
        load_dotenv_candidates([Path(env_file).resolve()], override=False)
    else:
        # Try common locations: <project>/.env and parent .env (repo root), plus CWD.
        project_root = xml_dir_path
        for _ in range(8):
            if (project_root / "src").exists() and (project_root / "docs").exists():
                break
            project_root = project_root.parent
        load_dotenv_candidates(
            [
                (project_root / ".env").resolve(),
                (project_root.parent / ".env").resolve(),
                (Path.cwd() / ".env").resolve(),
                (Path.cwd().parent / ".env").resolve(),
            ],
            override=False,
        )

    cfg = None
    if config and str(config).strip():
        cfg_path = Path(config).resolve()
        if cfg_path.exists():
            cfg = load_config(cfg_path)

    if openai_endpoint not in {"responses", "chat_completions"}:
        raise typer.BadParameter("openai-endpoint must be: responses|chat_completions")

    outputs = extract_from_xml_dir(
        xml_dir=xml_dir_path,
        out_dir=Path(out_dir).resolve() if out_dir else None,
        registry_csv=registry_path,
        config=cfg,
        max_papers=max_papers,
        random_seed=int(seed),
        dry_run=bool(dry_run),
        overwrite=bool(overwrite),
        resume=bool(resume),
        retry_errors=bool(retry_errors),
        watch=bool(watch),
        watch_sleep_s=float(watch_sleep_s),
        stop_after_idle_s=(float(stop_after_idle_min) * 60.0) if stop_after_idle_min is not None else None,
        sample_random=bool(shuffle),
        save_debug_bundles=bool(save_debug_bundles),
        pass_a_model=str(pass_a_model),
        pass_b_model=str(pass_b_model),
        openai_endpoint=str(openai_endpoint),
        openai_request_delay_s=float(openai_request_delay_s),
        max_blocks=int(max_blocks),
        max_total_chars=int(max_total_chars),
        prompt_a_path=Path(prompt_a).resolve() if prompt_a else None,
        prompt_b_path=Path(prompt_b).resolve() if prompt_b else None,
    )

    console.print(f"[green]OK[/green] extraction run dir: {outputs.run_dir}")
    console.print(f"Paper log: {outputs.paper_log_csv}")
    console.print(f"Segments: {outputs.segments_csv}")
    console.print(f"Run config: {outputs.run_config_json}")


@pmcoa_app.command("label-cases")
def pmcoa_label_cases(
    segments_csv: str = typer.Option(
        ...,
        "--segments-csv",
        help="Path to a Phase B segments.csv (e.g., data/interim/pmcoa/extractions/<run>/segments.csv).",
    ),
    xml_dir: str | None = typer.Option(
        None,
        "--xml-dir",
        help="Optional directory with PMC JATS/XML files (adds diagnostic snippets + block text provenance).",
    ),
    out_dir: str | None = typer.Option(
        None,
        "--out-dir",
        help="Output directory (default: <segments_csv>/../phase_c/label__<timestamp>).",
    ),
    include_pmcid: list[str] = typer.Option(
        [],
        "--include-pmcid",
        help="Force-include these PMCIDs in the run (can pass multiple times).",
    ),
    extract_all: bool = typer.Option(False, "--all", help="Label all case units (ignores --n)."),
    max_cases: int | None = typer.Option(
        40, "--n", help="Optional cap for a pilot labeling run (samples case units deterministically)."
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed for sampling."),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run", help="Dry-run (no API calls)."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing outputs in out_dir."),
    resume: bool = typer.Option(False, "--resume", help="Resume an existing run in out_dir."),
    retry_errors: bool = typer.Option(False, "--retry-errors", help="When resuming, retry prior status=error."),
    pass_c_model: str = typer.Option("gpt-5.2", "--pass-c-model", help="Model for Phase C labeling."),
    pass_c_escalate_model: str = typer.Option(
        "gpt-5.2", "--pass-c-escalate-model", help="Escalation model for low-confidence cases."
    ),
    num_shards: int = typer.Option(1, "--num-shards", help="Process only 1/N shards of case units (default: 1)."),
    shard_index: int = typer.Option(0, "--shard-index", help="Shard index in [0, N-1] when --num-shards > 1."),
    openai_endpoint: str = typer.Option("responses", "--openai-endpoint", help="responses|chat_completions"),
    openai_timeout_s: int = typer.Option(180, "--openai-timeout-s", help="HTTP timeout per OpenAI request."),
    openai_request_delay_s: float = typer.Option(0.2, "--openai-request-delay-s", help="Delay between OpenAI calls."),
    max_blocks: int = typer.Option(120, "--max-blocks", help="Max blocks sent to Phase C model."),
    max_total_chars: int = typer.Option(45_000, "--max-total-chars", help="Max total characters sent to Phase C model."),
    max_extra_snippets: int = typer.Option(8, "--max-extra-snippets", help="Extra diagnostic blocks to include."),
    prompt_c: str | None = typer.Option(None, "--prompt-c", help="Override Phase C prompt file path."),
    save_debug_bundles: bool = typer.Option(False, "--save-debug-bundles", help="Save inputs/responses per case."),
    env_file: str | None = typer.Option(None, "--env-file", help="Optional .env file to load before running."),
) -> None:
    """Phase C v1: label case units (FTLD inclusion + syndrome + symptom tags) from extracted segments."""
    segments_path = Path(segments_csv).resolve()
    xml_dir_path = Path(xml_dir).resolve() if xml_dir else None

    if retry_errors and not resume:
        raise typer.BadParameter("--retry-errors requires --resume.")
    if resume and not out_dir:
        raise typer.BadParameter("--resume requires --out-dir (stable run directory).")
    if extract_all:
        max_cases = None

    if int(num_shards) < 1:
        raise typer.BadParameter("--num-shards must be >= 1.")
    if int(shard_index) < 0 or int(shard_index) >= int(num_shards):
        raise typer.BadParameter("--shard-index must be in [0, --num-shards-1].")
    if int(openai_timeout_s) < 30:
        raise typer.BadParameter("--openai-timeout-s must be >= 30.")

    # Best-effort dotenv loading (never prints secrets).
    project_root = segments_path
    for _ in range(10):
        if (project_root / "src").exists() and (project_root / "docs").exists():
            break
        project_root = project_root.parent
    if env_file:
        load_dotenv_candidates([Path(env_file).resolve()], override=False)
    else:
        load_dotenv_candidates(
            [
                (project_root / ".env").resolve(),
                (project_root.parent / ".env").resolve(),
                (Path.cwd() / ".env").resolve(),
                (Path.cwd().parent / ".env").resolve(),
            ],
            override=False,
        )

    if openai_endpoint not in {"responses", "chat_completions"}:
        raise typer.BadParameter("openai-endpoint must be: responses|chat_completions")

    outputs = label_phase_c_v1(
        segments_csv=segments_path,
        xml_dir=xml_dir_path,
        out_dir=Path(out_dir).resolve() if out_dir else None,
        max_cases=max_cases,
        include_pmcids=list(include_pmcid) if include_pmcid else None,
        random_seed=int(seed),
        dry_run=bool(dry_run),
        overwrite=bool(overwrite),
        resume=bool(resume),
        retry_errors=bool(retry_errors),
        pass_c_model=str(pass_c_model),
        pass_c_escalate_model=str(pass_c_escalate_model),
        num_shards=int(num_shards),
        shard_index=int(shard_index),
        openai_endpoint=str(openai_endpoint),
        openai_timeout_s=int(openai_timeout_s),
        openai_request_delay_s=float(openai_request_delay_s),
        max_blocks=int(max_blocks),
        max_total_chars=int(max_total_chars),
        max_extra_snippets=int(max_extra_snippets),
        prompt_c_path=Path(prompt_c).resolve() if prompt_c else None,
        save_debug_bundles=bool(save_debug_bundles),
    )

    console.print(f"[green]OK[/green] Phase C run dir: {outputs.run_dir}")
    console.print(f"Labels: {outputs.labels_csv}")
    console.print(f"Run config: {outputs.run_config_json}")
    console.print(f"Report: {outputs.report_md}")


@pmcoa_app.command("analyze-splits")
def pmcoa_analyze_splits(
    segments_csv: str = typer.Option(
        ...,
        "--segments-csv",
        help="Path to a Phase B segments.csv (e.g., data/interim/pmcoa/extractions/<run>/segments.csv).",
    ),
    case_labels_csv: str = typer.Option(
        ...,
        "--case-labels-csv",
        help="Path to Phase C case_labels_long.csv (ideally the combined/latest per case).",
    ),
    config: str = typer.Option(
        "configs/project.yaml",
        "--config",
        "-c",
        help="Project config (embedding model, PCA rule, k, permutations).",
    ),
    out_dir: str | None = typer.Option(
        None,
        "--out-dir",
        help="Output directory (default: <segments_csv>/../phase_d/analysis__<timestamp>).",
    ),
    include_broad: bool = typer.Option(
        False, "--include-broad", help="Include ftld_broad cases in addition to ftld_strict (sensitivity)."
    ),
    pca_fit_on: str = typer.Option(
        "discovery",
        "--pca-fit-on",
        help="PCA fit set: discovery|all (default: discovery for reviewer-proof split isolation).",
    ),
    min_segments_per_case: int = typer.Option(
        1,
        "--min-segments-per-case",
        help="Minimum number of embedding-eligible segments required per case (default: 1).",
    ),
    one_case_per_pmcid: bool = typer.Option(
        False,
        "--one-case-per-pmcid",
        help="Sensitivity: keep only one case unit per PMCID (lexicographically smallest case_id).",
    ),
    write_latest_outputs: bool = typer.Option(
        True, "--write-latest/--no-write-latest", help="Write a convenience copy under phase_d/latest/."
    ),
) -> None:
    """Phase D: split-sample clustering (Discovery) + replication (Confirmation) on PMC OA cases."""
    segments_path = Path(segments_csv).resolve()
    labels_path = Path(case_labels_csv).resolve()
    cfg_path = Path(config).resolve()
    cfg = load_config(cfg_path)

    outputs = run_phase_d_pmcoa_split_analysis(
        config=cfg,
        segments_csv=segments_path,
        case_labels_csv=labels_path,
        out_dir=Path(out_dir).resolve() if out_dir else None,
        include_broad=bool(include_broad),
        pca_fit_on=str(pca_fit_on),
        min_segments_per_case=int(min_segments_per_case),
        one_case_per_pmcid=bool(one_case_per_pmcid),
        write_latest_outputs=bool(write_latest_outputs),
    )

    console.print(f"[green]OK[/green] Phase D run dir: {outputs.run_dir}")
    console.print(f"Report: {outputs.performance_report_md}")
    console.print(f"Replication summary: {outputs.replication_summary_csv}")
    console.print(f"Syndrome replication (secondary): {outputs.syndrome_replication_summary_csv}")
    console.print(f"Cluster characterization: {outputs.cluster_characterization_csv}")
    console.print(f"Numeric characterization: {outputs.numeric_characterization_csv}")
    console.print(f"Syndrome composition: {outputs.syndrome_composition_csv}")
    console.print(f"Flow table: {outputs.flow_table_csv}")
    console.print(f"Case table: {outputs.case_table_csv}")
    console.print(f"Symptom tags (matched clusters): {outputs.symptom_tags_matched_csv}")


@pmcoa_app.command("analyze-splits-suite")
def pmcoa_analyze_splits_suite(
    segments_csv: str = typer.Option(
        ...,
        "--segments-csv",
        help="Path to a Phase B segments.csv (e.g., data/interim/pmcoa/extractions/<run>/segments.csv).",
    ),
    case_labels_csv: str = typer.Option(
        ...,
        "--case-labels-csv",
        help="Path to Phase C case_labels_long.csv (combined/latest per case).",
    ),
    config: str = typer.Option(
        "configs/project.yaml",
        "--config",
        "-c",
        help="Project config (embedding model, PCA rule, permutations).",
    ),
    out_dir: str | None = typer.Option(
        None,
        "--out-dir",
        help="Output directory for the suite (default: results/phase_d/robustness__<timestamp>).",
    ),
    min_segments_per_case: int = typer.Option(
        1,
        "--min-segments-per-case",
        help="Minimum number of embedding-eligible segments required per case (default: 1).",
    ),
    one_case_per_pmcid: bool = typer.Option(
        False,
        "--one-case-per-pmcid",
        help="Apply one-case-per-PMCID dedupe to ALL suite runs (off by default).",
    ),
) -> None:
    """Run a pre-specified Phase D robustness suite (k + PCA fit + inclusion tier + embedding family)."""
    segments_path = Path(segments_csv).resolve()
    labels_path = Path(case_labels_csv).resolve()
    cfg_path = Path(config).resolve()
    base_cfg = load_config(cfg_path)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if out_dir:
        suite_dir = Path(out_dir).resolve()
    else:
        suite_dir = (cfg_path.parent.parent / "results" / "phase_d" / f"robustness__{ts}").resolve()
    suite_dir.mkdir(parents=True, exist_ok=True)

    scenarios: list[dict[str, object]] = [
        {
            "name": "primary_strict_pca_discovery_k4_bge",
            "include_broad": False,
            "pca_fit_on": "discovery",
            "k_fixed": 4,
            "embedding_model": "BAAI/bge-small-en-v1.5",
        },
        {
            "name": "sens_include_broad_pca_all_k4_bge",
            "include_broad": True,
            "pca_fit_on": "all",
            "k_fixed": 4,
            "embedding_model": "BAAI/bge-small-en-v1.5",
        },
        {
            "name": "sens_k3_strict_pca_discovery_bge",
            "include_broad": False,
            "pca_fit_on": "discovery",
            "k_fixed": 3,
            "embedding_model": "BAAI/bge-small-en-v1.5",
        },
        {
            "name": "sens_k5_strict_pca_discovery_bge",
            "include_broad": False,
            "pca_fit_on": "discovery",
            "k_fixed": 5,
            "embedding_model": "BAAI/bge-small-en-v1.5",
        },
        {
            "name": "sens_k6_strict_pca_discovery_bge",
            "include_broad": False,
            "pca_fit_on": "discovery",
            "k_fixed": 6,
            "embedding_model": "BAAI/bge-small-en-v1.5",
        },
        {
            "name": "sens_embed_e5_strict_pca_discovery_k4",
            "include_broad": False,
            "pca_fit_on": "discovery",
            "k_fixed": 4,
            "embedding_model": "intfloat/e5-small-v2",
        },
    ]

    summary_rows: list[dict[str, object]] = []

    for sc in scenarios:
        name = str(sc["name"])
        out_run = suite_dir / name
        cfg = base_cfg.model_copy(deep=True)
        cfg.clustering.k_fixed = int(sc["k_fixed"])  # type: ignore[arg-type]
        cfg.embeddings.model_name = str(sc["embedding_model"])

        try:
            outputs = run_phase_d_pmcoa_split_analysis(
                config=cfg,
                segments_csv=segments_path,
                case_labels_csv=labels_path,
                out_dir=out_run,
                include_broad=bool(sc["include_broad"]),
                pca_fit_on=str(sc["pca_fit_on"]),
                min_segments_per_case=int(min_segments_per_case),
                one_case_per_pmcid=bool(one_case_per_pmcid),
                write_latest_outputs=False,
            )
            rep = pd.read_csv(outputs.replication_summary_csv).iloc[0].to_dict()
            row: dict[str, object] = {
                "scenario": name,
                "status": "ok",
                "run_dir": str(outputs.run_dir),
                "replication_summary_csv": str(outputs.replication_summary_csv),
                "performance_report_md": str(outputs.performance_report_md),
            }
            row.update(rep)
            summary_rows.append(row)
            console.print(f"[green]OK[/green] {name}: {outputs.performance_report_md}")
        except Exception as exc:
            summary_rows.append(
                {
                    "scenario": name,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "run_dir": str(out_run),
                }
            )
            console.print(f"[red]ERR[/red] {name}: {type(exc).__name__}: {exc}")

    out_csv = suite_dir / "robustness_suite_summary.csv"
    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
    console.print(f"[green]OK[/green] suite summary: {out_csv}")


@pmcoa_app.command("stability-splits")
def pmcoa_stability_splits(
    phase_d_run_dir: str = typer.Option(
        ...,
        "--phase-d-run-dir",
        help="Path to an existing Phase D run directory (must contain case_embeddings.npz + tables/replication_summary.csv).",
    ),
    config: str = typer.Option(
        "configs/project.yaml",
        "--config",
        "-c",
        help="Project config (stability defaults live under `stability:`).",
    ),
    n_bootstrap: int | None = typer.Option(
        None,
        "--n-bootstrap",
        help="Override number of subsample replicates (default: config.stability.n_bootstrap).",
    ),
    subsample_fraction: float | None = typer.Option(
        None,
        "--subsample-fraction",
        help="Override subsample fraction (default: config.stability.subsample_fraction).",
    ),
    seed_values: str | None = typer.Option(
        None,
        "--seed-values",
        help="Comma-separated seeds for seed-sensitivity (default: config.stability.seed_values).",
    ),
) -> None:
    """Stability for Phase D split clustering (subsample bootstrap + seed sensitivity)."""
    cfg = load_config(Path(config).resolve())
    seeds = None
    if seed_values is not None:
        parts = [p.strip() for p in str(seed_values).split(",") if p.strip()]
        seeds = [int(x) for x in parts]

    outputs = run_phase_d_stability(
        config=cfg,
        phase_d_run_dir=Path(phase_d_run_dir).resolve(),
        n_bootstrap=int(n_bootstrap) if n_bootstrap is not None else None,
        subsample_fraction=float(subsample_fraction) if subsample_fraction is not None else None,
        seed_values=seeds,
    )

    console.print(f"[green]OK[/green] stability dir: {outputs.out_dir}")
    console.print(f"Bootstrap CSV: {outputs.bootstrap_csv}")
    console.print(f"Seed CSV: {outputs.seed_csv}")
    console.print(f"ARI figure: {outputs.fig_ari_path}")
    console.print(f"Jaccard figure: {outputs.fig_jaccard_path}")
    console.print(f"Report: {outputs.report_md}")


@pmcoa_app.command("make-rater-sample")
def pmcoa_make_rater_sample(
    phase_d_run_dir: str = typer.Option(
        ...,
        "--phase-d-run-dir",
        help="Existing Phase D run directory (used for split/cluster/syndrome stratification and for locating segments.csv).",
    ),
    out_dir: str | None = typer.Option(
        None,
        "--out-dir",
        help="Output directory (default: results/phase_d/validation/rater_sample__<timestamp>).",
    ),
    sample_size: int = typer.Option(200, "--n", help="Target sample size (clipped to available cases)."),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducible sampling."),
    max_chars_per_case: int = typer.Option(
        12000, "--max-chars-per-case", help="Truncation cap per extracted segment in packets."
    ),
    include_title_in_packets: bool = typer.Option(
        False, "--include-title", help="Include article title in packet header (off by default to reduce bias)."
    ),
    text_field: str = typer.Option(
        "text_raw", "--text-field", help="Which segment text to show: text_raw|text_clean."
    ),
) -> None:
    """Create a stratified clinician-validation sample (packets + blank rating templates)."""
    outputs = make_rater_sample(
        phase_d_run_dir=Path(phase_d_run_dir).resolve(),
        out_dir=Path(out_dir).resolve() if out_dir else None,
        sample_size=int(sample_size),
        random_seed=int(seed),
        max_chars_per_case=int(max_chars_per_case),
        include_title_in_packets=bool(include_title_in_packets),
        text_field=str(text_field),
    )
    console.print(f"[green]OK[/green] rater sample dir: {outputs.out_dir}")
    console.print(f"Internal manifest: {outputs.sample_manifest_internal_csv}")
    console.print(f"Packets (share): {outputs.packets_dir}")
    console.print(f"Rater1 template: {outputs.rater1_template_csv}")
    console.print(f"Rater2 template: {outputs.rater2_template_csv}")
    console.print(f"Clinician instructions: {outputs.rater_instructions_md}")
    console.print(f"Coverage report (internal): {outputs.coverage_report_md}")
    console.print(f"Meta (internal): {outputs.meta_json}")


@pmcoa_app.command("score-raters")
def pmcoa_score_raters(
    rater1_csv: str = typer.Option(..., "--rater1-csv", help="Filled rater 1 CSV (from rater1_template.csv)."),
    rater2_csv: str = typer.Option(..., "--rater2-csv", help="Filled rater 2 CSV (from rater2_template.csv)."),
    out_dir: str | None = typer.Option(
        None, "--out-dir", help="Output directory for κ report + adjudication sheet."
    ),
) -> None:
    """Compute Cohen's κ (binary) and emit an adjudication template for disagreements."""
    outputs = score_two_raters(
        rater1_csv=Path(rater1_csv).resolve(),
        rater2_csv=Path(rater2_csv).resolve(),
        out_dir=Path(out_dir).resolve() if out_dir else None,
    )
    console.print(f"[green]OK[/green] κ outputs: {outputs.out_dir}")
    console.print(f"κ CSV: {outputs.kappa_csv}")
    console.print(f"κ report: {outputs.kappa_md}")
    console.print(f"Adjudication template: {outputs.adjudication_csv}")


@pmcoa_app.command("make-rater-workbooks")
def pmcoa_make_rater_workbooks(
    rater_sample_dir: str = typer.Option(
        ...,
        "--rater-sample-dir",
        help="Path to an existing rater sample directory (must contain share/rater*_template.csv and share/packets/).",
    ),
    out_dir: str | None = typer.Option(
        None,
        "--out-dir",
        help="Output directory for the workbooks (default: <rater_sample_dir>/share).",
    ),
) -> None:
    """Create Excel workbooks with embedded packet text (Google Sheets friendly)."""
    outputs = make_rater_workbooks(
        rater_sample_dir=Path(rater_sample_dir).resolve(),
        out_dir=Path(out_dir).resolve() if out_dir else None,
    )
    console.print(f"[green]OK[/green] workbook dir: {outputs.out_dir}")
    console.print(f"HUMAN_RATER1_WORKBOOK: {outputs.rater1_xlsx}")
    console.print(f"HUMAN_RATER2_WORKBOOK: {outputs.rater2_xlsx}")


@pmcoa_app.command("subset-rater-sample")
def pmcoa_subset_rater_sample(
    parent_rater_sample_dir: str = typer.Option(
        ...,
        "--parent-rater-sample-dir",
        help="Existing rater sample directory (must contain internal/sample_manifest_internal.csv and share/packets/).",
    ),
    out_dir: str | None = typer.Option(
        None,
        "--out-dir",
        help="Output directory for the subset (default: sibling under the same validation/ folder).",
    ),
    sample_size: int = typer.Option(120, "--n", help="Subset size (clipped to available)."),
    seed: int = typer.Option(42, "--seed", help="Random seed (locks the subset)."),
) -> None:
    """Create a locked stratified subset for human validation (split × matched_cluster)."""
    outputs = make_rater_subset_n(
        parent_rater_sample_dir=Path(parent_rater_sample_dir).resolve(),
        out_dir=Path(out_dir).resolve() if out_dir else None,
        sample_size=int(sample_size),
        random_seed=int(seed),
    )
    console.print(f"[green]OK[/green] subset dir: {outputs.out_dir}")
    console.print(f"Share (send to raters): {outputs.share_dir}")
    console.print(f"Internal (keep private): {outputs.internal_dir}")
    console.print(f"Workbook rater1: {outputs.rater1_workbook_xlsx}")
    console.print(f"Workbook rater2: {outputs.rater2_workbook_xlsx}")
    console.print(f"Checksums: {outputs.share_sha256_csv}")


@pmcoa_app.command("results-packet")
def pmcoa_results_packet(
    phase_d_run_dir: str = typer.Option(
        ...,
        "--phase-d-run-dir",
        help="Existing Phase D run directory (tables/ + optional stability/).",
    ),
    out_dir: str | None = typer.Option(
        None, "--out-dir", help="Output directory (default: alongside the Phase D run)."
    ),
) -> None:
    """Assemble a manuscript-ready Phase D “results packet” (tables + figures)."""
    outputs = build_phase_d_results_packet(
        phase_d_run_dir=Path(phase_d_run_dir).resolve(),
        out_dir=Path(out_dir).resolve() if out_dir else None,
    )
    console.print(f"[green]OK[/green] packet dir: {outputs.out_dir}")
    console.print(f"Packet: {outputs.packet_md}")


@pmcoa_app.command("insilico-rater-study")
def pmcoa_insilico_rater_study(
    rater_sample_dir: str = typer.Option(
        ...,
        "--rater-sample-dir",
        help="Path to an existing rater sample directory (must contain share/ and internal/).",
    ),
    phase_d_run_dir: str | None = typer.Option(
        None,
        "--phase-d-run-dir",
        help="Optional base Phase D run directory. If omitted, will be inferred from rater_sample_dir/internal/meta.json.",
    ),
    out_dir: str | None = typer.Option(
        None,
        "--out-dir",
        help="Output directory (default: results/phase_f_insilico/insilico_rater_study__<timestamp>).",
    ),
    model: str = typer.Option("gpt-5.2", "--model", help="OpenAI model to use for the virtual raters."),
    temperature: float = typer.Option(0.2, "--temperature", help="Sampling temperature (non-zero to probe stability)."),
    endpoint: str = typer.Option(
        "responses",
        "--openai-endpoint",
        help="OpenAI endpoint: responses|chat_completions",
    ),
    request_delay_s: float = typer.Option(0.4, "--openai-request-delay-s", help="Delay between OpenAI requests (seconds)."),
    max_output_tokens: int = typer.Option(900, "--max-output-tokens", help="Max output tokens per case."),
    max_packet_chars: int = typer.Option(30000, "--max-packet-chars", help="Cap packet chars sent to the model."),
    max_cases: int | None = typer.Option(None, "--max-cases", help="Optional cap on number of cases (debug/testing)."),
    exclude_non_ftld: bool = typer.Option(
        False,
        "--exclude-non-ftld",
        help="Sensitivity: exclude cases where the in-silico consensus says is_ftld_spectrum=0 (only when text_adequate=1).",
    ),
    env_file: str | None = typer.Option(
        None, "--env-file", help="Optional .env file to load before running (never prints secrets)."
    ),
) -> None:
    """In-silico clinician study: two GPT raters + κ + Phase D/E relabel sensitivity (clearly labeled INSILICO)."""
    # Best-effort dotenv loading (never prints secrets).
    if env_file:
        load_dotenv_candidates([Path(env_file).resolve()], override=False)
    else:
        load_dotenv_candidates(
            [Path(".env").resolve(), (Path(__file__).resolve().parent.parent.parent / ".env").resolve()], override=False
        )

    if endpoint not in {"responses", "chat_completions"}:
        raise typer.BadParameter("openai-endpoint must be: responses|chat_completions")

    outputs = run_insilico_rater_study(
        rater_sample_dir=Path(rater_sample_dir).resolve(),
        phase_d_run_dir=Path(phase_d_run_dir).resolve() if phase_d_run_dir else None,
        out_dir=Path(out_dir).resolve() if out_dir else None,
        model=str(model),
        temperature=float(temperature),
        endpoint=str(endpoint),  # type: ignore[arg-type]
        request_delay_s=float(request_delay_s),
        max_output_tokens=int(max_output_tokens),
        max_packet_chars=int(max_packet_chars) if max_packet_chars is not None else None,
        max_cases=int(max_cases) if max_cases is not None else None,
        exclude_non_ftld=bool(exclude_non_ftld),
    )

    console.print(f"[green]OK[/green] in-silico study dir: {outputs.run_dir}")
    console.print(f"Report: {outputs.report_md}")
    console.print(f"Rater A CSV: {outputs.rater_a_csv}")
    console.print(f"Rater B CSV: {outputs.rater_b_csv}")
    console.print(f"Consensus CSV: {outputs.consensus_csv}")
    console.print(f"κ dir: {outputs.kappa_dir}")
    console.print(f"Phase D relabel dir: {outputs.phase_d_relabel_dir}")
    console.print(f"Phase E dir: {outputs.phase_e_dir}")


@pmcoa_app.command("phase-e")
def pmcoa_phase_e(
    phase_d_run_dir: str = typer.Option(
        ...,
        "--phase-d-run-dir",
        help="Phase D run directory to pool after replication (must contain tables/case_table_with_clusters.csv).",
    ),
    out_dir: str | None = typer.Option(
        None,
        "--out-dir",
        help="Output directory (default: results/phase_e/pool_after_replication__<timestamp>).",
    ),
    alpha: float = typer.Option(0.05, "--alpha", help="Alpha for Wilson proportion CIs."),
    direction_eps: float = typer.Option(
        0.01, "--direction-eps", help="Delta threshold for enriched/depleted vs split baseline (absolute)."
    ),
    max_top_syndromes: int = typer.Option(5, "--top-syndromes", help="Top N syndromes per cluster (pooled)."),
    max_top_tags: int = typer.Option(5, "--top-tags", help="Top N symptom tags per cluster (pooled)."),
    force: bool = typer.Option(False, "--force", help="Run even if Phase D replication gate fails (not recommended)."),
) -> None:
    """Phase E: post-replication pooled characterization (estimation only; no pooled p-values)."""
    outputs = run_phase_e_pool_after_replication(
        phase_d_run_dir=Path(phase_d_run_dir).resolve(),
        out_dir=Path(out_dir).resolve() if out_dir else None,
        alpha=float(alpha),
        direction_eps=float(direction_eps),
        max_top_syndromes=int(max_top_syndromes),
        max_top_tags=int(max_top_tags),
        force=bool(force),
    )
    console.print(f"[green]OK[/green] Phase E dir: {outputs.run_dir}")
    console.print(f"Packet: {outputs.results_packet_md}")
    console.print(f"Report: {outputs.report_md}")


def _load_refs_yaml(path: Path) -> dict[str, dict[str, object]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("References YAML must be a mapping")
    citations = payload.get("citations", {})
    if not isinstance(citations, dict):
        raise ValueError("References YAML must contain a 'citations:' mapping")
    out: dict[str, dict[str, object]] = {}
    for key, spec in citations.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Invalid citation spec for {key}: expected mapping")
        out[str(key)] = dict(spec)
    return out


def _resolve_specs_to_pmids(
    key_to_spec: dict[str, dict[str, object]],
    *,
    email: str | None,
    tool: str | None,
    api_key: str | None,
) -> dict[str, str]:
    key_to_pmid: dict[str, str] = {}
    for key, spec in key_to_spec.items():
        pmid = spec.get("pmid")
        doi = spec.get("doi")
        if pmid and doi:
            raise ValueError(f"Citation {key} specifies both pmid and doi; choose one.")
        if pmid:
            key_to_pmid[key] = str(pmid).strip()
            continue
        if doi:
            key_to_pmid[key] = resolve_doi_to_pmid(str(doi), tool=tool, email=email, api_key=api_key)
            continue
        raise ValueError(f"Citation {key} must specify either pmid or doi.")
    return key_to_pmid


@citations_app.command("search")
def citations_search(
    query: str = typer.Argument(..., help="PubMed query string (eg, DOI[DOI] or keywords)."),
    retmax: int = typer.Option(10, "--retmax", help="Max results."),
    tool: str | None = typer.Option(None, "--tool", help="NCBI E-utilities tool name."),
    email: str | None = typer.Option(None, "--email", help="Contact email for NCBI E-utilities (recommended)."),
    api_key: str | None = typer.Option(None, "--api-key", help="NCBI API key (optional)."),
) -> None:
    """Search PubMed and print PMID + AMA-formatted references."""
    try:
        pmids = pubmed_search_pmids(query, retmax=int(retmax), tool=tool, email=email, api_key=api_key)
        if not pmids:
            console.print("No results.")
            return
        articles = pubmed_fetch_articles(pmids, tool=tool, email=email, api_key=api_key)
        for a in articles:
            console.print(f"{a.pmid}\t{format_ama_journal_reference(a)}")
    except PubMedError as exc:
        console.print(f"[red]PubMed error[/red]: {exc}")
        raise typer.Exit(code=2) from exc


@citations_app.command("format")
def citations_format(
    pmids: str = typer.Argument(..., help="Comma-separated PMIDs."),
    tool: str | None = typer.Option(None, "--tool", help="NCBI E-utilities tool name."),
    email: str | None = typer.Option(None, "--email", help="Contact email for NCBI E-utilities (recommended)."),
    api_key: str | None = typer.Option(None, "--api-key", help="NCBI API key (optional)."),
) -> None:
    """Format one or more PMIDs as AMA references."""
    try:
        ids = [p.strip() for p in pmids.split(",") if p.strip()]
        articles = pubmed_fetch_articles(ids, tool=tool, email=email, api_key=api_key)
        by_pmid = {a.pmid: a for a in articles}
        for pmid in ids:
            a = by_pmid.get(pmid)
            if not a:
                raise PubMedError(f"PMID not returned by PubMed: {pmid}")
            console.print(format_ama_journal_reference(a))
    except PubMedError as exc:
        console.print(f"[red]PubMed error[/red]: {exc}")
        raise typer.Exit(code=2) from exc


@citations_app.command("apply")
def citations_apply(
    manuscript: str = typer.Option(..., "--manuscript", help="Input markdown file."),
    references: str = typer.Option(..., "--references", help="YAML mapping citation keys to PMID/DOI."),
    out: str | None = typer.Option(None, "--out", help="Output markdown path (default: same as input)."),
    inplace: bool = typer.Option(False, "--inplace", help="Edit the manuscript in place."),
    lockfile: str | None = typer.Option(
        None, "--lockfile", help="Write a JSON lockfile with fetched PubMed metadata and numbering."
    ),
    tool: str | None = typer.Option(None, "--tool", help="NCBI E-utilities tool name."),
    email: str | None = typer.Option(None, "--email", help="Contact email for NCBI E-utilities (recommended)."),
    api_key: str | None = typer.Option(None, "--api-key", help="NCBI API key (optional)."),
) -> None:
    """Apply {{cite:Key}} tags and write/refresh a ## References section (AMA style)."""
    if inplace and out:
        raise typer.BadParameter("Use either --inplace or --out, not both.")

    manuscript_path = Path(manuscript).resolve()
    refs_path = Path(references).resolve()
    out_path = Path(out).resolve() if out else manuscript_path

    key_to_spec = _load_refs_yaml(refs_path)
    md = manuscript_path.read_text(encoding="utf-8")
    cite_keys = extract_cite_keys(md)
    if not cite_keys:
        raise typer.BadParameter(
            f"No citation tags found in {manuscript_path}. Expected {{{{cite:Key}}}} tags."
        )

    missing = sorted(set(cite_keys) - set(key_to_spec.keys()))
    if missing:
        raise typer.BadParameter(f"Missing citation specs for keys: {missing}")

    key_to_pmid = _resolve_specs_to_pmids(key_to_spec, email=email, tool=tool, api_key=api_key)

    # Number in first-appearance order, de-duplicated.
    ordered_unique_keys: list[str] = []
    seen: set[str] = set()
    for k in cite_keys:
        if k in seen:
            continue
        seen.add(k)
        ordered_unique_keys.append(k)
    key_to_number = {k: i + 1 for i, k in enumerate(ordered_unique_keys)}

    try:
        articles = pubmed_fetch_articles(
            [key_to_pmid[k] for k in ordered_unique_keys],
            tool=tool,
            email=email,
            api_key=api_key,
        )
    except PubMedError as exc:
        console.print(f"[red]PubMed error[/red]: {exc}")
        raise typer.Exit(code=2) from exc
    by_pmid = {a.pmid: a for a in articles}

    refs_lines: list[str] = []
    lock_entries: list[dict[str, object]] = []
    for k in ordered_unique_keys:
        pmid = key_to_pmid[k]
        a = by_pmid.get(pmid)
        if not a:
            raise typer.BadParameter(f"PMID not returned by PubMed: {pmid} (key={k})")
        refs_lines.append(f"{key_to_number[k]}. {format_ama_journal_reference(a)}")
        lock_entries.append({"key": k, "number": key_to_number[k], "pmid": pmid, "article": a.to_json()})

    updated = replace_cite_tags(md, key_to_number)

    marker = "\n## References"
    if marker in updated:
        head, _ = updated.split(marker, 1)
        updated = head.rstrip() + "\n\n## References\n\n" + "\n".join(refs_lines) + "\n"
    else:
        updated = updated.rstrip() + "\n\n## References\n\n" + "\n".join(refs_lines) + "\n"

    if inplace:
        manuscript_path.write_text(updated, encoding="utf-8")
        out_path = manuscript_path
    else:
        out_path.write_text(updated, encoding="utf-8")

    if lockfile:
        lock_path = Path(lockfile).resolve()
        lock = {
            "manuscript": str(out_path),
            "references_yaml": str(refs_path),
            "entries": lock_entries,
        }
        lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
        console.print(f"Wrote: {lock_path}")

    console.print(f"[green]OK[/green] wrote: {out_path}")
