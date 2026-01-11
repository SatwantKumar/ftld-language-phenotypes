from __future__ import annotations

import csv
from pathlib import Path

from paperjn.pmcoa.extract_pipeline import extract_from_xml_dir


def _count_csv_rows(path: Path) -> int:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        assert header is not None
        return sum(1 for _ in reader)


def test_extract_pipeline_resume_dedupes_segments_if_paper_log_missing(tmp_path: Path) -> None:
    xml_dir = tmp_path / "jats"
    xml_dir.mkdir(parents=True, exist_ok=True)

    xml = """<?xml version="1.0" encoding="UTF-8"?>
<article article-type="case-report">
  <front>
    <article-meta>
      <title-group>
        <article-title>Test Title</article-title>
      </title-group>
    </article-meta>
  </front>
  <body>
    <sec>
      <title>Case Presentation</title>
      <p>A 64-year-old man presented with progressive speech difficulty.</p>
      <p>He was diagnosed with primary progressive aphasia.</p>
    </sec>
  </body>
</article>
"""
    (xml_dir / "PMC123.xml").write_text(xml, encoding="utf-8")

    project_root = Path(__file__).resolve().parents[1]
    prompt_a = project_root / "prompts" / "pmcoa" / "pass_a_route_v2.md"
    prompt_b = project_root / "prompts" / "pmcoa" / "pass_b_extract_v2.md"
    assert prompt_a.exists()
    assert prompt_b.exists()

    out_dir = tmp_path / "out"
    outputs1 = extract_from_xml_dir(
        xml_dir=xml_dir,
        out_dir=out_dir,
        registry_csv=None,
        config=None,
        max_papers=1,
        random_seed=42,
        dry_run=True,
        overwrite=False,
        resume=False,
        retry_errors=False,
        watch=False,
        watch_sleep_s=0.0,
        stop_after_idle_s=None,
        sample_random=False,
        save_debug_bundles=False,
        pass_a_model="gpt-5-nano",
        pass_b_model="gpt-5.2",
        openai_endpoint="responses",
        openai_request_delay_s=0.0,
        max_blocks=140,
        max_total_chars=60_000,
        prompt_a_path=prompt_a,
        prompt_b_path=prompt_b,
    )

    n_segments_before = _count_csv_rows(outputs1.segments_csv)
    assert n_segments_before == 1

    # Simulate an interrupted run where segments were written but paper_log.csv was not.
    outputs1.paper_log_csv.unlink()
    assert outputs1.paper_log_csv.exists() is False

    outputs2 = extract_from_xml_dir(
        xml_dir=xml_dir,
        out_dir=out_dir,
        registry_csv=None,
        config=None,
        max_papers=1,
        random_seed=42,
        dry_run=True,
        overwrite=False,
        resume=True,
        retry_errors=False,
        watch=False,
        watch_sleep_s=0.0,
        stop_after_idle_s=None,
        sample_random=False,
        save_debug_bundles=False,
        pass_a_model="gpt-5-nano",
        pass_b_model="gpt-5.2",
        openai_endpoint="responses",
        openai_request_delay_s=0.0,
        max_blocks=140,
        max_total_chars=60_000,
        prompt_a_path=prompt_a,
        prompt_b_path=prompt_b,
    )

    assert outputs2.paper_log_csv.exists()
    n_segments_after = _count_csv_rows(outputs2.segments_csv)
    assert n_segments_after == n_segments_before


def test_extract_pipeline_watch_mode_allows_empty_xml_dir(tmp_path: Path) -> None:
    xml_dir = tmp_path / "empty_jats"
    xml_dir.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parents[1]
    prompt_a = project_root / "prompts" / "pmcoa" / "pass_a_route_v2.md"
    prompt_b = project_root / "prompts" / "pmcoa" / "pass_b_extract_v2.md"

    out_dir = tmp_path / "out_watch"
    outputs = extract_from_xml_dir(
        xml_dir=xml_dir,
        out_dir=out_dir,
        registry_csv=None,
        config=None,
        max_papers=1,
        random_seed=42,
        dry_run=True,
        overwrite=False,
        resume=False,
        retry_errors=False,
        watch=True,
        watch_sleep_s=0.0,
        stop_after_idle_s=0.0,
        sample_random=False,
        save_debug_bundles=False,
        pass_a_model="gpt-5-nano",
        pass_b_model="gpt-5.2",
        openai_endpoint="responses",
        openai_request_delay_s=0.0,
        max_blocks=140,
        max_total_chars=60_000,
        prompt_a_path=prompt_a,
        prompt_b_path=prompt_b,
    )

    assert outputs.paper_log_csv.exists()
    assert outputs.segments_csv.exists()
    assert _count_csv_rows(outputs.segments_csv) == 0

