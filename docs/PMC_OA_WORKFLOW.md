# PMC Open Access (PMC OA) Corpus Workflow

This workflow builds a **reviewer-proof**, auditable paper registry and a manual download manifest
for a **time-split Discovery vs Confirmation** corpus drawn from the **PMC Open Access subset**.

## 1) Run the search (registry)
The search query is stored in `queries/pmcoa_search.yaml` for audit/reproducibility.

From the project root:

- `paperjn pmcoa search -q queries/pmcoa_search.yaml`

Optional (recommended) environment variables for NCBI E-utilities:
- `NCBI_TOOL` (e.g., your registered tool name)
- `NCBI_EMAIL` (contact email)
- `NCBI_API_KEY` (optional; increases rate limits)

This writes:
- `data/interim/pmcoa/registry__<query_id>__<timestamp>.csv`
- `data/interim/pmcoa/year_counts__<query_id>__<timestamp>.csv`
- `data/interim/pmcoa/split_recommendation__<query_id>__<timestamp>.json`

## 2) Create a manual-download manifest (PDFs)
- `paperjn pmcoa make-manifest -r data/interim/pmcoa/registry__<query_id>__<timestamp>.csv`

This writes:
- `data/interim/pmcoa/download_manifest__<query_id>__<timestamp>.csv`

and creates the default PDF folder:
- `data/external/pmcoa_pdfs/raw/`

## 3) Manual screening + download
Open the manifest CSV and fill in:
- `screen_include`: `y`/`n` (based on your inclusion/exclusion rules)
- `screen_reason`: optional rationale
- `download_status`: `todo` → `downloaded` → `failed` (as you download PDFs)

Save each PDF to the `local_pdf_path` location. Filenames are deterministic and include the PMCID.

## 4) Recommended alternative: skip PDFs (JATS/XML full text)
PMC OA articles can be fetched as full text **JATS/XML** via NCBI E-utilities.

- `paperjn pmcoa fetch-jats -r data/interim/pmcoa/registry__<query_id>__<timestamp>.csv --n 60 --seed 42`

This writes XML files to:
- `data/interim/pmcoa/jats/PMC*.xml`

and an auditable fetch log:
- `data/interim/pmcoa/jats/jats_fetch_log__<timestamp>.csv`

Batch pacing options (recommended for long runs):
- `--batch-size 25 --batch-sleep-s 10 --batch-jitter-s 2`

## 5) Next steps (planned)
After you have JATS/XML (recommended), the pipeline proceeds in two auditable steps:

### 5a) Parse JATS/XML → paragraph blocks (deterministic)
- `paperjn pmcoa parse-jats --xml-dir data/interim/pmcoa/jats`

This writes per-paper JSONL blocks + metadata:
- `data/interim/pmcoa/blocks/PMC*.blocks.jsonl`
- `data/interim/pmcoa/blocks/PMC*.meta.json`

and an auditable parse log:
- `data/interim/pmcoa/blocks/jats_parse_log__<timestamp>.csv`

### 5b) Extract clinical narrative segments (2-pass LLM)
Environment variable:
- `OPENAI_API_KEY` (required for real runs)

The CLI will also **auto-load** a `.env` from common locations (the project root `.env` or the parent folder’s `.env`) if `OPENAI_API_KEY` is not already set, or you can pass `--env-file /path/to/.env`.

Dry-run (no API calls; uses heuristics to validate plumbing):
- `paperjn pmcoa extract-segments --xml-dir data/interim/pmcoa/jats --dry-run`

Real run:
- `paperjn pmcoa extract-segments --xml-dir data/interim/pmcoa/jats --no-dry-run`

Outputs (one run folder per invocation):
- `data/interim/pmcoa/extractions/extract__<timestamp>/paper_log.csv`
- `data/interim/pmcoa/extractions/extract__<timestamp>/segments.csv`
- `data/interim/pmcoa/extractions/extract__<timestamp>/run_config.json`

Optional (debug, contains full text snippets; can be large):
- add `--save-debug-bundles`

### 5c) Expert validation (planned)
Dual expert ratings on extraction quality (blinded), with agreement reported (weighted kappa for ordinal ratings).

### 5d) Phase C (v1): case-level phenotype labeling
After segments are extracted, Phase C assigns **case-level FTLD inclusion + phenotype labels** and a small set of
**pre-specified symptom tags** for exploratory cluster characterization.

See: `docs/PHASE_C_LABELING.md`

Run (recommended):
- `paperjn pmcoa label-cases --segments-csv data/interim/pmcoa/extractions/<run>/segments.csv --xml-dir data/interim/pmcoa/jats --out-dir results/phase_c/label__<timestamp> --no-dry-run`

Resume after interruption:
- `paperjn pmcoa label-cases --segments-csv data/interim/pmcoa/extractions/<run>/segments.csv --xml-dir data/interim/pmcoa/jats --out-dir results/phase_c/label__<timestamp> --resume --retry-errors --no-dry-run`

Outputs:
- `results/phase_c/<run>/case_labels_long.csv`
- `results/phase_c/<run>/performance_report.md`
- `results/phase_c/<run>/run_config.json`

### 5e) Phase D (v1): split-sample clustering + replication
Phase D runs the locked primary analysis on the **FTLD-strict** subset:
- Cluster **Discovery** cases (k=4, PCA fit on Discovery).
- Match clusters to **Confirmation** via centroid cosine similarity.
- Test PPA enrichment (selection-aware in Discovery; fixed matched cluster in Confirmation).

Run:
- `paperjn pmcoa analyze-splits --segments-csv data/interim/pmcoa/extractions/<run>/segments.csv --case-labels-csv results/phase_c/<run>/case_labels_long.csv --out-dir results/phase_d/pmcoa_split__<timestamp> -c configs/project.yaml`

Outputs:
- `results/phase_d/<run>/performance_report.md`
- `results/phase_d/<run>/tables/replication_summary.csv`
- `results/phase_d/<run>/tables/case_table_with_clusters.csv`
