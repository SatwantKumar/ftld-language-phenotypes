# Language-Dominant Phenotypes in FTLD-Spectrum Narratives (PMC Open Access)

This repository contains a reproducible, auditable analysis pipeline (`paperjn`, under `src/paperjn/`) and
high-level analysis documentation for the study **Language-Dominant Phenotypes in the Frontotemporal Lobar
Degeneration Spectrum**.

Maintainer: Satwant Kumar (`Satwant.Dagar@gmail.com`)

## What is included (and intentionally excluded)
- Included: analysis code + CLI, locked prompts/schemas, the PMC OA query (`queries/`), a locked registry of
  PMCIDs used in the paper (`reproducibility/pmcoa_registry/`), and high-level analysis docs (`docs/`).
- Excluded (to keep the public repo safe): downloaded full text (JATS/XML), extracted narrative segments,
  embeddings, **any per-case human or AI label files**, and manuscript sources. The pipeline regenerates these
  locally.

## Setup
- Python: `>=3.10` (3.12 recommended)
- Install:
  - `pip install -e ".[dev,embeddings]"`
- Optional env vars (copy `.env.example` → `.env`):
  - `OPENAI_API_KEY` (required for extraction + Phase C labeling)
  - `NCBI_EMAIL` / `NCBI_API_KEY` (recommended for PMC/PubMed calls)

## Reproduce the PMC OA analysis
See `docs/PMC_OA_WORKFLOW.md` for the detailed workflow. The core steps are:

1. Use the locked registry (recommended):
   - `reproducibility/pmcoa_registry/registry__ftld_case_reports_oa_v1__20251227T223125Z.csv`
2. Fetch JATS/XML:
   - `paperjn pmcoa fetch-jats --registry <REGISTRY.csv> --out-dir data/interim/pmcoa/jats --all`
3. Extract narrative segments (requires OpenAI; run is large):
   - `paperjn pmcoa extract-segments --xml-dir data/interim/pmcoa/jats --registry <REGISTRY.csv> --all --no-dry-run`
4. Phase C labeling (requires OpenAI):
   - `paperjn pmcoa label-cases --segments-csv <segments.csv> --xml-dir data/interim/pmcoa/jats --all --no-dry-run`
5. Phase D split analysis (Discovery/Confirmation replication):
   - `paperjn pmcoa analyze-splits --segments-csv <segments.csv> --case-labels-csv <case_labels_long.csv> -c configs/project.yaml`

All intermediate artifacts are written to `data/` and `results/` and are **gitignored** by default.

## Safety policy
See `docs/PUBLIC_REPO_POLICY.md` for what must not be committed (per-case human/AI labels, rating workbooks,
extracted narrative text, downloaded full text).

## Citation
- See `CITATION.cff`
