# Data Dictionary (Draft)

## Curated table (`data/raw/curated_ftd_table.csv`)
Minimum required columns (primary analysis):
- `Subtype` (string): clinical subtype label (e.g., bvFTD, PPA variants, PSP/CBS, FTD-MND, rtFTD).
- `Temporal Stage` (categorical): `Early`, `Middle`, `Late`.
- `Description` (string): human-authored clinical description text for the information unit.

Recommended additional columns (interpretation/supplemental):
- `Category` (string): domain/category (e.g., Cognitive Deficits, Behavioral Symptoms, Neurological Symptoms).
- `Symptom` (string): symptom/feature name.
- `Pathology` (string): pathology descriptor (if available).
- `Region` (string): neuroanatomic region descriptor (if available).
- `References` (string): citations/URLs (if available).

## Derived artifacts (planned)
- `data/processed/curated_clean.parquet`: cleaned text + leakage-audit metadata.
- `data/processed/embeddings_{model}.npz`: embedding matrix + row ids.
- `results/tables/`: enrichment tables, stability summaries, literature replication tables.
- `results/figures/`: publication-quality figures.

## PMC OA extraction artifacts (`data/interim/pmcoa/extractions/<run>/`)
- `paper_log.csv`: one row per PMCID with Pass A/B status and counts.
- `segments.csv`: one row per extracted segment (the analysis “text unit” for downstream aggregation to case/paper level).

## Phase C v1 labeling artifacts (`results/phase_c/<run>/`)
- `case_labels_long.csv`: one row per `pmcid+case_id` per rater (LLM + humans), with evidence pointers.
- `performance_report.md`: QC/performance summary for the run (latest per case; resume-safe).
- `run_config.json`: run metadata (models, limits, prompt SHA256) for audit/reproducibility.
- `case_labels_adjudicated.csv` (planned): one row per `pmcid+case_id` after adjudication; joins into analysis.
