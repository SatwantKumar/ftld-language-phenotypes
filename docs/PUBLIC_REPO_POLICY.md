# Public Repository Policy (Safe Sharing)

This repository is intended to be **public** and **replicable** while avoiding redistribution of per-case labels,
clinician rating artifacts, and manuscript sources.

## Must not be committed
- Downloaded full text (PMC JATS/XML) and any derived copies.
- Extracted narrative segments (`segments.csv`) and any case-packet text exports.
- **Per-case labels** produced by Phase C (LLM labels) or by clinician ratings.
- Clinician workbooks/templates containing filled ratings (eg, `*.xlsx`).
- Manuscript sources (markdown, figures, docx) and any submission packages.

These artifacts are regenerated locally and are written under `data/` and `results/`, which are gitignored.

## OK to commit
- Code, locked prompts/schemas, and configuration.
- The PMC OA query definition (`queries/pmcoa_search.yaml`).
- High-level analysis documentation (eg, `docs/`).
- A locked list of PMCIDs used for retrieval (`reproducibility/pmcoa_registry/`), which contains bibliographic
  metadata only (no per-case labels).

## Before publishing (recommended)
Run a quick check to ensure nothing sensitive is staged:
- `git status`
- `git diff --staged`
- `python scripts/public_repo_audit.py --check`
