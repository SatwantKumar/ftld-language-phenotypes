# Phase C v1 — Case-Level Labeling (PMC OA)

Phase C converts extracted clinical narrative segments into **analysis-ready case labels** for:
1) defining the **FTLD-spectrum cohort** (inclusion/exclusion), and
2) anchoring **phenotype endpoints** (e.g., PPA vs non-PPA) for enrichment/replication testing,
while preserving **diagnosis-blind clustering** on redacted text.

## Scope (locked for v1)
**Unit of labeling**
- One row per `pmcid + case_id` case unit (`case_unknown` allowed for unsplittable papers).

**Label families included**
- **FTLD inclusion tier**: `exclude` vs `ftld_strict` vs `ftld_broad` (primary uses strict only).
- **FTLD phenotype**: `ftld_syndrome_reported` (primary endpoint derives `is_ppa`).
- **Duration/onset (minimal)**: only if explicitly stated (else null).
- **Imaging (coarse)**: modality present + laterality + broad region flags when stated.
- **Pathology/genetics (coarse)**: presence/type when stated.
- **Initial diagnosis (minimal)**: misdiagnosis flag + initial dx category when stated.
- **Symptom tags (exploratory)**: pre-specified high-signal tags; used to characterize clusters, not to
  define the primary endpoint.

Locked schema (pydantic model + allowed values): `src/paperjn/pmcoa/phase_c_schema_v1.py`

## Symptom tags (v1, pre-specified)
Each tag is rated with a 4-level status plus per-tag evidence pointers:
- **Status**: `present` | `explicitly_absent` | `not_reported` | `uncertain`
- **Evidence**: 1–3 `block_id` values supporting the call

Tags:
- `apraxia_of_speech`
- `agrammatism`
- `semantic_loss`
- `behavioral_change_disinhibition_or_apathy`
- `compulsions_or_rigid_routines`
- `parkinsonism`
- `oculomotor_vertical_gaze_palsy`
- `limb_apraxia_or_alien_limb`
- `mnd_signs`
- `psychosis_hallucinations`

## Rater decision rules (locked)
For each symptom tag:
- `present`: explicitly described (symptom/sign stated, exam finding documented, or clear diagnosis statement).
- `explicitly_absent`: explicit negation (e.g., “no hallucinations”, “no gaze palsy”, “no UMN/LMN signs”).
- `not_reported`: no mention either way.
- `uncertain`: equivocal/hedged language (“possible…”, “mild…?”, “suggestive of…”) or conflicting statements.

## Reliability evaluation (publication-ready)
Because many case reports do **not** document negatives, the primary reliability endpoint for each tag is:

**Binary kappa (primary):** `present` vs `not-present/unknown`
- Collapse: `present` → 1; `explicitly_absent`/`not_reported`/`uncertain` → 0
- Rationale: measures “did we detect the feature?” in a way robust to missingness.

**4-level agreement (secondary):**
- Report percent agreement + confusion matrix; 4-level κ is descriptive only.

**Acceptance criteria (symptom tags):**
- Tags with κ < 0.60 or very low prevalence are reported descriptively and not used to support strong mechanistic claims.

## How labels enter the scientific inquiry
- **Clustering** uses `text_clean` (diagnosis-leakage terms removed).
- **Endpoints** (e.g., PPA enrichment) use `ftld_syndrome_reported` derived labels from Phase C.
- **Symptom tags** are used to interpret clusters (exploratory characterization, not the main p-value story).

## Running Phase C v1 (LLM rater)
From the project root:
- `paperjn pmcoa label-cases --segments-csv data/interim/pmcoa/extractions/<run>/segments.csv --xml-dir data/interim/pmcoa/jats --out-dir results/phase_c/label__<timestamp> --no-dry-run`

Scaling/reliability options:
- `--all` to label all case units (ignore `--n`).
- `--resume --retry-errors` to safely continue after interruption.
- `--openai-request-delay-s 0.3` (or higher) to be conservative with rate limits.
- `--save-debug-bundles` to store per-case inputs/responses for auditing (can be large).

Outputs:
- `results/phase_c/<run>/case_labels_long.csv`
- `results/phase_c/<run>/performance_report.md`
- `results/phase_c/<run>/run_config.json`

Note: `case_labels_long.csv` contains **per-case labels** and must not be committed to the public repository.
See `docs/PUBLIC_REPO_POLICY.md`.
