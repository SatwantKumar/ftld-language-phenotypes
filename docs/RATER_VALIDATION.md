# Clinician Rater Validation (PMC OA)

This workflow supports the Phase D plan step: sample cases for clinician review, collect binary ratings (PPA + symptom tags), quantify inter-rater agreement (Cohen’s κ), and produce an adjudication sheet for disagreements.

## 1) Create a stratified sample + packets

Generate a sample from an existing Phase D run (stratified by `split × matched_cluster × ftld_syndrome_reported`):

```bash
paperjn pmcoa make-rater-sample \
  --phase-d-run-dir results/phase_d/<your_phase_d_run_dir> \
  --n 200 \
  --seed 42 \
  --text-field text_raw
```

Outputs:
- `internal/sample_manifest_internal.csv` — full metadata + packet paths (internal use; do not share)
- `share/packets/` — one `RAT####.md` per case with extracted narrative text (for clinician review; do not commit)
- `share/rater1_template.csv`, `share/rater2_template.csv` — blinded templates to fill (for clinician review; do not commit)
- `share/RATER_INSTRUCTIONS.md` — rubric + column definitions (safe to commit)
- `share/HUMAN_RATER1_WORKBOOK.xlsx`, `share/HUMAN_RATER2_WORKBOOK.xlsx` — optional Excel workbooks (do not commit)
- `internal/coverage_report.md` — sample vs full distribution (internal use)

## 2) Clinicians fill templates

Each clinician fills their own template:
- `text_adequate`: `0` or `1` (if `0`, leave all other label fields blank)
- `is_ftld_spectrum`: `0` or `1`
- `is_ppa`: `0` or `1`
- each `tag__<name>`: `0` or `1`
- leave blank if not confident / not rateable

### Optional: single-file Google Sheets workflow (recommended for ease)

Use `share/HUMAN_RATER*_WORKBOOK.xlsx`. It contains:
- `INSTRUCTIONS` tab (rubric)
- `RATINGS` tab (one row per case, with `packet_text` embedded)

Raters should fill only the rating columns; do not modify identifier columns.

## 3) Compute κ and generate adjudication sheet

```bash
paperjn pmcoa score-raters \
  --rater1-csv <filled_rater1.csv> \
  --rater2-csv <filled_rater2.csv>
```

Outputs:
- `kappa_report.csv` + `kappa_report.md`
- `adjudication_template.csv` (only cases/fields with disagreements where both raters provided values)

## 4) Adjudication

Fill `*_final` columns in `adjudication_template.csv` with the adjudicated binary label (`0/1`).

Downstream integration (rerunning Phase D on adjudicated labels) is implemented as the next step once adjudicated labels exist.

## Optional: In-silico “virtual rater” dry run (pipeline check; NOT human validation)

This creates two GPT-based virtual raters (A/B) + κ report, then runs a Phase D relabel sensitivity + Phase E pooling
using the in-silico consensus labels. Outputs are clearly labeled `INSILICO` and written under `results/phase_f_insilico/`.

```bash
paperjn pmcoa insilico-rater-study \
  --rater-sample-dir results/phase_d/validation/rater_sample__<timestamp>
```

## Safety note
Clinician packets and filled rating files contain per-case content and/or labels. Keep them out of the public
repository (they are written under `results/` and `*.xlsx` is gitignored). See `docs/PUBLIC_REPO_POLICY.md`.
