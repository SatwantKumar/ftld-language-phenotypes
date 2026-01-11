# Top-Shelf Storyline Plan (Conditional on Human Clinician Validation Success)

This plan assumes the clinician validation substudy is completed and achieves high agreement (eg, Cohen κ ≥0.80 for primary `is_ppa`, with strong supporting agreement for `is_ftld_spectrum` and `text_adequate`). Until then, this is a *conditional* manuscript framing and figure plan.

## Core “Big Claim” (What the field should remember)
Published clinical narratives contain a **replicable language-dominant phenotype signal** across the FTLD spectrum that can be recovered *without supervised training* using diagnosis-redacted narrative embeddings, and this signal remains after **blinded clinician validation** of key labels.

## 3-Part Message Hierarchy (Keep reviewers oriented)
1) **Clinical/biologic motivation:** Language-led phenotypes cut across syndromes and matter for prognosis/trials; narrative descriptions encode this structure but are hard to aggregate at scale.
2) **Method contribution (rigor):** A prespecified, leakage-audited, discovery/confirmation framework can extract and confirm phenotype structure from open-access narratives with transparent robustness and stability quantification.
3) **Validation + generalizability:** Clinician validation supports label accuracy; replication and stability analyses bound generalizability; the framework is reusable for other phenotypes and external corpora.

## How Validation Changes the Paper (Upgrade path)
### What becomes a stronger “evidentiary anchor”
- **Primary endpoint repeated on adjudicated labels:** rerun the locked Confirmation test using adjudicated `is_ppa` (and optionally adjudicated `is_ftld_spectrum` as a stricter inclusion filter).
- **Report agreement transparently:** Cohen κ for `is_ppa` (primary), and κ (or percent agreement) for supporting fields; report adjudication rules and missingness (eg, text inadequate).
- **Demonstrate robustness to label noise:** show the primary endpoint effect size/p-value stability when using (a) LLM labels, (b) clinician labels, (c) adjudicated consensus labels.

### What stays explicitly nonconfirmatory
- Symptom-tag profiles, imaging/genetics/pathology summaries remain **descriptive/exploratory characterization**, not a new “p-value story.”

## Manuscript Packaging (JAMA Neurology constraints)
Keep the main article within ≤3000 words and ≤5 tables/figures; move transparency artifacts to the Supplement.

### Recommended main tables/figures (maximizes impact per slot)
- **Table 1:** Cohort assembly and analytic cohort (article-level + case-level flow).
- **Table 2:** Primary + key secondary replication endpoints (Discovery/Confirmation; P values; effect sizes).
- **Table 3 (new):** Clinician validation summary (n rated, missing/“text inadequate”, κ for `is_ppa` and supporting fields; adjudication outcomes).  
  - If tables are limited, convert to **Figure 2 panel** or move to Supplement but keep at least κ for the primary label in main text.
- **Figure 1:** Cluster matching across splits (centroid similarity matrix) with explicit matched-cluster IDs.
- **Figure 2:** Primary result visualization (forest plot for Confirmation ORs + CI for PPA and prespecified syndromes) **plus** clinician validation κ (small panel) if not Table 3.
- **Figure 3:** Cluster characterization heatmap (symptom tags by matched cluster, split-stratified) with explicit “descriptive/exploratory” labeling.

### Recommended supplement (transparency that impresses reviewers)
- **Specification curve** (k, PCA fit, inclusion tier, embedding family): shows boundary conditions without adding endpoints.
- **Permutation null plots** (Discovery selection-aware, Confirmation fixed-cluster): makes inference legible.
- **Consensus co-assignment** (bootstrap heatmaps) + “core language cases” definition: shows stability and acknowledges uncertainty.
- Full extraction/label schemas, prompts checksums, and reproducibility logs.

## Narrative Edits (High ROI wording shifts)
- **Avoid “we cluster diagnoses”:** consistently say “we clustered diagnosis-redacted narrative embeddings; labels used only for post hoc enrichment testing.”
- **Make the inferential logic explicit:** Discovery = construct identification; Confirmation = inference; validation = label accuracy; pooled = precision only.
- **Use conservative causal language:** “associated with / enriched for / replicates” (no “predicts” or “improves outcomes”).

## Reviewer Anticipation (Preempt the top critiques)
- **Bias:** acknowledge literature/reporting bias; state why the design still supports inference about replicability *within this corpus*; position external replication as next step.
- **Label validity:** clinician κ + adjudication directly address the main critique.
- **Model dependence:** show it transparently (spec curve) and keep the claim bounded (replicable construct, not universal taxonomy).

## Execution Checklist (after validation results arrive)
1) Freeze the rater file versions and adjudication output; write a validation report with κ and adjudication counts.
2) Re-run Phase D (locked) using adjudicated `is_ppa` (and optionally adjudicated `is_ftld_spectrum` filter).
3) Update Table 2 (or add a small sensitivity table) showing that the primary endpoint persists under adjudicated labels.
4) Update main text (Methods + Results + Limitations) to include validation details and avoid overstating generalizability.
5) Update supplement figures/tables; regenerate submission DOCX; run full consistency audit.

