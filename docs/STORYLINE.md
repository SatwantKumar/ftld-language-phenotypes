# Storyline Notes (Working)

This document summarizes (1) what the study currently demonstrates, (2) why it matters scientifically, and (3) what is strongest/weakest from a reviewer perspective. It is intended to guide manuscript framing and figure selection.

## What We Did (Design + Data)
- **Goal:** Identify hypothesis-generating, text-derived phenotype structure in the frontotemporal lobar degeneration (FTLD) spectrum using published clinical descriptions, and test replication in an independent corpus split.
- **Source corpus:** PubMed Central Open Access (PMC OA) full text (JATS/XML).
- **Pipeline (locked):**
  1) Retrieve full text → parse to paragraph blocks with section-path metadata.
  2) **Pass A (routing):** detect “case-like” articles and candidate case-presentation sections using a conservative prompt plus deterministic patient-level anchor guardrail.
  3) **Pass B (extraction):** extract verbatim clinical narrative blocks into prespecified segment types; only narrative segment types are embedding-eligible.
  4) **Leakage prevention:** redact diagnosis/subtype terms from embedding-eligible text (audit acceptance criterion: 0 remaining matches).
  5) **Phase C labeling:** extract author-reported FTLD inclusion tier + syndrome + symptom tags (labels not used for clustering).
  6) **Phase D analysis:** embed narrative text, cluster cases (k=4, fixed PCA projection), match clusters across splits, and test replication using prespecified permutation tests.
- **Discovery/Confirmation split (prespecified):** publication year ≤2021 vs >2021.

## Core Results (Phase D/E; frozen)
### Cohort assembly (strict analytic cohort)
- PMC OA query hits with JATS retrieved: **5356** articles.
- Articles contributing ≥1 extracted narrative segment: **932**.
- Case units labeled: **1267**.
- FTLD strict: **280** case units (150 Discovery; 130 Confirmation).
- Analytic cohort (FTLD strict + ≥1 embedding-eligible narrative segment): **278** case units (149 Discovery; 129 Confirmation).

### Primary replication endpoint (PPA enrichment in the matched “language-dominant” cluster)
- **Discovery:** language-dominant cluster contained **24/29 (83%)** primary progressive aphasia (PPA) cases vs **11/120 (9.2%)** outside; selection-aware permutation **p≈0.0028**; OR **≈42.4**.
- **Confirmation:** matched cluster contained **14/41 (34%)** PPA vs **13/88 (15%)** outside; fixed-cluster permutation **p≈0.013**; OR **≈3.0**.
- **Interpretation:** a language-dominant *text phenotype construct* is recoverable without supervised training and replicates across a time-split corpus; effect size is smaller in Confirmation, consistent with realistic heterogeneity/noise.

### Secondary (prespecified) syndrome endpoints
- **Progressive supranuclear palsy (PSP):** enrichment replicated in Confirmation after Benjamini–Hochberg FDR control (q≈0.0012; OR **≈7.3**).
- Other prespecified syndrome endpoints did not meet the locked Confirmation multiplicity threshold (some show nominal signals).

### Stability/robustness (Phase D + Phase G visualizations)
- **Sensitivity:** primary replication persists for **k=3** and for **include_broad + PCA fit on all**, attenuates at higher k (5–6) and does **not** replicate with an independent embedding family (E5).
- **Bootstrap stability:** partition stability is higher in Discovery than Confirmation (median ARI ~0.79 vs ~0.53), while language-cluster membership is more stable than the full partition (median Jaccard ~0.88 vs ~0.68).
- **Consensus membership (Phase G):** a stable “core” subset of cases remains in the language construct across bootstraps (membership probability ≥0.8; Discovery=28, Confirmation=21).

### Post-replication pooled characterization (Phase E; descriptive only)
- After replication criteria are met, pooled estimates improve precision for describing cluster composition, but **no pooled hypothesis tests** are performed (split-based inference remains the evidentiary anchor).

## Scientific Impact (Why This Matters)
- **New data substrate:** turns narrative clinical descriptions in the biomedical literature into an analyzable phenotype corpus at scale, without manual chart abstraction.
- **Design-based rigor:** uses an explicit discovery/confirmation framework with selection-aware inference (Discovery) and a fixed-cluster confirmatory test (Confirmation), reducing researcher degrees of freedom.
- **Clinical relevance:** demonstrates that language-led descriptions contain a reproducible signal even amid cross-syndrome overlap—useful for hypothesis generation about biology, prognosis, and trial stratification.
- **Generalizable framework:** the same pipeline can be extended to other syndromes/phenotypes and future corpora (including external replication datasets).

## What’s Strong for Reviewers
- **Prespecified split + locked primary analysis** (k, PCA rule, embedding family, matching, endpoints, permutation schemes).
- **Leakage auditing** with an explicit acceptance criterion (0 diagnosis-term matches in embedding-eligible text).
- **Transparent robustness** (k-sensitivity, PCA fit sensitivity, embedding-family sensitivity) and stability quantification (ARI/Jaccard, co-assignment).
- **Reproducible artifacts** (run directories, logs, prompts/schemas checksums, tables/figures regenerated from frozen outputs).

## What’s Weak / Likely Reviewer Concerns (and how we address them)
- **Publication/reporting bias:** literature cases are not population-representative; address by framing as hypothesis-generating phenotyping and emphasizing planned external replication.
- **Label accuracy:** author-reported diagnoses + LLM extraction can be wrong; address with blinded clinician validation (pending) and reporting κ and adjudication rules.
- **Model dependence:** replication attenuates for some k values and fails with an independent embedding family; address by explicitly presenting this as boundary conditions (specification curve) and keeping claims conservative.
- **Confirmation instability:** lower stability in Confirmation suggests noise/heterogeneity; address by emphasizing matched-cluster replication of a prespecified construct, not a definitive “taxonomy.”

## Bottom-Line Claim (Current, Defensible)
In an open-access corpus of published FTLD-spectrum clinical narratives, diagnosis-redacted narrative embeddings recover a language-dominant phenotype construct that replicates in an independent time-split confirmation corpus under a locked, permutation-based inference framework; additional syndrome patterns (PSP) are detectable, while several endpoints do not replicate, underscoring both promise and limits of literature-derived phenotyping.

