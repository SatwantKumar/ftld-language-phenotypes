# Statistical Analysis Plan (SAP) — paperJN

## Objective
Identify **hypothesis-generating**, data-driven phenotypic structure in frontotemporal dementia (FTD)
from **clinical descriptions** and test whether a **language-dominant phenotype** is reproducibly
enriched for primary progressive aphasia (PPA) variants across stages and in an external literature corpus.

## Design and analysis units
- **Design**: Unsupervised discovery + confirmatory enrichment testing with selection-aware permutations.
- **Primary discovery dataset**: stage-resolved curated table of phenotypic “information units”.
- **Unit of analysis**: one information unit (not a patient).
- **Stages analyzed**: Early, Middle, Late (analyzed separately, then compared).
- **External replication dataset**: published research / clinical reports; each document (or extracted
  clinical-description segment) is a unit.

## Inclusion and preprocessing
**Curated table**
- Include rows with non-empty `Description` and valid `Temporal Stage` in {Early, Middle, Late}.
- Require non-missing `Subtype`.

**Fixed preprocessing**
- Lowercase, normalize whitespace.
- No manual row-level edits after dataset lock.

**Label-leakage prevention (mandatory)**
- Remove/replace explicit subtype identifiers and near-synonyms from all texts prior to embedding
  (see `configs/project.yaml:leakage_blacklist`).
- **Leakage audit acceptance**: automated scan confirms **0** remaining blacklist matches in analysis text.

## Representation (embeddings)
- **Primary embedding model**: `BAAI/bge-small-en-v1.5` (SentenceTransformers; 384 dimensions),
  with model name + version/hash logged.
- **Primary text**: curated **human-authored** `Description` only (no LLM enrichment).
- **Clustering representation**: cluster on a fixed PCA projection of embeddings using
  **95% variance explained, capped at 50 components**. Retain full 384-d embeddings for
  similarity/matching and reporting.
- Store embeddings in a safe format (e.g., parquet/npz), not parsed via `eval`.

## Unsupervised discovery (per stage)
For each stage independently:
1. Embed all units in that stage.
2. Reduce embeddings for clustering (PCA: 95% variance explained, capped at 50 components).
3. Cluster reduced embeddings (primary: k-means; fixed seed).
4. Use fixed `k=4` clusters for all stages (primary).
5. Compute within-stage cluster composition by subtype.

## Primary endpoint, estimands, and null hypotheses
**Primary endpoint (per stage)**
- Define the **language-dominant cluster** as the cluster maximizing:
  `PPA_share = P(svPPA) + P(nfvPPA) + P(lvPPA)` within that cluster.

**Primary estimands**
- For combined PPA and for each PPA subtype:
  - within-cluster proportion,
  - stage-level baseline proportion,
  - effect size (RR and/or OR) with 95% CI.

**Primary null hypothesis (per stage; combined PPA)**
- After accounting for clustering + cluster selection, the maximum-PPA cluster is no more enriched
  for PPA than expected under exchangeability of subtype labels within stage.

## Multiplicity control
- **Primary family**: 3 stage-level tests for combined PPA enrichment (Early/Middle/Late); BH-FDR q=0.05.
- **Secondary**: subtype-specific tests within stage; BH-FDR within stage.

## Selection-aware permutation inference (primary)
For each stage, run `B` permutations (target B=5000; minimum 1000 if compute-limited).

For each permutation:
1. Hold the (unsupervised) stage-specific clustering fixed (clustering does not use phenotype labels).
2. Permute PPA/non-PPA labels within that stage.
3. Re-select the max-PPA cluster under permuted labels and record the permuted max statistic
   (max PPA_share across clusters).

Compute one-sided enrichment p-values:
`p = (1 + #{perm_stat >= obs_stat}) / (1 + B)`.

Apply BH-FDR as specified above.

## External replication (literature corpus)
**Construction**
- Pre-specify search sources, query terms, date range, inclusion/exclusion.

**Phase C v1 labeling (PMC OA)**
- For PMC OA full text, derive analysis labels at the **case unit** level (`pmcid+case_id`) using a locked schema
  (`docs/PHASE_C_LABELING.md`).
- Primary phenotype endpoint for replication: `is_ppa`, derived from `ftld_syndrome_reported`.
- Pre-specified symptom tags are **exploratory cluster characterization** only (not used to define the primary endpoint).
- Human validation: dual blinded expert labeling on a stratified random subset, with Cohen’s κ reported; for symptom tags,
  the primary κ endpoint is binary (`present` vs `not-present/unknown`).

**Embedding + assignment**
- Embed literature texts with the same model (`BAAI/bge-small-en-v1.5`).
- Assign each unit to the closest language centroid (or a pooled language centroid defined a priori).

**Literature null hypothesis**
- Assignment to the language phenotype is independent of PPA label under the fixed assignment rule.

**Permutation**
- Permute PPA/non-PPA labels among literature units; recompute enrichment under fixed assignment.

## Replication acceptance criteria
Replication is achieved if:
1. **Across stages**: in each stage, selection-aware permutation + BH-FDR gives q<0.05 for combined PPA,
   and a pre-specified minimum effect size is met (e.g., RR ≥ 2.0).
2. **Across literature**: enrichment of PPA-labeled literature units for assignment to the language phenotype
   is significant (permutation p<0.05; FDR per locked plan) and meets a minimum effect (e.g., OR ≥ 2.0).
3. **Robustness**: direction of effect persists under ≥2 pre-specified sensitivity settings.

## Sensitivity analyses (locked list)
- **Embedding family**: `intfloat/e5-small-v2` (SentenceTransformers; 384 dimensions), all other steps identical
  (leakage blacklist, PCA rule, k=4).
- **k robustness**: repeat the primary pipeline with `k ∈ {3,4,5,6}` (fixed per run) and report whether the
  combined-PPA enrichment direction persists and whether results remain statistically supported.

## Supportive analyses (non-primary; pre-specified)
These analyses do not alter the locked primary endpoint; they contextualize the Early-stage result and
quantify stability.

**Early diagnostics**
- Report cluster sizes and PPA shares in Early (k=4).
- Plot the Early permutation null distribution of the max PPA share statistic.

**Early cross-stage prototype assignment**
- Build reference centroids from Middle+Late clusters in the full embedding space (384-d).
- Assign Early units to their nearest reference centroid (cosine similarity; fixed assignment rule).
- Test PPA enrichment among Early units assigned to a reference “language” centroid using permutation
  of Early PPA labels under the fixed assignment.

**Stability**
- Subsample-based stability (without replacement) on the fixed PCA representation: report ARI vs baseline
  clustering and Jaccard overlap of the “language cluster” membership.
- Seed sensitivity: re-fit stage-wise k-means across a pre-specified set of seeds and report ARI vs baseline.
