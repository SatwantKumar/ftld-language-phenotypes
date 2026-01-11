# Literature Corpus Protocol (Template)

## Objective
Construct an external corpus of published research and clinical reports with extractable clinical-description
text to test replication of the language-dominant phenotype.

## Sources
- PubMed / MEDLINE
- (Optional) Semantic Scholar / Crossref
- (Optional) journal sites for case reports/series (respect licensing)

## Search strategy (to be locked)
- Query terms:
  - frontotemporal dementia OR FTD
  - primary progressive aphasia OR PPA OR semantic variant OR logopenic OR nonfluent
  - case report OR case series OR clinical features
- Date range:
- Language restrictions:

## Screening
- Inclusion criteria:
  - Contains clinical description narrative (case description, symptom narrative, phenotype description).
  - Mentions or implies subtype label(s) for gold-labeling (PPA vs non-PPA).
- Exclusion criteria:
  - No clinical description content.
  - Editorials/commentaries without clinical phenotype text.

## Extraction
- Extract clinical description segments (methods to be defined: abstract-only vs full text).
- Apply the same subtype-term blacklist removal used in the curated analysis.

## Deliverables
- PRISMA-style flow summary (counts by screening step)
- A structured corpus file with:
  - document id, title, year, source, subtype label(s), extracted text, and provenance.

## Analysis-ready corpus file (for `paperjn`)
The pipeline expects a single CSV (default path: `data/processed/literature_corpus.csv`; configurable in
`configs/project.yaml` under `literature:`) with at minimum:

- `doc_id` (string/int; unique id per document/segment)
- `text` (string; extracted clinical description segment)
- `is_ppa` (boolean/int; gold label for PPA vs non-PPA for replication testing)

Notes:
- The analysis applies the same `leakage_blacklist` redaction rules used for the curated table and fails if
  any blacklist matches remain after redaction.
- Replication uses a fixed nearest-centroid assignment rule (centroids derived from the curated discovery run),
  and permutation inference by shuffling `is_ppa` labels among literature units.
