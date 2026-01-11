# Reproducibility Notes

## Principles
- One config file (`configs/project.yaml`) controls the pipeline.
- Randomness is controlled via a single `random_seed`.
- Any change to inputs, preprocessing, embedding model, clustering, or inference increments a run id and
  produces a new snapshot under `results/snapshots/`.

## Minimum reproducibility checklist (for submission)
- Dataset lock: checksum + provenance for `data/raw/curated_ftd_table.csv`.
- Blacklist audit: saved report showing zero subtype-token leakage after cleaning.
- Embedding model(s): primary `BAAI/bge-small-en-v1.5` (384d); sensitivity `intfloat/e5-small-v2` (384d);
  record name + version/hash for each run.
- Clustering representation: PCA (95% variance explained, capped at 50 components) + fixed `k=4` primary.
- Full command log: stored under `results/logs/`.
