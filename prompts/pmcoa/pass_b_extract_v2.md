You are extracting **patient-level clinical narrative** phenotype text from a paper's JATS/XML body.

You will be given a list of paragraph blocks. Each block has:
- `block_id` (stable id)
- `sec_path_str` (section path)
- `text` (verbatim paragraph text)

Goal:
Select `block_ids` that form standardized clinical narrative segments. Do NOT paraphrase or rewrite.
You must only reference `block_ids` from the provided list.

Hard constraints (important):
- Each `block_id` may appear in **at most one** segment across the entire output.
- Prefer building segments from **contiguous** blocks in the original order when possible.
- Exclude blocks that are primarily methods, references, funding/COI, or generic background.
- If you do not see patient-level narrative, return `segments=[]`.

Case splitting (conservative):
- Only split into `case_1` / `case_2` / ... if the provided text/section path explicitly delineates cases (e.g., “Case 1”, “Patient 2”).
- Otherwise set `case_id = "case_unknown"` for all segments.

Segment types (choose from this fixed list):
- `case_presentation` (include_for_embedding=true)
- `disease_course` (include_for_embedding=true)
- `clinical_features_summary` (include_for_embedding=true)
- `neuropsych_language_testing` (include_for_embedding=false)
- `imaging_biomarkers` (include_for_embedding=false)
- `pathology_genetics` (include_for_embedding=false)
- `treatment_response` (include_for_embedding=false)
- `other` (include_for_embedding=false)

Assignment rule for mixed-content paragraphs:
- If a paragraph contains mixed content (e.g., symptoms + genetics + treatment), assign it to **one best** segment type (prefer `case_presentation` / `disease_course` / `clinical_features_summary`) and do **not** duplicate it across multiple segment types.

Output format:
- Return a single JSON object and nothing else.
- Keys exactly: `n_cases_est` (integer or null), `segments` (array).
- Each segment object keys exactly: `case_id`, `segment_type`, `block_ids` (array), `include_for_embedding` (boolean).
- `block_ids` must be non-empty and must come from the provided list.
