You are screening a scientific paper's full text for **patient-level clinical narrative** suitable for phenotype extraction.

You will be given:
- `title`
- `article_type` (if available)
- `candidate_sections`: a list of section paths with short previews

Definitions (be conservative):
- **Case-like = YES** only if the paper contains *patient-level narrative* (e.g., “A 62-year-old…”, “Case 1…”, “Patient 2…”, “case report”, “case presentation”, “case series” with individual patients).
- **Case-like = NO** for: cohort/registry analyses, clinical trials, cross-sectional studies, meta-analyses, narrative reviews, guidelines, methods papers, curricula, or any paper that only reports group-level statistics (e.g., “n=64”, medians, tables) without individual case narratives.

Task:
1) Decide whether the paper is **case-like** (boolean).
2) Estimate number of cases ONLY if the **section paths or previews explicitly delineate cases** (e.g., “Patient 1”, “Case 2”). Otherwise return null.
3) Select the subset of **section paths** that most likely contain patient narrative content.

Selection rules:
- Select ONLY from the provided `section_path` values (exact string match).
- Prefer sections whose heading/path or preview explicitly indicates patient narrative: “case”, “patient”, “case report”, “case presentation”, “clinical course”, “history”, “clinical features”.
- Do NOT select generic sections like “Results”, “Discussion”, or “Methods” unless they are explicitly subdivided into patient/case subsections and the preview looks narrative.
- If you cannot identify any patient-narrative sections, set `is_case_like=false` and `selected_sec_paths=[]`.

Output format:
- Return a single JSON object and nothing else.
- Keys exactly: `is_case_like` (boolean), `n_cases_est` (integer or null), `selected_sec_paths` (array of strings), `case_markers_found` (array of strings).
