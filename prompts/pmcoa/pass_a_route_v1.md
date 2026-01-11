You are screening a scientific paper's full text for *clinical narrative* sections (case presentation / symptoms / clinical course).

You will be given:
- title
- article_type (if available)
- a list of candidate section paths with short previews

Task:
1) Decide whether the paper is "case-like" (contains patient/case narrative suitable for clinical-phenotype extraction).
2) Estimate number of cases ONLY if the section paths/previews explicitly indicate case numbering (e.g., "Case 1", "Patient 2"). Otherwise return null.
3) Select the subset of section paths that most likely contain patient narrative content (not methods/background/references).

Output format:
- Return a single JSON object and nothing else.
- Keys exactly: is_case_like (boolean), n_cases_est (integer or null), selected_sec_paths (array of strings), case_markers_found (array of strings).
- selected_sec_paths must be chosen from the provided section_path values.

