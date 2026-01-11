import pandas as pd

from paperjn.pmcoa.extraction_schema import NanoRouteOutput
from paperjn.pmcoa.llm_extraction import pass_a_case_like_guardrail


def test_pass_a_guardrail_accepts_case_narrative() -> None:
    blocks = pd.DataFrame(
        [
            {
                "source": "body",
                "sec_path_str": "2. Case Report",
                "block_index": 0,
                "text": "A 64-year-old man presented with progressive speech difficulty.",
            }
        ]
    )
    out = NanoRouteOutput(
        is_case_like=True,
        n_cases_est=None,
        selected_sec_paths=["2. Case Report"],
        case_markers_found=[],
    )
    passed, markers = pass_a_case_like_guardrail(out, blocks)
    assert passed is True
    assert any("Case Report" in m for m in markers)


def test_pass_a_guardrail_rejects_group_level_patients_section() -> None:
    blocks = pd.DataFrame(
        [
            {
                "source": "body",
                "sec_path_str": "Results > Baseline demographic characteristics of all patients with INPH",
                "block_index": 0,
                "text": "From 2015 to 2019, 104 patients were evaluated. The median age was 75 years.",
            }
        ]
    )
    out = NanoRouteOutput(
        is_case_like=True,
        n_cases_est=None,
        selected_sec_paths=["Results > Baseline demographic characteristics of all patients with INPH"],
        case_markers_found=[],
    )
    passed, markers = pass_a_case_like_guardrail(out, blocks)
    assert passed is False
    assert markers == ["no_narrative_anchor"]

