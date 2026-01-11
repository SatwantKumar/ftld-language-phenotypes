import pandas as pd

from paperjn.pmcoa.routing import route_candidate_sections


def test_route_candidate_sections_handles_plurals_and_excludes_methods() -> None:
    blocks = pd.DataFrame(
        [
            {
                "source": "body",
                "sec_path_str": "Methods",
                "block_id": "b0",
                "n_chars": 100,
                "block_index": 0,
            },
            {
                "source": "body",
                "sec_path_str": "Clinical Features in Patients",
                "block_id": "b1",
                "n_chars": 200,
                "block_index": 1,
            },
            {
                "source": "body",
                "sec_path_str": "References",
                "block_id": "b2",
                "n_chars": 50,
                "block_index": 2,
            },
        ]
    )

    routed = route_candidate_sections(blocks)
    assert "Methods" not in routed.must_include
    assert "Methods" not in routed.candidates
    assert "References" not in routed.must_include
    assert "References" not in routed.candidates
    assert "Clinical Features in Patients" in (routed.must_include + routed.candidates)

