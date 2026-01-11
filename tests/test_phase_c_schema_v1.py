import pytest

from paperjn.pmcoa.phase_c_schema_v1 import CaseLabelV1, tag_status_to_binary_present


def test_tag_status_to_binary_present_collapse() -> None:
    assert tag_status_to_binary_present("present") == 1
    assert tag_status_to_binary_present("explicitly_absent") == 0
    assert tag_status_to_binary_present("not_reported") == 0
    assert tag_status_to_binary_present("uncertain") == 0


def test_case_label_v1_requires_evidence_block_ids() -> None:
    with pytest.raises(Exception):
        CaseLabelV1(
            pmcid="PMC1",
            case_id="case_unknown",
            ftld_inclusion_tier="exclude",
            label_confidence="high",
            evidence_block_ids=[],
        )

