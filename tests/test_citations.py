from __future__ import annotations

import pytest

from paperjn.citations import (
    PubMedArticle,
    PubMedAuthor,
    extract_cite_keys,
    format_ama_journal_reference,
    replace_cite_tags,
)


def test_extract_cite_keys_order_and_duplicates() -> None:
    md = "A{{cite:K1}} B{{cite:K2}} C{{cite:K1}}"
    assert extract_cite_keys(md) == ["K1", "K2", "K1"]


def test_replace_cite_tags() -> None:
    md = "A{{cite:K1}} B{{cite:K2}} C{{cite:K1}}"
    out = replace_cite_tags(md, {"K1": 1, "K2": 2})
    assert out == "A^1^ B^2^ C^1^"


def test_replace_cite_tags_missing_key_raises() -> None:
    with pytest.raises(KeyError):
        replace_cite_tags("{{cite:K1}}", {"K2": 2})


def test_format_ama_reference_et_al_rule_and_issue() -> None:
    article = PubMedArticle(
        pmid="123",
        title="Test title",
        journal_abbrev="J Test",
        year="2024",
        volume="12",
        issue="3",
        pages="45-50",
        doi="10.1000/jtest.1",
        authors=[
            PubMedAuthor(last_name="Alpha", initials="A"),
            PubMedAuthor(last_name="Beta", initials="B"),
            PubMedAuthor(last_name="Gamma", initials="G"),
            PubMedAuthor(last_name="Delta", initials="D"),
            PubMedAuthor(last_name="Epsilon", initials="E"),
            PubMedAuthor(last_name="Zeta", initials="Z"),
            PubMedAuthor(last_name="Eta", initials="E"),
        ],
    )
    ref = format_ama_journal_reference(article)
    assert ref.startswith("Alpha A, Beta B, Gamma G, et al.")
    assert "J Test." in ref
    assert "2024;12(3):45-50." in ref
    assert "doi:10.1000/jtest.1." in ref

