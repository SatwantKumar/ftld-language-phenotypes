from paperjn.nlp.leakage import audit_text_for_leakage, remove_blacklisted_terms


def test_leakage_blacklist_removal_and_audit() -> None:
    blacklist = ["svPPA", "bvFTD", "progressive supranuclear palsy"]
    text = "Features resemble svPPA with overlap vs bvFTD; not PSP (progressive supranuclear palsy)."

    cleaned = remove_blacklisted_terms(text, blacklist)
    assert "svPPA" not in cleaned
    assert "bvFTD" not in cleaned
    assert "progressive supranuclear palsy" not in cleaned.lower()
    audit = audit_text_for_leakage(cleaned, blacklist)

    assert audit.n_matches == 0
