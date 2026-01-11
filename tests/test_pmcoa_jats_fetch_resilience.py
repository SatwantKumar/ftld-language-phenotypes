from __future__ import annotations

from pathlib import Path

import pandas as pd

from paperjn.pmcoa.jats import fetch_jats_xml, fetch_jats_from_registry
from paperjn.pmcoa.ncbi_eutils import EUtilsClient


def test_fetch_jats_xml_handles_request_exception(monkeypatch, tmp_path: Path) -> None:
    import paperjn.pmcoa.jats as jats_mod

    class StubRequests:
        def get(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise RuntimeError("boom")

    monkeypatch.setattr(jats_mod, "_require_requests", lambda: StubRequests())
    monkeypatch.setattr(jats_mod.time, "sleep", lambda *_: None)

    client = EUtilsClient(tool="paperjn", request_delay_s=0.0)
    out_xml = tmp_path / "PMC123.xml"

    res = fetch_jats_xml(client, pmc_id="PMC123", out_xml=out_xml, overwrite=True, timeout_s=1)
    assert res.status == "error"
    assert res.out_xml is None
    assert out_xml.exists() is False
    assert list(tmp_path.glob("*.part")) == []


def test_fetch_jats_from_registry_catches_unhandled_exc(monkeypatch, tmp_path: Path) -> None:
    import paperjn.pmcoa.jats as jats_mod

    def boom(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("boom")

    monkeypatch.setattr(jats_mod, "fetch_jats_xml", boom)

    reg = tmp_path / "registry.csv"
    pd.DataFrame(
        [
            {
                "pmcid": "PMC123",
                "pmid": "123",
                "doi": None,
                "year": 2020,
                "title": "Test",
            }
        ]
    ).to_csv(reg, index=False)

    out_dir = tmp_path / "jats"
    client = EUtilsClient(tool="paperjn", request_delay_s=0.0)
    df_log, _ = fetch_jats_from_registry(
        client=client,
        registry_csv=reg,
        out_dir=out_dir,
        max_papers=1,
        random_seed=42,
        overwrite=False,
        batch_size=1,
        batch_sleep_s=0.0,
        batch_jitter_s=0.0,
        mode="sequential",
    )

    assert df_log.shape[0] == 1
    assert df_log.iloc[0]["status"] == "error"
    assert "Unhandled exception" in str(df_log.iloc[0]["error"])

