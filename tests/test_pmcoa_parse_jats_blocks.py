from pathlib import Path

from paperjn.pmcoa.parse_jats_blocks import parse_jats_xml


def test_parse_jats_xml_minimal(tmp_path: Path) -> None:
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<article article-type="case-report">
  <front>
    <article-meta>
      <title-group>
        <article-title>Test Title</article-title>
      </title-group>
      <abstract>
        <p>Abstract paragraph.</p>
      </abstract>
    </article-meta>
  </front>
  <body>
    <sec>
      <title>Case Presentation</title>
      <p>First paragraph.</p>
      <p>Second paragraph.</p>
    </sec>
  </body>
</article>
"""
    p = tmp_path / "PMC123.xml"
    p.write_text(xml, encoding="utf-8")

    parsed = parse_jats_xml(p, pmcid="PMC123")
    assert parsed.metadata["pmcid"] == "PMC123"
    assert parsed.metadata["article_type"] == "case-report"
    assert parsed.metadata["has_body"] is True
    assert parsed.metadata["n_blocks_abstract"] == 1
    assert parsed.metadata["n_blocks_body"] == 2
    assert set(parsed.blocks["source"]) == {"abstract", "body"}
    assert any(parsed.blocks["sec_path_str"].astype(str).str.contains("Case Presentation"))

