from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import pandas as pd

from paperjn.nlp.text import normalize_whitespace


PARSER_VERSION = "jats_blocks_v1"


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


def _itertext(elem: ET.Element) -> str:
    return normalize_whitespace(" ".join(t for t in elem.itertext() if t and t.strip()))


def _first_text(elem: ET.Element | None) -> str | None:
    if elem is None:
        return None
    text = _itertext(elem)
    return text if text else None


def _find_first(root: ET.Element, path: str) -> ET.Element | None:
    # JATS files often have no explicit namespace, but some do; be robust.
    found = root.find(path)
    if found is not None:
        return found
    # Fallback: search by local-name.
    want = path.split("/")[-1]
    want = want.split("[", 1)[0]
    for e in root.iter():
        if _strip_ns(e.tag) == want:
            return e
    return None


def _sec_title(sec: ET.Element) -> str | None:
    label = _first_text(sec.find("./label"))
    title = _first_text(sec.find("./title"))
    if label and title:
        return normalize_whitespace(f"{label} {title}")
    return title or label


def _block_id(pmcid: str, source: str, sec_path: list[str], text: str) -> str:
    key = "\n".join([pmcid, source, " > ".join(sec_path), text])
    digest = sha1(key.encode("utf-8")).hexdigest()[:16]
    return f"{pmcid}__{source}__{digest}"


@dataclass(frozen=True)
class ParsedJATS:
    metadata: dict[str, Any]
    blocks: pd.DataFrame


def parse_jats_xml(xml_path: Path, *, pmcid: str | None = None) -> ParsedJATS:
    """Parse a PMC JATS/XML file into ordered paragraph blocks with section provenance."""
    xml_path = Path(xml_path)
    pmcid = pmcid or xml_path.stem

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # The <article> node might be nested inside <pmc-articleset>.
    article = root
    if _strip_ns(article.tag) != "article":
        maybe = _find_first(root, ".//article")
        if maybe is not None:
            article = maybe

    article_type = article.attrib.get("article-type")
    title_el = _find_first(article, ".//article-title")
    title = _first_text(title_el)

    blocks: list[dict[str, Any]] = []
    block_index = 0

    skip_tags = {"fig", "table-wrap", "ref-list", "ack", "notes", "supplementary-material"}

    def walk(elem: ET.Element, *, source: str, sec_path: list[str]) -> None:
        nonlocal block_index
        tag = _strip_ns(elem.tag)
        if tag in skip_tags:
            return

        if tag == "sec":
            t = _sec_title(elem)
            next_path = sec_path + ([t] if t else [])
            for child in list(elem):
                walk(child, source=source, sec_path=next_path)
            return

        if tag == "p":
            text = _itertext(elem)
            if text:
                bid = _block_id(pmcid, source, sec_path, text)
                blocks.append(
                    {
                        "pmcid": pmcid,
                        "source": source,
                        "block_index": int(block_index),
                        "block_id": bid,
                        "sec_path": sec_path,
                        "sec_path_str": " > ".join(sec_path),
                        "text": text,
                        "n_chars": int(len(text)),
                        "parser_version": PARSER_VERSION,
                        "xml_path": str(xml_path),
                    }
                )
                block_index += 1
            return

        for child in list(elem):
            walk(child, source=source, sec_path=sec_path)

    # Abstract blocks (optional, used for routing/QC).
    abstract_elems = [e for e in article.iter() if _strip_ns(e.tag) == "abstract"]
    for abs_el in abstract_elems[:1]:
        walk(abs_el, source="abstract", sec_path=["Abstract"])

    # Body blocks (primary)
    body_el = _find_first(article, ".//body")
    has_body = body_el is not None
    if body_el is not None:
        walk(body_el, source="body", sec_path=[])

    df = pd.DataFrame(blocks)
    if not df.empty:
        df = df.sort_values(["source", "block_index"]).reset_index(drop=True)

    metadata = {
        "pmcid": pmcid,
        "article_type": article_type,
        "title": title,
        "xml_path": str(xml_path),
        "parser_version": PARSER_VERSION,
        "has_body": bool(has_body),
        "n_blocks_total": int(len(df)),
        "n_blocks_body": int((df["source"] == "body").sum()) if not df.empty else 0,
        "n_blocks_abstract": int((df["source"] == "abstract").sum()) if not df.empty else 0,
    }
    return ParsedJATS(metadata=metadata, blocks=df)

