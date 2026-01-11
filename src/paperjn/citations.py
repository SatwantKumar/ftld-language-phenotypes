from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from typing import Any, Iterable

_DEFAULT_TOOL = "paperjn"
_DEFAULT_EMAIL = "unknown@example.com"


@dataclass(frozen=True)
class PubMedAuthor:
    last_name: str | None = None
    initials: str | None = None
    collective_name: str | None = None


@dataclass(frozen=True)
class PubMedArticle:
    pmid: str
    title: str
    journal_abbrev: str | None
    year: str | None
    volume: str | None
    issue: str | None
    pages: str | None
    doi: str | None
    authors: list[PubMedAuthor]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class PubMedError(RuntimeError):
    pass


def _text_or_none(el: ET.Element | None) -> str | None:
    if el is None:
        return None
    text = "".join(el.itertext()).strip()
    return text or None


def _collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _expand_page_range(pages: str) -> str:
    """Expand abbreviated MedlinePgn ranges (eg, 1672-82 -> 1672-1682).

    PubMed's MedlinePgn often abbreviates the end page by omitting repeated leading
    digits. For JAMA-style references, we expand to full end-page numbers where the
    pattern is unambiguous. Non-numeric formats (eg, article IDs) are preserved.
    """

    def expand_token(token: str) -> str:
        token = token.strip()
        if "-" not in token:
            return token
        start_raw, end_raw = token.split("-", 1)
        start_raw = start_raw.strip()
        end_raw = end_raw.strip()

        m_start = re.fullmatch(r"([A-Za-z]*)(\d+)", start_raw)
        m_end = re.fullmatch(r"([A-Za-z]*)(\d+)", end_raw)
        if not m_start or not m_end:
            return token

        start_prefix, start_num = m_start.group(1), m_start.group(2)
        end_prefix, end_num = m_end.group(1), m_end.group(2)
        if not end_prefix:
            end_prefix = start_prefix

        if len(end_num) < len(start_num):
            end_num = start_num[: len(start_num) - len(end_num)] + end_num

        return f"{start_prefix}{start_num}-{end_prefix}{end_num}"

    parts = [p.strip() for p in pages.split(",") if p.strip()]
    if not parts:
        return pages
    return ", ".join([expand_token(p) for p in parts])


def _user_agent(tool: str | None, email: str | None) -> str:
    tool_eff = (tool or _DEFAULT_TOOL).strip() or _DEFAULT_TOOL
    email_eff = (email or _DEFAULT_EMAIL).strip() or _DEFAULT_EMAIL
    return f"{tool_eff}/0.1 (contact: {email_eff})"


def _http_get(url: str, *, timeout_s: float = 30.0, tool: str | None, email: str | None) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": _user_agent(tool, email),
            "Accept": "*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return resp.read()


def _eutils_url(
    endpoint: str,
    params: dict[str, str],
    *,
    tool: str | None = None,
    email: str | None = None,
    api_key: str | None = None,
) -> str:
    base = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/{endpoint}"
    p = dict(params)
    p.setdefault("tool", tool or _DEFAULT_TOOL)
    p.setdefault("email", email or _DEFAULT_EMAIL)
    if api_key:
        p["api_key"] = api_key
    return base + "?" + urllib.parse.urlencode(p)


def pubmed_search_pmids(
    query: str,
    *,
    retmax: int = 20,
    tool: str | None = None,
    email: str | None = None,
    api_key: str | None = None,
    throttle_s: float = 0.34,
) -> list[str]:
    """Return PubMed IDs (PMIDs) for a query using ESearch."""
    if not query.strip():
        raise ValueError("Query cannot be empty")

    url = _eutils_url(
        "esearch.fcgi",
        params={
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": str(retmax),
        },
        tool=tool,
        email=email,
        api_key=api_key,
    )
    raw = _http_get(url, tool=tool, email=email)
    time.sleep(throttle_s)
    payload = json.loads(raw.decode("utf-8"))
    ids = payload.get("esearchresult", {}).get("idlist", [])
    if not isinstance(ids, list):
        raise PubMedError("Unexpected ESearch response shape")
    return [str(x) for x in ids]


def resolve_doi_to_pmid(
    doi: str,
    *,
    tool: str | None = None,
    email: str | None = None,
    api_key: str | None = None,
) -> str:
    doi = doi.strip()
    if not doi:
        raise ValueError("DOI cannot be empty")
    pmids = pubmed_search_pmids(f"{doi}[DOI]", retmax=5, tool=tool, email=email, api_key=api_key)
    if not pmids:
        raise PubMedError(f"No PubMed matches for DOI: {doi}")
    if len(pmids) > 1:
        raise PubMedError(f"Ambiguous DOI→PMID mapping for DOI {doi}: {pmids}")
    return pmids[0]


def pubmed_fetch_articles(
    pmids: Iterable[str],
    *,
    tool: str | None = None,
    email: str | None = None,
    api_key: str | None = None,
    throttle_s: float = 0.34,
) -> list[PubMedArticle]:
    """Fetch PubMed article metadata using EFetch (XML) and parse into PubMedArticle objects."""
    ids = [str(x).strip() for x in pmids if str(x).strip()]
    if not ids:
        raise ValueError("No PMIDs provided")

    url = _eutils_url(
        "efetch.fcgi",
        params={
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "xml",
        },
        tool=tool,
        email=email,
        api_key=api_key,
    )
    raw = _http_get(url, tool=tool, email=email)
    time.sleep(throttle_s)

    root = ET.fromstring(raw)
    articles: list[PubMedArticle] = []
    for pubmed_article in root.findall(".//PubmedArticle"):
        pmid = _text_or_none(pubmed_article.find(".//MedlineCitation/PMID"))
        if not pmid:
            continue

        title_raw = _text_or_none(pubmed_article.find(".//Article/ArticleTitle")) or ""
        title = _collapse_ws(title_raw).rstrip(".")

        journal_abbrev = _text_or_none(pubmed_article.find(".//Article/Journal/ISOAbbreviation"))
        year = (
            _text_or_none(pubmed_article.find(".//Article/Journal/JournalIssue/PubDate/Year"))
            or _text_or_none(pubmed_article.find(".//Article/Journal/JournalIssue/PubDate/MedlineDate"))
        )
        if year:
            m = re.search(r"(19|20)\\d{2}", year)
            year = m.group(0) if m else year

        volume = _text_or_none(pubmed_article.find(".//Article/Journal/JournalIssue/Volume"))
        issue = _text_or_none(pubmed_article.find(".//Article/Journal/JournalIssue/Issue"))
        pages = _text_or_none(pubmed_article.find(".//Article/Pagination/MedlinePgn"))

        doi = None
        for el in pubmed_article.findall(".//ArticleIdList/ArticleId"):
            if el.get("IdType") == "doi":
                doi = (el.text or "").strip() or None
                break

        authors: list[PubMedAuthor] = []
        for author_el in pubmed_article.findall(".//Article/AuthorList/Author"):
            collective = _text_or_none(author_el.find("CollectiveName"))
            if collective:
                authors.append(PubMedAuthor(collective_name=_collapse_ws(collective)))
                continue
            last = _text_or_none(author_el.find("LastName"))
            initials = _text_or_none(author_el.find("Initials"))
            if last or initials:
                authors.append(PubMedAuthor(last_name=last, initials=initials))

        articles.append(
            PubMedArticle(
                pmid=pmid,
                title=title,
                journal_abbrev=journal_abbrev,
                year=year,
                volume=volume,
                issue=issue,
                pages=pages,
                doi=doi,
                authors=authors,
            )
        )

    if not articles:
        raise PubMedError("No PubMedArticle records parsed from EFetch response")
    return articles


def format_ama_journal_reference(article: PubMedArticle) -> str:
    """Format a PubMedArticle as an AMA-style journal reference."""

    def fmt_author(a: PubMedAuthor) -> str:
        if a.collective_name:
            return a.collective_name
        last = a.last_name or ""
        initials = a.initials or ""
        return (last + " " + initials).strip()

    authors = [fmt_author(a) for a in article.authors if fmt_author(a)]
    if not authors:
        author_str = ""
    elif len(authors) <= 6:
        author_str = ", ".join(authors) + "."
    else:
        author_str = ", ".join(authors[:3]) + ", et al."

    title = article.title.rstrip(".") + "."
    journal = (article.journal_abbrev or "").rstrip(".") + "."

    year = article.year or ""
    vol = article.volume or ""
    issue = article.issue or ""
    pages = _expand_page_range(article.pages or "")

    vol_issue = vol
    if vol and issue:
        vol_issue = f"{vol}({issue})"
    elif vol and not issue:
        vol_issue = vol
    elif issue and not vol:
        vol_issue = f"({issue})"

    parts: list[str] = []
    if author_str:
        parts.append(author_str)
    parts.append(title)
    if journal.strip("."):
        parts.append(journal)

    yvp = ""
    if year:
        yvp = year
    if vol_issue:
        yvp = (yvp + ";" + vol_issue) if yvp else vol_issue
    if pages:
        yvp = (yvp + ":" + pages) if yvp else pages
    if yvp:
        if not yvp.endswith("."):
            yvp += "."
        parts.append(yvp)

    if article.doi:
        parts.append(f"doi:{article.doi}.")

    return " ".join([p for p in parts if p])


_CITE_TAG_RE = re.compile(r"\{\{cite:([A-Za-z0-9_.-]+)\}\}")


def extract_cite_keys(markdown: str) -> list[str]:
    """Return citation keys in order of appearance for tags like: {{cite:KeyName}}."""
    return [m.group(1) for m in _CITE_TAG_RE.finditer(markdown)]


def replace_cite_tags(markdown: str, key_to_number: dict[str, int]) -> str:
    """Replace {{cite:Key}} tags with ^N^ markers (Markdown-friendly superscript)."""

    def repl(m: re.Match[str]) -> str:
        key = m.group(1)
        if key not in key_to_number:
            raise KeyError(f"Missing citation key mapping for: {key}")
        return f"^{key_to_number[key]}^"

    return _CITE_TAG_RE.sub(repl, markdown)
