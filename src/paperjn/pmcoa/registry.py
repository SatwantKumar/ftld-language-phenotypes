from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd

from paperjn.pmcoa.models import PMCQuery, PMCQueryFile
from paperjn.pmcoa.ncbi_eutils import EUtilsClient, esearch_all, esummary, make_client


_YEAR_RE = re.compile(r"(19|20)\d{2}")


def _parse_year(pubdate: str | None) -> int | None:
    if not pubdate:
        return None
    m = _YEAR_RE.search(str(pubdate))
    return int(m.group(0)) if m else None


def _get_article_id(rec: dict, idtype: str) -> str | None:
    for item in rec.get("articleids", []) or []:
        if str(item.get("idtype", "")).lower() == idtype.lower():
            val = item.get("value")
            return str(val) if val else None
    return None


def _first_author(rec: dict) -> str | None:
    authors = rec.get("authors", []) or []
    if not authors:
        return None
    name = authors[0].get("name")
    return str(name) if name else None


def _safe_url(pmcid: str | None) -> dict[str, str | None]:
    if not pmcid:
        return {"pmc_url": None, "ncbi_pmc_url": None, "pdf_url": None}
    pmc_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"
    return {
        "pmc_url": pmc_url,
        "ncbi_pmc_url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/",
        "pdf_url": f"{pmc_url}pdf/",
    }


def build_registry(
    *,
    query_file: PMCQueryFile,
    query: PMCQuery,
    out_csv: Path,
    year_counts_csv: Path | None = None,
    max_records: int | None = None,
) -> pd.DataFrame:
    client: EUtilsClient = make_client(
        tool=query_file.ncbi.tool,
        tool_env=query_file.ncbi.tool_env,
        email_env=query_file.ncbi.email_env,
        api_key_env=query_file.ncbi.api_key_env,
        request_delay_s=query_file.ncbi.request_delay_s,
    )

    ids, total_count = esearch_all(
        client,
        db=query_file.ncbi.db,
        term=query.term,
        retmax_per_call=5000,
        max_records=max_records,
    )

    query_term_norm = " ".join(str(query.term).split())

    rows: list[dict] = []
    batch_size = 200
    for i in range(0, len(ids), batch_size):
        batch = ids[i : i + batch_size]
        js = esummary(client, db=query_file.ncbi.db, ids=batch)
        result = js.get("result", {})
        uids = result.get("uids", [])
        for uid in uids:
            rec = result.get(str(uid), {})
            if not isinstance(rec, dict):
                continue
            pmcid = _get_article_id(rec, "pmcid")
            pmid = _get_article_id(rec, "pmid")
            doi = _get_article_id(rec, "doi")
            pubdate = rec.get("pubdate")
            year = _parse_year(str(pubdate) if pubdate is not None else None)
            urls = _safe_url(pmcid)
            rows.append(
                {
                    "query_id": query.id,
                    "query_term": query_term_norm,
                    "uid": str(uid),
                    "pmcid": pmcid,
                    "pmid": pmid,
                    "doi": doi,
                    "year": year,
                    "pubdate": str(pubdate) if pubdate is not None else None,
                    "title": rec.get("title"),
                    "journal_abbrev": rec.get("source"),
                    "journal_full": rec.get("fulljournalname"),
                    "first_author": _first_author(rec),
                    "pmclivedate": rec.get("pmclivedate"),
                    "pmc_url": urls["pmc_url"],
                    "ncbi_pmc_url": urls["ncbi_pmc_url"],
                    "pdf_url": urls["pdf_url"],
                    "ncbi_total_count": total_count,
                }
            )

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    if year_counts_csv is not None:
        year_counts_csv.parent.mkdir(parents=True, exist_ok=True)
        _year_counts(df).to_csv(year_counts_csv, index=False)

    return df


def _year_counts(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["year"] = pd.to_numeric(d["year"], errors="coerce").astype("Int64")
    out = (
        d.dropna(subset=["year"])
        .groupby("year", as_index=False)
        .size()
        .rename(columns={"size": "n_papers"})
        .sort_values("year")
    )
    return out


@dataclass(frozen=True)
class SplitRecommendation:
    cutoff_year: int
    n_discovery: int
    n_confirmation: int
    min_per_split: int
    tie_breaker: str
    note: str


def recommend_time_split(df: pd.DataFrame, *, min_per_split: int, tie_breaker: str) -> SplitRecommendation:
    years = pd.to_numeric(df["year"], errors="coerce")
    years = years.dropna().astype(int)
    if years.empty:
        raise ValueError("No parseable years in registry; cannot recommend a time split.")

    year_counts = years.value_counts().sort_index()
    uniq_years = year_counts.index.to_list()

    best = None

    total = int(years.shape[0])
    cumsum = year_counts.cumsum()

    for cutoff in uniq_years:
        n_disc = int(cumsum.loc[cutoff])
        n_conf = int(total - n_disc)
        feasible = n_disc >= int(min_per_split) and n_conf >= int(min_per_split)

        imbalance = abs(n_disc - n_conf)
        # Primary objective: feasible, then minimize imbalance.
        # Tie-breaker: earlier or later cutoff year.
        score = (0 if feasible else 1, imbalance, cutoff if tie_breaker == "earlier" else -cutoff)

        if best is None or score < best[0]:
            best = (score, cutoff, n_disc, n_conf, feasible)

    assert best is not None
    _, cutoff, n_disc, n_conf, feasible = best
    note = "Balanced split (meets min_per_split)." if feasible else "Best-effort split (min_per_split not met)."
    return SplitRecommendation(
        cutoff_year=int(cutoff),
        n_discovery=int(n_disc),
        n_confirmation=int(n_conf),
        min_per_split=int(min_per_split),
        tie_breaker=str(tie_breaker),
        note=note,
    )


def load_query_file(path: Path) -> PMCQueryFile:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return PMCQueryFile.model_validate(data)
