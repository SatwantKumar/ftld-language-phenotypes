from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Any


def _require_requests():
    try:
        import requests  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: requests. Install with `pip install -e '.[literature]'`."
        ) from exc
    return requests


def _env_or_none(name: str | None) -> str | None:
    if not name:
        return None
    val = os.environ.get(name)
    return val if val else None


@dataclass(frozen=True)
class EUtilsClient:
    tool: str = "paperjn"
    email: str | None = None
    api_key: str | None = None
    request_delay_s: float = 0.35
    user_agent: str = "paperjn/0.1 (PMC OA tooling)"
    base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def _sleep(self) -> None:
        time.sleep(max(float(self.request_delay_s), 0.0))

    def get_json(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        requests = _require_requests()

        params = dict(params)
        params.setdefault("retmode", "json")
        if self.tool:
            params.setdefault("tool", self.tool)
        if self.email:
            params.setdefault("email", self.email)
        if self.api_key:
            params.setdefault("api_key", self.api_key)

        url = f"{self.base_url}/{endpoint}"

        backoff_s = 1.0
        last_error: str | None = None
        for attempt in range(8):
            self._sleep()
            try:
                resp = requests.get(url, params=params, headers={"User-Agent": self.user_agent}, timeout=60)
            except Exception as exc:
                last_error = f"Request error: {type(exc).__name__}: {exc}"
                time.sleep(backoff_s)
                backoff_s *= 1.7
                continue

            if resp.status_code == 429 or resp.status_code >= 500:
                last_error = f"HTTP {resp.status_code}"
                time.sleep(backoff_s)
                backoff_s *= 1.7
                continue

            resp.raise_for_status()
            return resp.json()

        raise RuntimeError(f"NCBI request repeatedly failed ({last_error or 'unknown'}; endpoint={endpoint})")


def make_client(
    *,
    tool: str,
    tool_env: str | None,
    email_env: str | None,
    api_key_env: str | None,
    request_delay_s: float,
) -> EUtilsClient:
    tool_override = _env_or_none(tool_env)
    tool_effective = tool_override if tool_override is not None else tool
    return EUtilsClient(
        tool=tool_effective,
        email=_env_or_none(email_env),
        api_key=_env_or_none(api_key_env),
        request_delay_s=request_delay_s,
    )


def esearch(client: EUtilsClient, *, db: str, term: str, retmax: int, retstart: int) -> dict[str, Any]:
    return client.get_json(
        "esearch.fcgi",
        {"db": db, "term": term, "retmax": int(retmax), "retstart": int(retstart)},
    )


def esearch_all(
    client: EUtilsClient,
    *,
    db: str,
    term: str,
    retmax_per_call: int = 5000,
    max_records: int | None = None,
) -> tuple[list[str], int]:
    js0 = esearch(client, db=db, term=term, retmax=0, retstart=0)
    count = int(js0["esearchresult"]["count"])
    if max_records is not None:
        count = min(count, int(max_records))

    ids: list[str] = []
    retstart = 0
    while retstart < count:
        batch = min(int(retmax_per_call), count - retstart)
        js = esearch(client, db=db, term=term, retmax=batch, retstart=retstart)
        idlist = js["esearchresult"].get("idlist", [])
        ids.extend([str(x) for x in idlist])
        retstart += batch

        if not idlist:
            break

    return ids, int(js0["esearchresult"]["count"])


def esummary(client: EUtilsClient, *, db: str, ids: list[str]) -> dict[str, Any]:
    if not ids:
        return {"result": {"uids": []}}
    return client.get_json("esummary.fcgi", {"db": db, "id": ",".join(ids)})
