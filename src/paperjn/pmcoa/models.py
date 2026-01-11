from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class NCBIConfig(BaseModel):
    db: str = "pmc"
    tool: str = "paperjn"
    tool_env: str | None = "NCBI_TOOL"
    email_env: str | None = "NCBI_EMAIL"
    api_key_env: str | None = "NCBI_API_KEY"
    request_delay_s: float = 0.35


class PMCQuery(BaseModel):
    id: str
    term: str
    description: str | None = None


class TimeSplitConfig(BaseModel):
    strategy: Literal["time_balance"] = "time_balance"
    # Optional: once you accept a cutoff year, lock it here for reproducibility.
    cutoff_year: int | None = None
    min_per_split: int = 200
    tie_breaker: Literal["earlier", "later"] = "earlier"


class PMCQueryFile(BaseModel):
    version: int = 1
    created_utc: str = Field(default_factory=lambda: datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"))
    backend: Literal["ncbi_eutils"] = "ncbi_eutils"
    ncbi: NCBIConfig = Field(default_factory=NCBIConfig)
    queries: list[PMCQuery]
    split: TimeSplitConfig = Field(default_factory=TimeSplitConfig)
