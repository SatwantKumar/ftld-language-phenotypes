from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


SegmentType = Literal[
    "case_presentation",
    "disease_course",
    "clinical_features_summary",
    "neuropsych_language_testing",
    "imaging_biomarkers",
    "pathology_genetics",
    "treatment_response",
    "other",
]


class NanoRouteOutput(BaseModel):
    is_case_like: bool
    n_cases_est: int | None = Field(default=None, ge=1)
    selected_sec_paths: list[str] = Field(default_factory=list)
    case_markers_found: list[str] = Field(default_factory=list)


class ExtractedSegment(BaseModel):
    case_id: str = Field(description="case_1, case_2, ... or case_unknown")
    segment_type: SegmentType
    block_ids: list[str] = Field(min_length=1)
    include_for_embedding: bool


class GPT52ExtractionOutput(BaseModel):
    n_cases_est: int | None = Field(default=None, ge=1)
    segments: list[ExtractedSegment] = Field(default_factory=list)

