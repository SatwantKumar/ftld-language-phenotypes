from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

PhaseCSchemaVersion = Literal["pmcoa_case_labels_v1"]


FTLDInclusionTier = Literal["exclude", "ftld_strict", "ftld_broad"]

FTLDInclusionBasis = Literal[
    "author_reports_ftld_or_ppa_or_psp_or_cbs_or_ftd_mnd",
    "meets_consensus_criteria_explicitly_stated",
    "genetic_confirmed_pathogenic",
    "neuropath_confirmed",
    "specialist_dx_reported",
    "unclear_basis",  # allowed only for ftld_broad
]

NonFTLDPrimaryCategory = Literal[
    "vascular_stroke_or_poststroke_aphasia",
    "primary_psychiatric_or_functional",
    "brain_tumor_or_mass",
    "infection_inflammatory_or_autoimmune",
    "tbi_or_structural",
    "toxic_metabolic_or_medication",
    "epilepsy_seizure_related",
    "other_neurodegenerative_non_ftld",
    "other_neurologic",
    "unclear",
]

FTLDSyndrome = Literal[
    "bvFTD",
    "svPPA",
    "nfvPPA",
    "lvPPA",
    "PPA_unspecified",
    "PSP",
    "CBS",
    "FTD_MND",
    "FTLD_unspecified",
    "not_reported",  # allowed only for ftld_broad
]

LabelConfidence = Literal["high", "medium", "low"]


# 4-level status used for symptom tags and other presence/absence-style fields.
TagStatus = Literal["present", "explicitly_absent", "not_reported", "uncertain"]

SymptomTagName = Literal[
    "apraxia_of_speech",
    "agrammatism",
    "semantic_loss",
    "behavioral_change_disinhibition_or_apathy",
    "compulsions_or_rigid_routines",
    "parkinsonism",
    "oculomotor_vertical_gaze_palsy",
    "limb_apraxia_or_alien_limb",
    "mnd_signs",
    "psychosis_hallucinations",
]


ImagingModality = Literal["mri", "ct", "fdg_pet", "spect", "other"]
ImagingLaterality = Literal["left", "right", "bilateral", "diffuse", "unknown"]
ImagingRegion = Literal[
    "frontal",
    "temporal",
    "parietal",
    "insula",
    "cingulate",
    "basal_ganglia",
    "brainstem",
    "cerebellum",
    "other",
    "unknown",
]

GeneticsStatus = Literal["confirmed_pathogenic", "reported_uncertain", "tested_negative", "not_reported", "unclear"]
NeuropathStatus = Literal["confirmed", "not_reported", "unclear"]
PathologyType = Literal["tau", "tdp43", "fus", "mixed", "other", "unknown"]

InitialDxCategory = Literal[
    "ftld",
    "psychiatric",
    "ad",
    "vascular_stroke",
    "other_neuro",
    "other",
    "not_reported",
    "unknown",
]

MisdiagnosedPriorToFTLD = Literal["yes", "no", "not_reported", "unknown"]


class SymptomTagV1(BaseModel):
    tag: SymptomTagName
    status: TagStatus
    evidence_block_ids: list[str] = Field(default_factory=list, description="1–3 block IDs supporting the tag call.")


class CaseLabelV1(BaseModel):
    label_schema_version: PhaseCSchemaVersion = "pmcoa_case_labels_v1"

    pmcid: str
    case_id: str

    ftld_inclusion_tier: FTLDInclusionTier
    ftld_inclusion_basis: list[FTLDInclusionBasis] = Field(default_factory=list)
    non_ftld_primary_category: NonFTLDPrimaryCategory | None = None
    non_ftld_specific_dx_free_text: str | None = None

    ftld_syndrome_reported: FTLDSyndrome | None = None
    ftld_syndrome_inferred: FTLDSyndrome | None = None

    # Optional families (v1): minimal/coarse
    symptom_duration_months: float | None = Field(default=None, ge=0)
    age_at_onset_years: float | None = Field(default=None, ge=0)
    age_at_presentation_years: float | None = Field(default=None, ge=0)

    imaging_modalities: list[ImagingModality] = Field(default_factory=list)
    imaging_laterality: ImagingLaterality | None = None
    imaging_regions: list[ImagingRegion] = Field(default_factory=list)

    genetics_status: GeneticsStatus | None = None
    genes_reported: list[str] = Field(default_factory=list)
    neuropath_status: NeuropathStatus | None = None
    pathology_types: list[PathologyType] = Field(default_factory=list)

    misdiagnosed_prior_to_ftld: MisdiagnosedPriorToFTLD | None = None
    initial_dx_category: InitialDxCategory | None = None

    symptom_tags: list[SymptomTagV1] = Field(default_factory=list)

    label_confidence: LabelConfidence
    needs_fulltext_review: bool = False
    evidence_block_ids: list[str] = Field(min_length=1)
    notes: str | None = None


def tag_status_to_binary_present(status: TagStatus) -> int:
    """Collapse 4-level tag status into binary for primary kappa: present vs not-present/unknown."""
    return 1 if status == "present" else 0

