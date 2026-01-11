from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ColumnsConfig(BaseModel):
    stage: str
    subtype: str
    text: str


class PhenotypesConfig(BaseModel):
    ppa_markers: list[str] = Field(default_factory=lambda: ["svppa", "nfvppa", "lvppa"])


class EmbeddingsConfig(BaseModel):
    model_name: str
    sensitivity_model_names: list[str] = Field(default_factory=list)
    normalize_embeddings: bool = True
    batch_size: int = 64
    device: str | None = None


class DimensionalityReductionConfig(BaseModel):
    method: str = "pca"
    pca_variance_threshold: float = 0.95
    pca_max_components: int = 50


class ClusteringConfig(BaseModel):
    algorithm: str = "kmeans"
    k_strategy: str = "fixed"
    k_fixed: int = 4
    k_sensitivity_values: list[int] = Field(default_factory=lambda: [3, 4, 5, 6])
    k_min: int = 2
    k_max: int = 10


class InferenceConfig(BaseModel):
    permutations: int = 5000
    family_fdr_q: float = 0.05


class PathsConfig(BaseModel):
    curated_input_csv: str
    interim_dir: str
    processed_dir: str
    results_dir: str


class StabilityConfig(BaseModel):
    n_bootstrap: int = 200
    subsample_fraction: float = 0.8
    seed_values: list[int] = Field(default_factory=lambda: list(range(10)))


class LiteratureConfig(BaseModel):
    corpus_csv: str = "data/processed/literature_corpus.csv"
    text_column: str = "text"
    is_ppa_column: str = "is_ppa"
    doc_id_column: str = "doc_id"


class ProjectConfig(BaseModel):
    random_seed: int = 42
    columns: ColumnsConfig
    stages: list[str] = Field(default_factory=lambda: ["Early", "Middle", "Late"])
    phenotypes: PhenotypesConfig = Field(default_factory=PhenotypesConfig)
    leakage_blacklist: list[str] = Field(default_factory=list)
    embeddings: EmbeddingsConfig
    dimensionality_reduction: DimensionalityReductionConfig = Field(
        default_factory=DimensionalityReductionConfig
    )
    clustering: ClusteringConfig
    inference: InferenceConfig
    paths: PathsConfig
    stability: StabilityConfig = Field(default_factory=StabilityConfig)
    literature: LiteratureConfig = Field(default_factory=LiteratureConfig)


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    data: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return ProjectConfig.model_validate(data)
