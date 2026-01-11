from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CuratedTableSpec:
    stage_col: str
    subtype_col: str
    text_col: str
    stages: list[str]


def load_curated_table(path: str, spec: CuratedTableSpec) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in [spec.stage_col, spec.subtype_col, spec.text_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Curated table is missing required columns: {missing}")

    df = df.copy()
    df = df[df[spec.stage_col].isin(spec.stages)]
    df = df[df[spec.text_col].notna()]
    df = df[df[spec.subtype_col].notna()]
    df[spec.text_col] = df[spec.text_col].astype(str)
    df[spec.subtype_col] = df[spec.subtype_col].astype(str)
    df[spec.stage_col] = df[spec.stage_col].astype(str)
    df = df[df[spec.text_col].str.strip().astype(bool)]
    df = df.reset_index(drop=True)
    return df

