"""Dataset-agnostic field manifest.

We do not yet have the real YieldSAT files (access request pending), and
guessing their exact internal folder/file naming would give us code that
looks finished but silently breaks once the real data arrives. Instead,
every later pipeline stage (preprocessing, features, models) reads a
manifest: one row per field, with the paths it needs and nothing else.

Once YieldSAT access is confirmed, only one adapter function needs to be
written: something that reads their actual folder structure and produces
rows in this same format. Nothing downstream has to change.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd

MANIFEST_COLUMNS = [
    "field_id",
    "country",
    "crop_type",
    "optical_path",
    "sar_path",
]


@dataclass
class FieldRecord:
    field_id: str
    country: str
    crop_type: str
    optical_path: str
    sar_path: str


def save_manifest(records: list[FieldRecord], csv_path: str | Path) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([asdict(r) for r in records], columns=MANIFEST_COLUMNS)
    df.to_csv(csv_path, index=False)


def load_manifest(csv_path: str | Path) -> list[FieldRecord]:
    df = pd.read_csv(csv_path)
    missing = set(MANIFEST_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Manifest at {csv_path} is missing columns: {missing}")
    return [FieldRecord(**row) for row in df[MANIFEST_COLUMNS].to_dict(orient="records")]
