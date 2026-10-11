"""Pick a balanced, reproducible subsample of real fields for building the
first real feature table, instead of processing all 1,302 fields right
away. Each field needs its own Sentinel-1 pull, so scoping this down first
keeps the first real run fast enough to debug.
"""

from __future__ import annotations

import random
from pathlib import Path

from .config import RANDOM_SEED
from .yieldsat_loader import iter_country_field_dirs, load_field_metadata

DEFAULT_QUALITY_ALLOWED = {"Good", "Average"}


def select_sample_fields(
    dataset_root: Path,
    countries: list[str],
    crops: list[str],
    n_per_group: int = 50,
    quality_allowed: set[str] = None,
    seed: int = RANDOM_SEED,
) -> list[Path]:
    """Up to n_per_group fields for each (country, crop) combination,
    filtered to quality_allowed, chosen with a fixed seed so the sample is
    the same every time this is run.
    """
    quality_allowed = quality_allowed or DEFAULT_QUALITY_ALLOWED
    rng = random.Random(seed)

    selected: list[Path] = []
    for country in countries:
        groups: dict[str, list[Path]] = {crop: [] for crop in crops}
        for field_dir in iter_country_field_dirs(dataset_root, country):
            meta = load_field_metadata(field_dir)
            if meta.crop in groups and meta.quality in quality_allowed:
                groups[meta.crop].append(field_dir)

        for crop, candidates in groups.items():
            candidates = sorted(candidates)  # deterministic order before shuffling
            rng.shuffle(candidates)
            selected.extend(candidates[:n_per_group])

    return selected
