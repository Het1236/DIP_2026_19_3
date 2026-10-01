"""Synthetic multi-date optical dev data, shaped like the real YieldSAT raw
release, not the single-composite fields from synthetic.py.

synthetic.py generates one optical.tif and one sar.tif per field, which is
enough for a basic environment check but has nothing to mask, so it can't
test cloud handling. This module generates the multi-date shape instead:
one folder per field with s2_images/S2_L2A_<date>.tif and
scl_masks/S2_L2A_SCL_<date>.tif, matching yieldsat_loader.py's real layout.
Some dates are synthetically "cloudy" so preprocessing/optical.py has
something real to filter out.

This is still fake, dev-only data. Never quote its numbers as a result.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from .config import RANDOM_SEED
from .synthetic import CROP_OPTICAL_MEANS, OPTICAL_BANDS, _rng_for_field

# Sentinel-2 Scene Classification Layer codes we care about. Real SCL has
# more classes (see the YieldSAT paper, Table 3), this is the subset needed
# to tell "usable" pixels from "cloud-contaminated" ones.
SCL_VEGETATION = 4
SCL_CLOUD_SHADOW = 3
SCL_CLOUD_MEDIUM_PROB = 8
SCL_CLOUD_HIGH_PROB = 9
CLOUD_SCL_CLASSES = {SCL_CLOUD_SHADOW, SCL_CLOUD_MEDIUM_PROB, SCL_CLOUD_HIGH_PROB}


def generate_field_timeseries(
    field_id: str,
    crop_type: str,
    n_dates: int = 8,
    size: int = 48,
    cloud_probability: float = 0.35,
) -> list[dict]:
    """Return a list of {date, optical, scl} for one field, oldest first."""
    rng = _rng_for_field(field_id)
    optical_means = np.array(CROP_OPTICAL_MEANS[crop_type], dtype=np.float32)

    dates = [f"2021{(1 + i // 4):02d}{1 + (i % 4) * 7:02d}" for i in range(n_dates)]
    # decide which dates are cloudy in one batch, up front, so the cloud
    # pattern for a given field_id does not depend on `size`: every later
    # rng call here draws a size-dependent number of values, which would
    # otherwise shift the rng stream and silently change which dates come
    # out cloudy if size changes between calls with the same field_id.
    cloudy_flags = rng.random(n_dates) < cloud_probability

    frames = []
    for date, is_cloudy in zip(dates, cloudy_flags):
        optical = optical_means[:, None, None] + rng.normal(0, 0.015, size=(4, size, size))
        scl = np.full((size, size), SCL_VEGETATION, dtype=np.uint8)

        if is_cloudy:
            cloud_fraction = rng.uniform(0.4, 1.0)
            cloud_mask = rng.random((size, size)) < cloud_fraction
            # clouds look bright and flat across bands, nothing like a crop signature
            optical[:, cloud_mask] = rng.uniform(0.35, 0.55)
            scl[cloud_mask] = SCL_CLOUD_HIGH_PROB

        optical = np.clip(optical, 0, 1).astype(np.float32)
        frames.append({"date": date, "optical": optical, "scl": scl})

    return frames


def _write_geotiff(path: Path, array: np.ndarray, field_index: int, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixel_size = 0.00009
    transform = from_origin(-60.0 + field_index * 0.05, -30.0, pixel_size, pixel_size)
    count = array.shape[0] if array.ndim == 3 else 1
    data = array if array.ndim == 3 else array[None, :, :]
    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size,
        count=count, dtype=data.dtype, crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(data)


def write_field_timeseries(out_dir: Path, field_id: str, frames: list[dict], field_index: int, size: int) -> Path:
    field_root = Path(out_dir) / field_id
    for frame in frames:
        _write_geotiff(field_root / "s2_images" / f"S2_L2A_{frame['date']}.tif", frame["optical"], field_index, size)
        _write_geotiff(field_root / "scl_masks" / f"S2_L2A_SCL_{frame['date']}.tif", frame["scl"], field_index, size)
    return field_root


def generate_synthetic_timeseries_dataset(
    out_dir: str | Path,
    crop_types: list[str],
    n_fields_per_crop: int = 2,
    n_dates: int = 8,
    size: int = 48,
) -> list[dict]:
    out_dir = Path(out_dir)
    records = []
    field_index = 0
    for crop_type in crop_types:
        for i in range(n_fields_per_crop):
            field_id = f"synthetic_{crop_type}_{i:03d}"
            frames = generate_field_timeseries(field_id, crop_type, n_dates=n_dates, size=size)
            field_root = write_field_timeseries(out_dir, field_id, frames, field_index, size)
            records.append({"field_id": field_id, "crop_type": crop_type, "field_root": field_root})
            field_index += 1
    return records
