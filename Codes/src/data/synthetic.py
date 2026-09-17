"""Synthetic dev data generator.

We do not have real YieldSAT / Sentinel-1 access confirmed yet, so this
generates small, fake-but-structured fields (optical + SAR GeoTIFFs) that
look enough like the real thing to develop and test the rest of the
pipeline against: same file format (GeoTIFF via rasterio), same manifest
row shape, same band layout.

This is a development aid only. Nothing produced here should ever be
quoted as a project result. Every array is randomly generated from a
fixed seed per field so it is reproducible across the team.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from .config import CROP_TYPES, RANDOM_SEED
from .manifest import FieldRecord, save_manifest

# band order for the synthetic optical raster
OPTICAL_BANDS = ["blue", "green", "red", "nir"]
# band order for the synthetic SAR raster
SAR_BANDS = ["VV", "VH"]

# rough, made-up per-crop reflectance/backscatter signatures so crops are
# distinguishable but not trivially separable (real crops overlap too)
CROP_OPTICAL_MEANS = {
    "corn": [0.06, 0.09, 0.07, 0.40],
    "soybean": [0.05, 0.08, 0.06, 0.32],
    "wheat": [0.07, 0.10, 0.10, 0.24],
}
CROP_SAR_MEANS_DB = {
    "corn": [-9.0, -15.0],
    "soybean": [-11.0, -17.5],
    "wheat": [-8.0, -13.5],
}


def _rng_for_field(field_id: str) -> np.random.Generator:
    # deterministic per-field seed derived from the global seed + field id,
    # so re-running the generator reproduces the exact same field
    seed = (RANDOM_SEED + abs(hash(field_id))) % (2**32)
    return np.random.default_rng(seed)


def generate_synthetic_field(field_id: str, crop_type: str, size: int = 64) -> dict:
    if crop_type not in CROP_TYPES:
        raise ValueError(f"Unknown crop_type {crop_type!r}, expected one of {CROP_TYPES}")

    rng = _rng_for_field(field_id)

    optical_means = np.array(CROP_OPTICAL_MEANS[crop_type], dtype=np.float32)
    optical = optical_means[:, None, None] + rng.normal(0, 0.015, size=(4, size, size))
    optical = np.clip(optical, 0, 1).astype(np.float32)

    sar_means_db = np.array(CROP_SAR_MEANS_DB[crop_type], dtype=np.float32)
    # multiplicative speckle noise (gamma distributed, mean 1) is the
    # standard way to fake SAR speckle rather than plain Gaussian noise
    looks = 4.0
    speckle = rng.gamma(shape=looks, scale=1.0 / looks, size=(2, size, size))
    sar_linear = (10 ** (sar_means_db[:, None, None] / 10.0)) * speckle
    sar_db = (10 * np.log10(sar_linear)).astype(np.float32)

    return {"optical": optical, "sar": sar_db}


def _write_geotiff(path: Path, array: np.ndarray, band_names: list[str], field_index: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixel_size = 0.00009  # ~10m in degrees, close enough for synthetic dev data
    # offset each field so they don't spatially overlap, purely cosmetic here
    origin_lon = -60.0 + field_index * 0.05
    origin_lat = -30.0
    transform = from_origin(origin_lon, origin_lat, pixel_size, pixel_size)

    with rasterio.open(
        path, "w", driver="GTiff", height=array.shape[1], width=array.shape[2],
        count=array.shape[0], dtype=array.dtype, crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(array)
        dst.descriptions = tuple(band_names)


def generate_synthetic_dataset(
    out_dir: str | Path,
    countries: list[str],
    crop_types: list[str] | None = None,
    n_fields_per_country_per_crop: int = 5,
    size: int = 64,
) -> list[FieldRecord]:
    crop_types = crop_types or CROP_TYPES
    out_dir = Path(out_dir)

    records: list[FieldRecord] = []
    field_index = 0
    for country in countries:
        for crop_type in crop_types:
            for i in range(n_fields_per_country_per_crop):
                field_id = f"{country}_{crop_type}_{i:03d}"
                data = generate_synthetic_field(field_id, crop_type, size=size)

                optical_path = out_dir / country / field_id / "optical.tif"
                sar_path = out_dir / country / field_id / "sar.tif"
                _write_geotiff(optical_path, data["optical"], OPTICAL_BANDS, field_index)
                _write_geotiff(sar_path, data["sar"], SAR_BANDS, field_index)

                records.append(FieldRecord(
                    field_id=field_id,
                    country=country,
                    crop_type=crop_type,
                    optical_path=str(optical_path),
                    sar_path=str(sar_path),
                ))
                field_index += 1

    save_manifest(records, out_dir / "manifest.csv")
    return records
