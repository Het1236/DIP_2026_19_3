"""Adapter for the raw per-field YieldSAT release.

Layout follows the YieldSAT CVPR paper (arXiv 2604.00940v1) and the
"yieldsat-overview-notebook" on yieldsat.github.io, with one difference:
our download is organized per country rather than under a single shared
original-preprocessed/ folder. Everything else (band order, band count,
file names, metadata keys) matches the public documentation.

Real layout on disk, one directory per field:

    dataset/<Country>/<Country>_raw/<field_id>/
        metadata-<field_id>.json
        s2_images/S2_L2A_<date>.tif       one file per acquisition date, 12 bands, uint16
        scl_masks/S2_L2A_SCL_<date>.tif   same dates as s2_images, 1 band, uint8
        yield_masks/mean_scaled_yield_masked_regional_statistical_outlier.tif
        yield_masks/number_scaled_yield_masked_regional_statistical_outlier.tif
        yield_masks/std_scaled_yield_masked_regional_statistical_outlier.tif
        dem/{aspect,curvature,dem,slope,twi}-<field_id>.tif
        soil/{cec,cfvo,clay,nitrogen,phh2o,sand,silt,soc}_0_200cm-<field_id>.tif
        weather/<field_id>.csv
    dataset/<Country>/<Country>_preprocessed/merge_s2-soil-dem-weather-coords.nc

Our project only needs metadata, s2_images and scl_masks (optical side) plus
the crop label. dem, soil and weather are YieldSAT's auxiliary modalities for
their own yield regression benchmark, not part of our SAR-optical fusion
brief, so this adapter does not read them.

s2_images values are raw uint16 digital numbers, not reflectance. Divide by
10000 (standard Sentinel-2 L2A scale factor) to get reflectance in the 0-1
range our vegetation index formulas expect. Pixel value 0 is nodata (outside
the field boundary or a genuinely missing pixel), mask it out before any
band math, confirmed via each file's own `nodata` tag.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Confirmed 12-band order for a raw s2_images/*.tif file, both from
# YieldSAT's ML tutorial notebook and directly from a real file's band
# descriptions.
S2_BAND_ORDER = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
S2_REFLECTANCE_SCALE = 10000
S2_NODATA_VALUE = 0

YIELD_MASK_FILENAME = "mean_scaled_yield_masked_regional_statistical_outlier.tif"
YIELD_COUNT_FILENAME = "number_scaled_yield_masked_regional_statistical_outlier.tif"
YIELD_STD_FILENAME = "std_scaled_yield_masked_regional_statistical_outlier.tif"

# Indices into S2_BAND_ORDER for the 4 bands our features/optical_indices.py
# and every synthetic fixture use internally (blue, green, red, nir), so a
# real 12-band composite can be reduced to the same 4-band convention used
# everywhere else in the codebase without changing optical_indices.py.
BGRN_BAND_INDICES = [S2_BAND_ORDER.index(b) for b in ["B02", "B03", "B04", "B08"]]


def select_bgrn_bands(composite_12band) -> "object":
    """Reduce a real 12-band S2 composite to our internal 4-band (blue,
    green, red, nir) convention, matching data/synthetic.py's OPTICAL_BANDS
    and features/optical_indices.py's expected band order.
    """
    return composite_12band[BGRN_BAND_INDICES]


@dataclass
class YieldSATFieldMeta:
    field_id: str
    country: str
    crop: str
    farm_identifier: str
    year: int
    centroid_lat: float
    centroid_lon: float
    projected_crs_epsg: int
    area_ha: float
    seeding_date: str
    harvesting_date: str
    quality: str


def load_field_metadata(field_root: Path) -> YieldSATFieldMeta:
    field_id = field_root.name
    metadata_path = field_root / f"metadata-{field_id}.json"
    data = json.loads(metadata_path.read_text())
    return YieldSATFieldMeta(
        field_id=data["field_shared_name"],
        country=data["adm_units"]["country"],
        crop=data["crop"],
        farm_identifier=data["farm_identifier"],
        year=data["year"],
        centroid_lat=data["centroid_latitude_wgs84"],
        centroid_lon=data["centroid_longitude_wgs84"],
        projected_crs_epsg=data["projected_crs_epsg"],
        area_ha=data["area_calculated"],
        seeding_date=data["seeding_date"],
        harvesting_date=data["harvesting_date"],
        quality=data["yieldmap_quality"],
    )


def list_s2_image_paths(field_root: Path) -> list[Path]:
    return sorted((field_root / "s2_images").glob("*.tif"))


def list_scl_mask_paths(field_root: Path) -> list[Path]:
    return sorted((field_root / "scl_masks").glob("*.tif"))


def get_yield_mask_path(field_root: Path) -> Path:
    return field_root / "yield_masks" / YIELD_MASK_FILENAME


def raw_country_dir(dataset_root: Path, country: str) -> Path:
    """dataset/<Country>/<Country>_raw"""
    return Path(dataset_root) / country / f"{country}_raw"


def preprocessed_country_path(dataset_root: Path, country: str) -> Path:
    """dataset/<Country>/<Country>_preprocessed/merge_s2-soil-dem-weather-coords.nc"""
    return Path(dataset_root) / country / f"{country}_preprocessed" / "merge_s2-soil-dem-weather-coords.nc"


def iter_country_field_dirs(dataset_root: Path, country: str):
    country_dir = raw_country_dir(dataset_root, country)
    for field_dir in sorted(country_dir.iterdir()):
        if field_dir.is_dir():
            yield field_dir


def load_real_field_frames(field_root: Path):
    """Read a real field's Sentinel-2 time series as (optical, scl) pairs
    ready for preprocessing.optical.build_cloud_free_composite: optical
    scaled from raw uint16 digital numbers to reflectance (see module
    docstring), scl unchanged.
    """
    import rasterio

    s2_paths = list_s2_image_paths(field_root)
    scl_paths = list_scl_mask_paths(field_root)
    optical_frames, scl_frames = [], []
    for s2_path, scl_path in zip(s2_paths, scl_paths):
        with rasterio.open(s2_path) as src:
            optical_frames.append(src.read().astype("float32") / S2_REFLECTANCE_SCALE)
        with rasterio.open(scl_path) as src:
            scl_frames.append(src.read(1))
    return optical_frames, scl_frames
