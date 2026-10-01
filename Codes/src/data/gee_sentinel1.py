"""Pull a small Sentinel-1 VV/VH composite for one field's AOI via GEE.

Uses ee.Image.getDownloadURL for a direct, synchronous GeoTIFF download
instead of an asynchronous export task to Drive or Cloud Storage. This only
works for AOIs small enough to fit in one HTTP response, which is exactly
what one field is, a few hundred pixels at 10m, so the whole pull is one
function call with no task polling, no Drive folder to check later.

Run any function here the same way as gee_auth_check.py and
gee_smoke_test.py (`python -m src.data.<module>` from Codes/), not with a
raw `python -c` one-liner, since that can fail to pick up the Earth Engine
credentials correctly on some machines.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import requests

from .config import DEV_CACHE_DIR, GEE_PROJECT_ID


def _ee():
    import ee
    ee.Initialize(project=GEE_PROJECT_ID)
    return ee


def square_bbox_from_centroid(lat: float, lon: float, area_ha: float) -> tuple[float, float, float, float]:
    """Plain-math fallback AOI: a square of the given area, centered on a
    field's centroid. Only used when we don't have a real field raster to
    read exact bounds from. Returns (min_lon, min_lat, max_lon, max_lat).
    """
    side_m = math.sqrt(area_ha * 10000)
    half_lat_deg = (side_m / 2) / 111320
    half_lon_deg = (side_m / 2) / (111320 * math.cos(math.radians(lat)) + 1e-9)
    return (lon - half_lon_deg, lat - half_lat_deg, lon + half_lon_deg, lat + half_lat_deg)


def aoi_from_centroid(lat: float, lon: float, area_ha: float):
    ee = _ee()
    return ee.Geometry.Rectangle(list(square_bbox_from_centroid(lat, lon, area_ha)))


def aoi_from_raster_bounds(raster_path):
    """Preferred AOI source once we have a real field raster (optical or
    yield mask): its exact bounds, reprojected to WGS84. Guarantees the SAR
    pull lines up with YieldSAT's own grid, a centroid guess would not.
    """
    import rasterio
    from rasterio.warp import transform_bounds

    with rasterio.open(raster_path) as src:
        bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    ee = _ee()
    return ee.Geometry.Rectangle(list(bounds))


def fetch_sentinel1_vvvh(aoi_geom, start_date: str, end_date: str, out_path, scale: int = 10):
    """Median VV/VH composite over the AOI and date range, downloaded as a
    2-band GeoTIFF (VV, VH, both in dB). Raises if no scenes are found.
    """
    ee = _ee()
    collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi_geom)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    )
    n_scenes = collection.size().getInfo()
    if n_scenes == 0:
        raise ValueError(f"no Sentinel-1 IW VV+VH scenes found for {start_date} to {end_date} over this AOI")

    composite = collection.select(["VV", "VH"]).median().clip(aoi_geom)
    url = composite.getDownloadURL({"scale": scale, "region": aoi_geom, "format": "GEO_TIFF"})

    response = requests.get(url, timeout=120)
    response.raise_for_status()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(response.content)
    return out_path, n_scenes


if __name__ == "__main__":
    # Real-data smoke test for the whole SAR pull, run with:
    #   python -m src.data.gee_sentinel1
    # Pulls a small real AOI (same sample area as gee_smoke_test.py), not a
    # YieldSAT field, we don't have real field boundaries yet.
    out_path = DEV_CACHE_DIR / "real_sar_test" / "sample_aoi.tif"
    aoi = _ee().Geometry.Rectangle([-62.42, -34.42, -62.41, -34.41])
    try:
        path, n = fetch_sentinel1_vvvh(aoi, "2024-01-01", "2024-02-01", out_path)
        print(f"Downloaded real Sentinel-1 VV/VH composite from {n} scenes to {path}")
    except Exception as exc:  # noqa: BLE001
        print(f"Real SAR pull FAILED: {exc}")
        sys.exit(1)
