"""Build the real optical+SAR feature table over a balanced subsample of
real fields, not all 1,302 at once. Peak-growing-season compositing,
quality-filtered, resumable (skips field_ids already in the output CSV, so
an interrupted run can just be started again).

Must run through `python -m` (Earth Engine credential access, see
DECISIONS.md entry 10):

    python -m scripts.build_feature_table
    python -m scripts.build_feature_table --countries Uruguay --crops soybean \
        --n-per-group 405 --out ../Results/features/uruguay_features.csv
"""

import argparse
import csv
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import rasterio

from data.config import CODES_ROOT, DATASET_ROOT, DEV_CACHE_DIR, GEE_PROJECT_ID, RESULTS_DIR
from data.gee_sentinel1 import aoi_from_raster_bounds, fetch_sentinel1_vvvh
from data.sampling import select_sample_fields
from data.yieldsat_loader import (
    get_yield_mask_path,
    list_s2_image_paths,
    list_scl_mask_paths,
    load_field_metadata,
    select_bgrn_bands,
    S2_REFLECTANCE_SCALE,
)
from preprocessing.optical import build_cloud_free_composite
from preprocessing.sar import backscatter_stats, coregister_to_reference, lee_filter
from preprocessing.seasonal import peak_window_bounds, select_peak_window_indices
from features.optical_indices import compute_ndvi, summarize_field_indices
from features.glcm import compute_field_texture_features

N_PER_GROUP = 50
COUNTRIES = ["Argentina", "Brazil"]
CROPS = ["corn", "soybean", "wheat"]
OUT_PATH = RESULTS_DIR / "features" / "real_field_features.csv"
GLCM_BAND_NAMES = ["nir", "ndvi", "vv", "vh"]
GLCM_FEATURE_NAMES = ["contrast", "homogeneity", "energy", "correlation"]
GLCM_COLUMNS = [f"{band}_glcm_{feat}" for band in GLCM_BAND_NAMES for feat in GLCM_FEATURE_NAMES]
FIELDNAMES = [
    "field_id", "country", "crop", "year", "quality",
    "n_optical_dates_in_window", "n_optical_dates_kept",
    "mean_ndvi", "mean_evi", "mean_gndvi", "mean_ndwi",
    "n_sar_scenes", "mean_vv_db", "std_vv_db", "mean_vh_db", "std_vh_db", "mean_vv_minus_vh_db",
    *GLCM_COLUMNS,
    "status",
]


def already_done(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    with open(out_path, newline="") as f:
        return {row["field_id"] for row in csv.DictReader(f)}


def load_windowed_frames(field_dir, seeding_date, harvesting_date):
    s2_paths = list_s2_image_paths(field_dir)
    scl_paths = list_scl_mask_paths(field_dir)
    date_strs = [p.stem.replace("S2_L2A_", "") for p in s2_paths]

    kept_indices = select_peak_window_indices(date_strs, seeding_date, harvesting_date)
    if not kept_indices:
        raise ValueError("no S2 acquisitions fall inside the peak-season window")

    optical_frames, scl_frames = [], []
    for i in kept_indices:
        with rasterio.open(s2_paths[i]) as src:
            optical_frames.append(src.read().astype("float32") / S2_REFLECTANCE_SCALE)
        with rasterio.open(scl_paths[i]) as src:
            scl_frames.append(src.read(1))
    return optical_frames, scl_frames, len(kept_indices)


def process_field(field_dir: Path) -> dict:
    meta = load_field_metadata(field_dir)
    row = {"field_id": meta.field_id, "country": meta.country, "crop": meta.crop,
           "year": meta.year, "quality": meta.quality}

    optical_frames, scl_frames, n_in_window = load_windowed_frames(field_dir, meta.seeding_date, meta.harvesting_date)
    row["n_optical_dates_in_window"] = n_in_window

    composite, _valid_count, n_kept = build_cloud_free_composite(optical_frames, scl_frames)
    row["n_optical_dates_kept"] = n_kept
    bgrn = select_bgrn_bands(composite)
    row.update(summarize_field_indices(bgrn))
    nir_band = bgrn[3]
    ndvi_map = compute_ndvi(bgrn)

    yield_path = get_yield_mask_path(field_dir)
    aoi = aoi_from_raster_bounds(yield_path)
    window_start, window_end = peak_window_bounds(meta.seeding_date, meta.harvesting_date)

    sar_raw_path = DEV_CACHE_DIR / "feature_table_sar" / meta.field_id / "sar_raw.tif"
    sar_raw_path, n_scenes = fetch_sentinel1_vvvh(aoi, str(window_start), str(window_end), sar_raw_path)
    row["n_sar_scenes"] = n_scenes

    sar_coreg_path = coregister_to_reference(sar_raw_path, yield_path, sar_raw_path.parent / "sar_coregistered.tif")
    with rasterio.open(sar_coreg_path) as src:
        vv, vh = src.read(1), src.read(2)
    vv_filtered, vh_filtered = lee_filter(vv), lee_filter(vh)
    row.update(backscatter_stats(vv_filtered, vh_filtered))

    row.update(compute_field_texture_features({
        "nir": nir_band, "ndvi": ndvi_map, "vv": vv_filtered, "vh": vh_filtered,
    }))

    row["status"] = "ok"
    return row


def main() -> None:
    global OUT_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument("--countries", nargs="+", default=COUNTRIES)
    parser.add_argument("--crops", nargs="+", default=CROPS)
    parser.add_argument("--n-per-group", type=int, default=N_PER_GROUP)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()
    OUT_PATH = args.out

    if not GEE_PROJECT_ID:
        raise SystemExit("Set GEE_PROJECT_ID=fusioncrop before running this script.")
    if not DATASET_ROOT.exists():
        raise SystemExit(f"{DATASET_ROOT} not found.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    done_ids = already_done(OUT_PATH)
    write_header = not OUT_PATH.exists()

    fields = select_sample_fields(DATASET_ROOT, args.countries, args.crops, n_per_group=args.n_per_group)
    todo = [f for f in fields if f.name not in done_ids]
    print(f"Sample size: {len(fields)} fields. Already done: {len(done_ids)}. Remaining: {len(todo)}.")

    with open(OUT_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        for i, field_dir in enumerate(todo, 1):
            field_id = field_dir.name
            try:
                row = process_field(field_dir)
            except Exception as exc:  # noqa: BLE001 - one bad field must not kill the run
                row = {k: "" for k in FIELDNAMES}
                row["field_id"] = field_id
                row["status"] = f"error: {exc}"
                print(f"[{i}/{len(todo)}] {field_id}: FAILED ({exc})")
                traceback.print_exc(limit=1)
            else:
                print(f"[{i}/{len(todo)}] {field_id}: ok "
                      f"(NDVI={row['mean_ndvi']:.3f}, {row['n_sar_scenes']} SAR scenes)")
            writer.writerow(row)
            f.flush()

    print(f"\nDone. Output: {OUT_PATH}")


if __name__ == "__main__":
    main()
