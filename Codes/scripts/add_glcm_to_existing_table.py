"""Add GLCM texture columns to the feature table built before Week 5, using
only local computation, no new Earth Engine pulls.

The 270 already-completed fields already have their SAR composite cached
locally at .dev_cache/feature_table_sar/<field_id>/sar_coregistered.tif
from the original run, and their optical composite is cheap to recompute
from the local dataset/ files (no network involved either way). Re-pulling
SAR for 270 fields again just to add texture columns would waste roughly
the same hour-plus the original run took, for data we already have.

Run with (no GEE_PROJECT_ID needed, nothing here touches Earth Engine):
    python -m scripts.add_glcm_to_existing_table
"""

import csv
import shutil
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import rasterio

from data.config import DATASET_ROOT, DEV_CACHE_DIR, RESULTS_DIR
from data.yieldsat_loader import (
    get_yield_mask_path,
    list_s2_image_paths,
    list_scl_mask_paths,
    load_field_metadata,
    raw_country_dir,
    select_bgrn_bands,
    S2_REFLECTANCE_SCALE,
)
from preprocessing.optical import build_cloud_free_composite
from preprocessing.seasonal import select_peak_window_indices
from features.optical_indices import compute_ndvi
from features.glcm import compute_field_texture_features

IN_PATH = RESULTS_DIR / "features" / "real_field_features.csv"
SAR_CACHE_DIR = DEV_CACHE_DIR / "feature_table_sar"
GLCM_BAND_NAMES = ["nir", "ndvi", "vv", "vh"]
GLCM_FEATURE_NAMES = ["contrast", "homogeneity", "energy", "correlation"]
GLCM_COLUMNS = [f"{band}_glcm_{feat}" for band in GLCM_BAND_NAMES for feat in GLCM_FEATURE_NAMES]


def field_dir_for(country: str, field_id: str) -> Path:
    return raw_country_dir(DATASET_ROOT, country) / field_id


def compute_texture_for_row(row: dict) -> dict:
    field_dir = field_dir_for(row["country"], row["field_id"])
    meta = load_field_metadata(field_dir)

    s2_paths = list_s2_image_paths(field_dir)
    scl_paths = list_scl_mask_paths(field_dir)
    date_strs = [p.stem.replace("S2_L2A_", "") for p in s2_paths]
    kept_indices = select_peak_window_indices(date_strs, meta.seeding_date, meta.harvesting_date)

    optical_frames, scl_frames = [], []
    for i in kept_indices:
        with rasterio.open(s2_paths[i]) as src:
            optical_frames.append(src.read().astype("float32") / S2_REFLECTANCE_SCALE)
        with rasterio.open(scl_paths[i]) as src:
            scl_frames.append(src.read(1))
    composite, _valid_count, _n_kept = build_cloud_free_composite(optical_frames, scl_frames)
    bgrn = select_bgrn_bands(composite)
    nir_band = bgrn[3]
    ndvi_map = compute_ndvi(bgrn)

    sar_coreg_path = SAR_CACHE_DIR / row["field_id"] / "sar_coregistered.tif"
    if not sar_coreg_path.exists():
        raise FileNotFoundError(f"no cached SAR composite at {sar_coreg_path}, cannot add SAR texture")
    with rasterio.open(sar_coreg_path) as src:
        vv, vh = src.read(1), src.read(2)

    # the cached file is already despeckled+coregistered from the original
    # run (lee_filter applied before backscatter_stats), texture features
    # should use the same despeckled bands, not raw speckled ones
    from preprocessing.sar import lee_filter
    vv_filtered, vh_filtered = lee_filter(vv), lee_filter(vh)

    return compute_field_texture_features({
        "nir": nir_band, "ndvi": ndvi_map, "vv": vv_filtered, "vh": vh_filtered,
    })


def main() -> None:
    if not IN_PATH.exists():
        raise SystemExit(f"{IN_PATH} not found, run scripts/build_feature_table.py first.")

    with open(IN_PATH, newline="") as f:
        rows = list(csv.DictReader(f))

    backup_path = IN_PATH.with_suffix(".csv.bak")
    shutil.copy(IN_PATH, backup_path)
    print(f"Backed up existing table to {backup_path}")

    n_ok, n_skip, n_fail = 0, 0, 0
    for i, row in enumerate(rows, 1):
        if row.get("status") != "ok":
            n_skip += 1
            continue
        try:
            texture = compute_texture_for_row(row)
            row.update(texture)
            n_ok += 1
            print(f"[{i}/{len(rows)}] {row['field_id']}: texture added")
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            row["status"] = f"ok, but texture failed: {exc}"
            print(f"[{i}/{len(rows)}] {row['field_id']}: TEXTURE FAILED ({exc})")
            traceback.print_exc(limit=1)

    fieldnames = list(rows[0].keys())
    for col in GLCM_COLUMNS:
        if col not in fieldnames:
            fieldnames.insert(fieldnames.index("status"), col)

    with open(IN_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. texture added: {n_ok}, skipped (not ok): {n_skip}, texture failed: {n_fail}")
    print(f"Updated {IN_PATH}")


if __name__ == "__main__":
    main()
