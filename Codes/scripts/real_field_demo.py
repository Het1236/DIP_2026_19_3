"""Real end-to-end run: one real YieldSAT field per crop type (Argentina),
optical cloud masking + composite + indices, real Sentinel-1 pull +
despeckling + co-registration to the optical grid, all on the same field.

Must be run through `python -m` so Earth Engine picks up the local
credentials correctly:

    python -m scripts.real_field_demo

Needs dataset/ present locally and GEE_PROJECT_ID=fusioncrop set.
"""

import csv
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import rasterio

from data.config import DATASET_ROOT, GEE_PROJECT_ID, RESULTS_DIR
from data.gee_sentinel1 import aoi_from_raster_bounds, fetch_sentinel1_vvvh
from data.yieldsat_loader import (
    get_yield_mask_path,
    iter_country_field_dirs,
    load_field_metadata,
    load_real_field_frames,
    select_bgrn_bands,
)
from preprocessing.optical import build_cloud_free_composite
from preprocessing.sar import backscatter_stats, coregister_to_reference, lee_filter
from features.optical_indices import compute_ndvi, summarize_field_indices


def parse_yieldsat_date(date_str: str) -> str:
    """'04.11.2023' -> '2023-11-04', the format GEE's filterDate expects."""
    return dt.datetime.strptime(date_str, "%d.%m.%Y").strftime("%Y-%m-%d")


def pick_one_field_per_crop(country: str, crops: list[str]):
    chosen = {}
    for field_dir in iter_country_field_dirs(DATASET_ROOT, country):
        meta = load_field_metadata(field_dir)
        if meta.crop in crops and meta.crop not in chosen:
            chosen[meta.crop] = field_dir
        if len(chosen) == len(crops):
            break
    return chosen


def process_field(field_dir: Path, results_dir: Path, dev_cache_dir: Path) -> dict:
    meta = load_field_metadata(field_dir)
    print(f"\n=== {meta.field_id} ({meta.crop}) ===")

    optical_frames, scl_frames = load_real_field_frames(field_dir)
    composite, valid_count, n_kept = build_cloud_free_composite(optical_frames, scl_frames)
    bgrn = select_bgrn_bands(composite)
    indices = summarize_field_indices(bgrn)
    print(f"optical: kept {n_kept}/{len(optical_frames)} dates")
    print(f"indices: {indices}")

    yield_path = get_yield_mask_path(field_dir)
    aoi = aoi_from_raster_bounds(yield_path)
    start = parse_yieldsat_date(meta.seeding_date)
    end = parse_yieldsat_date(meta.harvesting_date)

    sar_raw_path = dev_cache_dir / "real_fields" / meta.field_id / "sar_raw.tif"
    try:
        sar_raw_path, n_scenes = fetch_sentinel1_vvvh(aoi, start, end, sar_raw_path)
        print(f"SAR: pulled composite from {n_scenes} scenes ({start} to {end})")
    except ValueError as exc:
        print(f"SAR: {exc}")
        return {"field_id": meta.field_id, "crop": meta.crop, "n_dates_kept": n_kept,
                "n_dates_total": len(optical_frames), **indices, "sar_status": "no scenes"}

    with rasterio.open(sar_raw_path) as src:
        vv_raw, vh_raw = src.read(1), src.read(2)

    sar_coreg_path = coregister_to_reference(sar_raw_path, yield_path, dev_cache_dir / "real_fields" / meta.field_id / "sar_coregistered.tif")
    with rasterio.open(sar_coreg_path) as src:
        vv_coreg, vh_coreg = src.read(1), src.read(2)

    vv_filtered = lee_filter(vv_coreg, window=5)
    vh_filtered = lee_filter(vh_coreg, window=5)
    sar_stats = backscatter_stats(vv_filtered, vh_filtered)
    print(f"SAR stats (despeckled, co-registered): {sar_stats}")

    ndvi = compute_ndvi(bgrn)
    rgb = np.clip(bgrn[[2, 1, 0]].transpose(1, 2, 0) * 3.5, 0, 1)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    axes[0].imshow(rgb)
    axes[0].set_title(f"{meta.field_id}\noptical composite (RGB)")
    axes[1].imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
    axes[1].set_title("NDVI")
    axes[2].imshow(vv_raw, cmap="gray")
    axes[2].set_title(f"SAR VV, raw\n({vv_raw.shape[0]}x{vv_raw.shape[1]}, own grid)")
    axes[3].imshow(vv_filtered, cmap="gray")
    axes[3].set_title(f"SAR VV, despeckled\n+ co-registered to optical grid\n({vv_filtered.shape[0]}x{vv_filtered.shape[1]})")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    out_path = results_dir / f"{meta.field_id}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved {out_path}")

    return {
        "field_id": meta.field_id, "crop": meta.crop,
        "n_dates_kept": n_kept, "n_dates_total": len(optical_frames),
        **indices, "sar_status": f"{n_scenes} scenes", **sar_stats,
    }


def main() -> None:
    if not GEE_PROJECT_ID:
        raise SystemExit("Set GEE_PROJECT_ID=fusioncrop before running this script.")
    if not DATASET_ROOT.exists():
        raise SystemExit(f"{DATASET_ROOT} not found.")

    results_dir = RESULTS_DIR / "real_fields"
    results_dir.mkdir(parents=True, exist_ok=True)
    dev_cache_dir = DATASET_ROOT.parent / "Codes" / ".dev_cache"

    fields = pick_one_field_per_crop("Argentina", ["corn", "soybean", "wheat"])
    rows = []
    for crop, field_dir in fields.items():
        rows.append(process_field(field_dir, results_dir, dev_cache_dir))

    summary_path = results_dir / "summary.csv"
    all_keys = sorted({k for row in rows for k in row.keys()}, key=lambda k: (k != "field_id", k))
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {summary_path}")


if __name__ == "__main__":
    main()
