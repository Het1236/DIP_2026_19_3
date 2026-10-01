"""Demo of the cloud masking and vegetation index pipeline, run on synthetic
multi-date optical data before real YieldSAT access was available.

Run from Codes/:
    python scripts/week3_demo.py

For each synthetic field: builds a cloud-free composite from its multi-date
time series, computes NDVI/EVI/GNDVI/NDWI, saves a figure per field and a
summary CSV to Results/optical/.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import rasterio

from data.config import CROP_TYPES, DEV_CACHE_DIR, RESULTS_DIR
from data.synthetic_timeseries import generate_synthetic_timeseries_dataset
from preprocessing.optical import build_cloud_free_composite, build_valid_mask
from features.optical_indices import compute_ndvi, summarize_field_indices


def load_frames(field_root: Path):
    s2_paths = sorted((field_root / "s2_images").glob("*.tif"))
    scl_paths = sorted((field_root / "scl_masks").glob("*.tif"))
    optical_frames, scl_frames = [], []
    for s2_path, scl_path in zip(s2_paths, scl_paths):
        with rasterio.open(s2_path) as src:
            optical_frames.append(src.read())
        with rasterio.open(scl_path) as src:
            scl_frames.append(src.read(1))
    return optical_frames, scl_frames


def main() -> None:
    out_dir = DEV_CACHE_DIR / "synthetic_timeseries"
    records = generate_synthetic_timeseries_dataset(
        out_dir=out_dir, crop_types=CROP_TYPES, n_fields_per_crop=2, n_dates=8, size=48,
    )
    print(f"Generated {len(records)} synthetic multi-date fields under {out_dir}")

    results_dir = RESULTS_DIR / "optical"
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for record in records:
        field_id, crop_type, field_root = record["field_id"], record["crop_type"], record["field_root"]
        optical_frames, scl_frames = load_frames(field_root)

        cloud_fractions = [1 - build_valid_mask(scl).mean() for scl in scl_frames]
        composite, valid_count, n_kept = build_cloud_free_composite(optical_frames, scl_frames)
        indices = summarize_field_indices(composite)

        summary_rows.append({
            "field_id": field_id,
            "crop_type": crop_type,
            "n_dates_total": len(optical_frames),
            "n_dates_kept": n_kept,
            **indices,
        })

        rgb = composite[[2, 1, 0]].transpose(1, 2, 0)
        rgb = np.clip(rgb / (np.nanmax(rgb) + 1e-6), 0, 1)
        ndvi = compute_ndvi(composite)

        fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
        axes[0].imshow(rgb)
        axes[0].set_title(f"{field_id}\ncloud-free composite (RGB)")
        im = axes[1].imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
        axes[1].set_title("NDVI")
        axes[2].imshow(valid_count, cmap="viridis")
        axes[2].set_title(f"valid observations per pixel\n({n_kept}/{len(optical_frames)} dates kept)")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(results_dir / f"{field_id}.png", dpi=120)
        plt.close(fig)

        print(f"{field_id} ({crop_type}): kept {n_kept}/{len(optical_frames)} dates, "
              f"cloud fraction per date = {[f'{c:.2f}' for c in cloud_fractions]}")

    summary_path = results_dir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved per-field figures and {summary_path}")
    print("Note: this is synthetic dev data, not a project result.")


if __name__ == "__main__":
    main()
