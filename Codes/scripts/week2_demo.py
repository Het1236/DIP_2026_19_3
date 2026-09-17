"""Week 2 deliverable: prove the environment works end to end on real
GeoTIFF I/O, using synthetic data since real YieldSAT/Sentinel-1 access is
still pending.

Run from Codes/:
    python scripts/week2_demo.py

Generates a handful of synthetic fields, writes them as GeoTIFFs, reads
one back with rasterio, computes NDVI, and saves a figure comparing
optical, SAR and NDVI for one field per crop type to
Results/dev/week2_demo.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import rasterio

from data.config import COUNTRIES, CROP_TYPES, DEV_CACHE_DIR, RESULTS_DIR
from data.synthetic import generate_synthetic_dataset


def ndvi_from_optical(optical_path: str) -> "object":
    with rasterio.open(optical_path) as src:
        red = src.read(3).astype("float32")
        nir = src.read(4).astype("float32")
    return (nir - red) / (nir + red + 1e-6)


def main() -> None:
    out_dir = DEV_CACHE_DIR / "synthetic_fields"
    records = generate_synthetic_dataset(
        out_dir=out_dir,
        countries=COUNTRIES,
        crop_types=CROP_TYPES,
        n_fields_per_country_per_crop=2,
        size=64,
    )
    print(f"Generated {len(records)} synthetic fields under {out_dir}")

    # pick one field per crop type (first country) to visualize
    sample_by_crop = {}
    for r in records:
        if r.crop_type not in sample_by_crop:
            sample_by_crop[r.crop_type] = r

    fig, axes = plt.subplots(len(sample_by_crop), 3, figsize=(9, 3 * len(sample_by_crop)))
    if len(sample_by_crop) == 1:
        axes = [axes]

    for row, (crop_type, record) in enumerate(sample_by_crop.items()):
        with rasterio.open(record.optical_path) as src:
            rgb = src.read([3, 2, 1]).transpose(1, 2, 0)
            rgb = rgb / rgb.max()
        with rasterio.open(record.sar_path) as src:
            vv = src.read(1)
        ndvi = ndvi_from_optical(record.optical_path)

        axes[row][0].imshow(rgb)
        axes[row][0].set_title(f"{crop_type}: optical (RGB)")
        axes[row][1].imshow(vv, cmap="gray")
        axes[row][1].set_title(f"{crop_type}: SAR VV (dB)")
        im = axes[row][2].imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
        axes[row][2].set_title(f"{crop_type}: NDVI")
        for ax in axes[row]:
            ax.axis("off")

    fig.tight_layout()
    out_path = RESULTS_DIR / "dev" / "week2_demo.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"Saved demo figure to {out_path}")
    print("Note: this is synthetic dev data, not a project result.")


if __name__ == "__main__":
    main()
