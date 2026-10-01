"""Demo of despeckling and backscatter statistics, run on a real Sentinel-1
composite pulled via GEE (src/data/gee_sentinel1.py). Not tied to a real
YieldSAT field yet, just a real farmland AOI used to check the filter
behaves correctly on real sensor noise.

Run from Codes/ after src/data/gee_sentinel1.py has been run once to
download the sample AOI (python -m src.data.gee_sentinel1):
    python scripts/week4_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import rasterio

from data.config import DEV_CACHE_DIR, RESULTS_DIR
from preprocessing.sar import backscatter_stats, lee_filter


def main() -> None:
    sar_path = DEV_CACHE_DIR / "real_sar_test" / "sample_aoi.tif"
    if not sar_path.exists():
        raise SystemExit(
            f"{sar_path} not found. Run `python -m src.data.gee_sentinel1` first to pull it."
        )

    with rasterio.open(sar_path) as src:
        vv, vh = src.read(1), src.read(2)

    vv_filtered = lee_filter(vv, window=5)
    vh_filtered = lee_filter(vh, window=5)

    stats_raw = backscatter_stats(vv, vh)
    stats_filtered = backscatter_stats(vv_filtered, vh_filtered)

    print(f"Field size: {vv.shape[0]} x {vv.shape[1]} pixels")
    print(f"Raw       : {stats_raw}")
    print(f"Despeckled: {stats_filtered}")
    print(f"VV pixel-to-pixel std, raw vs despeckled: {vv.std():.3f} -> {vv_filtered.std():.3f}")

    results_dir = RESULTS_DIR / "sar"
    results_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for ax, arr, title in zip(
        axes.flat,
        [vv, vv_filtered, vh, vh_filtered],
        ["VV, raw", "VV, Lee filtered", "VH, raw", "VH, Lee filtered"],
    ):
        im = ax.imshow(arr, cmap="gray", vmin=np.percentile(arr, 2), vmax=np.percentile(arr, 98))
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("Real Sentinel-1 sample AOI, before and after despeckling")
    fig.tight_layout()
    out_path = results_dir / "week4_real_sar_despeckle.png"
    fig.savefig(out_path, dpi=130)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
