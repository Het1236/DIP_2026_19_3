"""Despeckling and grid alignment for a field's SAR composite.

The despeckling formula matches Learning_Notes/Project_Explained.docx
section 2.2: I_hat = I_bar + k * (I - I_bar), where k depends on how noisy
the local window looks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject


def lee_filter(band: np.ndarray, window: int = 5) -> np.ndarray:
    """Simplified Lee despeckling filter, applied per band (VV or VH,
    already in dB). window is the size of the local neighbourhood used to
    estimate local mean and variance.
    """
    from scipy.ndimage import uniform_filter

    mean = uniform_filter(band, size=window)
    mean_sq = uniform_filter(band.astype(np.float64) ** 2, size=window)
    local_variance = np.clip(mean_sq - mean ** 2, 0, None)
    overall_variance = local_variance.mean()

    weight = local_variance / (local_variance + overall_variance + 1e-6)
    return (mean + weight * (band - mean)).astype(np.float32)


def coregister_to_reference(
    src_path,
    reference_path,
    out_path,
    resampling: Resampling = Resampling.bilinear,
):
    """Resample src_path onto reference_path's exact grid (transform, CRS,
    width, height), so SAR and optical pixels line up one-to-one.
    """
    with rasterio.open(reference_path) as ref:
        ref_transform, ref_crs = ref.transform, ref.crs
        ref_width, ref_height = ref.width, ref.height

    with rasterio.open(src_path) as src:
        dst_array = np.zeros((src.count, ref_height, ref_width), dtype=np.float32)
        for band_index in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, band_index),
                destination=dst_array[band_index - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=resampling,
            )
        profile = src.profile.copy()

    profile.update(height=ref_height, width=ref_width, transform=ref_transform, crs=ref_crs, count=dst_array.shape[0])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(dst_array)
    return out_path


def backscatter_stats(vv: np.ndarray, vh: np.ndarray) -> dict:
    """Field-level SAR statistics. VV and VH are in dB, so their ratio in
    linear units is just their difference in dB.
    """
    return {
        "mean_vv_db": float(np.nanmean(vv)),
        "std_vv_db": float(np.nanstd(vv)),
        "mean_vh_db": float(np.nanmean(vh)),
        "std_vh_db": float(np.nanstd(vh)),
        "mean_vv_minus_vh_db": float(np.nanmean(vv - vh)),
    }
