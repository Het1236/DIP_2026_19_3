"""Cloud masking and cloud-free compositing for a field's Sentinel-2 time
series.

Takes the per-date (optical, SCL) pairs described in
data/yieldsat_loader.py (real layout) and data/synthetic_timeseries.py (dev
layout) and reduces them to one clean composite per field.
"""

from __future__ import annotations

import warnings

import numpy as np

# Sentinel-2 SCL classes we treat as usable, as an allow-list rather than a
# block-list: 4 vegetation, 5 not-vegetated/bare soil, 6 water, 7
# unclassified. Everything else (0 no data, 1 saturated, 2 dark area, 3
# cloud shadow, 8/9 cloud medium/high probability, 10 cirrus, 11 snow) is
# excluded. A real field raster is a rectangle around a non-rectangular
# field boundary, so class 0 (no data) shows up on every field's corner
# pixels, an allow-list keeps those out automatically instead of needing
# every unwanted class listed explicitly.
USABLE_SCL_CLASSES = {4, 5, 6, 7}


def build_valid_mask(scl: np.ndarray) -> np.ndarray:
    """True where a pixel's SCL class is usable (real field data, not
    cloud, shadow, snow, saturated, or outside the field boundary).
    """
    return np.isin(scl, list(USABLE_SCL_CLASSES))


def build_cloud_free_composite(
    optical_frames: list[np.ndarray],
    scl_frames: list[np.ndarray],
    min_valid_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Combine a field's time series into one cloud-free composite.

    Parameters
    ----------
    optical_frames: list of (bands, H, W) arrays, one per acquisition date.
    scl_frames: list of (H, W) SCL arrays, same dates, same order.
    min_valid_fraction: dates with fewer usable pixels than this fraction
        of the field are dropped entirely before compositing, rather than
        contributing mostly-cloud pixels to the median.

    Returns
    -------
    composite: (bands, H, W) array, median of valid observations per pixel.
        A pixel with zero valid observations across all kept dates is NaN.
    valid_count: (H, W) array, how many dates contributed to each pixel.
    n_dates_kept: how many of the input dates passed the cloud-fraction check.
    """
    if len(optical_frames) != len(scl_frames):
        raise ValueError("optical_frames and scl_frames must be the same length")
    if not optical_frames:
        raise ValueError("need at least one date to build a composite")

    kept_optical = []
    kept_masks = []
    for optical, scl in zip(optical_frames, scl_frames):
        valid_mask = build_valid_mask(scl)
        if valid_mask.mean() < min_valid_fraction:
            continue
        kept_optical.append(optical)
        kept_masks.append(valid_mask)

    n_dates_kept = len(kept_optical)
    if n_dates_kept == 0:
        raise ValueError("every date was too cloudy, nothing to composite")

    bands, h, w = kept_optical[0].shape
    stacked = np.stack(kept_optical, axis=0)  # (dates, bands, H, W)
    mask_stack = np.stack(kept_masks, axis=0)  # (dates, H, W)

    masked = np.where(mask_stack[:, None, :, :], stacked, np.nan)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        # corner pixels outside the field boundary can be nodata on every
        # date, so an all-NaN slice here is expected, not a bug
        warnings.simplefilter("ignore", category=RuntimeWarning)
        composite = np.nanmedian(masked, axis=0)  # (bands, H, W)

    valid_count = mask_stack.sum(axis=0)
    return composite.astype(np.float32), valid_count.astype(np.int32), n_dates_kept
