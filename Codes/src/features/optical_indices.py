"""Vegetation indices computed from a 4-band optical composite.

Band order matches data/synthetic.py's OPTICAL_BANDS: blue, green, red, nir.
Formulas match the ones explained in Learning_Notes/Project_Explained.docx
section 2.1.
"""

from __future__ import annotations

import numpy as np

BLUE, GREEN, RED, NIR = 0, 1, 2, 3
EPS = 1e-6


def compute_ndvi(composite: np.ndarray) -> np.ndarray:
    nir, red = composite[NIR], composite[RED]
    return (nir - red) / (nir + red + EPS)


def compute_evi(composite: np.ndarray) -> np.ndarray:
    nir, red, blue = composite[NIR], composite[RED], composite[BLUE]
    return 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + EPS)


def compute_gndvi(composite: np.ndarray) -> np.ndarray:
    nir, green = composite[NIR], composite[GREEN]
    return (nir - green) / (nir + green + EPS)


def compute_ndwi(composite: np.ndarray) -> np.ndarray:
    green, nir = composite[GREEN], composite[NIR]
    return (green - nir) / (green + nir + EPS)


def summarize_field_indices(composite: np.ndarray) -> dict:
    """Field-level mean of each index, ignoring NaN (cloud-gap) pixels."""
    with np.errstate(all="ignore"):
        return {
            "mean_ndvi": float(np.nanmean(compute_ndvi(composite))),
            "mean_evi": float(np.nanmean(compute_evi(composite))),
            "mean_gndvi": float(np.nanmean(compute_gndvi(composite))),
            "mean_ndwi": float(np.nanmean(compute_ndwi(composite))),
        }
