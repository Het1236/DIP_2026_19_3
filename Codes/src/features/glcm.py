"""GLCM (Gray Level Co-occurrence Matrix) texture features.

Concept and formulas explained in depth in
Learning_Notes/Project_Explained.docx section 4.2. In short: two pixels the
same average brightness can look completely different (smooth vs patchy),
and GLCM captures that difference as a few summary numbers computed from
how often pairs of neighbouring pixel values occur together.

We average over 4 directions (0, 45, 90, 135 degrees) at distance 1, a
standard way to make the texture measure not depend on which way the field
happens to be oriented in the image.
"""

from __future__ import annotations

import numpy as np
from skimage.feature import graycomatrix, graycoprops

DEFAULT_LEVELS = 32
DEFAULT_DISTANCES = [1]
DEFAULT_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]


def quantize_image(image_2d: np.ndarray, levels: int = DEFAULT_LEVELS) -> np.ndarray:
    """Reduce a float image to `levels` gray levels as uint8, the format
    graycomatrix needs. NaN pixels (cloud gaps, field-boundary nodata) are
    filled with the image's own mean first, GLCM has no concept of missing
    data, and dropping pixels would break the neighbour-pair structure it
    depends on. Clips to the 1st-99th percentile before scaling so a
    handful of extreme outlier pixels don't compress everything else into
    one or two gray levels.
    """
    valid = np.isfinite(image_2d)
    if not valid.any():
        raise ValueError("image has no finite pixels to compute texture from")

    filled = np.where(valid, image_2d, np.nanmean(image_2d[valid]))
    lo, hi = np.nanpercentile(filled, [1, 99])
    if hi <= lo:
        hi = lo + 1e-6
    clipped = np.clip(filled, lo, hi)
    scaled = (clipped - lo) / (hi - lo) * (levels - 1)
    return scaled.astype(np.uint8)


def compute_glcm_features(
    image_2d: np.ndarray,
    levels: int = DEFAULT_LEVELS,
    distances: list[int] = None,
    angles: list[float] = None,
) -> dict:
    """The four Haralick features from Learning_Notes section 4.2, for one
    2D band. Averaged across all distance/angle pairs requested.
    """
    distances = distances or DEFAULT_DISTANCES
    angles = angles or DEFAULT_ANGLES

    quantized = quantize_image(image_2d, levels=levels)
    glcm = graycomatrix(
        quantized, distances=distances, angles=angles,
        levels=levels, symmetric=True, normed=True,
    )
    return {
        "contrast": float(graycoprops(glcm, "contrast").mean()),
        "homogeneity": float(graycoprops(glcm, "homogeneity").mean()),
        # scikit-image's own "energy" property is sqrt(ASM); "ASM" is the
        # raw sum(P(i,j)^2), which is what Learning_Notes calls Energy.
        # Using ASM here to match our own documented formula exactly.
        "energy": float(graycoprops(glcm, "ASM").mean()),
        "correlation": float(graycoprops(glcm, "correlation").mean()),
    }


def compute_field_texture_features(bands: dict[str, np.ndarray], levels: int = DEFAULT_LEVELS) -> dict:
    """bands: e.g. {"nir": array, "ndvi": array, "vv": array, "vh": array}.
    Returns one flat dict with keys like "nir_glcm_contrast", ready to
    merge straight into a feature table row.
    """
    out = {}
    for band_name, array in bands.items():
        for feature_name, value in compute_glcm_features(array, levels=levels).items():
            out[f"{band_name}_glcm_{feature_name}"] = value
    return out
