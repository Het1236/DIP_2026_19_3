import numpy as np
import pytest

from features.glcm import compute_field_texture_features, compute_glcm_features, quantize_image


def test_quantize_image_output_range_and_dtype():
    rng = np.random.default_rng(0)
    image = rng.uniform(0, 1, size=(16, 16)).astype(np.float32)
    quantized = quantize_image(image, levels=32)
    assert quantized.dtype == np.uint8
    assert quantized.min() >= 0
    assert quantized.max() <= 31


def test_quantize_image_fills_nan_instead_of_crashing():
    image = np.full((8, 8), 0.5, dtype=np.float32)
    image[0, 0] = np.nan
    image[3, 3] = np.nan
    quantized = quantize_image(image, levels=16)
    assert np.isfinite(quantized).all()


def test_quantize_image_raises_if_all_nan():
    image = np.full((4, 4), np.nan)
    with pytest.raises(ValueError):
        quantize_image(image)


def test_uniform_image_is_perfectly_smooth_texture():
    uniform = np.full((24, 24), 0.5, dtype=np.float32)
    feats = compute_glcm_features(uniform)
    assert feats["contrast"] == pytest.approx(0.0, abs=1e-6)
    assert feats["homogeneity"] == pytest.approx(1.0, abs=1e-6)
    assert feats["energy"] == pytest.approx(1.0, abs=1e-6)


def test_noisy_image_has_more_contrast_and_less_homogeneity_than_uniform():
    rng = np.random.default_rng(1)
    uniform = np.full((24, 24), 0.5, dtype=np.float32)
    noisy = rng.uniform(0, 1, size=(24, 24)).astype(np.float32)

    uniform_feats = compute_glcm_features(uniform)
    noisy_feats = compute_glcm_features(noisy)

    assert noisy_feats["contrast"] > uniform_feats["contrast"]
    assert noisy_feats["homogeneity"] < uniform_feats["homogeneity"]
    assert noisy_feats["energy"] < uniform_feats["energy"]


def test_compute_field_texture_features_returns_prefixed_keys():
    rng = np.random.default_rng(2)
    bands = {
        "nir": rng.uniform(0, 1, size=(16, 16)).astype(np.float32),
        "vv": rng.uniform(-20, -5, size=(16, 16)).astype(np.float32),
    }
    features = compute_field_texture_features(bands)

    expected_keys = {
        "nir_glcm_contrast", "nir_glcm_homogeneity", "nir_glcm_energy", "nir_glcm_correlation",
        "vv_glcm_contrast", "vv_glcm_homogeneity", "vv_glcm_energy", "vv_glcm_correlation",
    }
    assert set(features.keys()) == expected_keys
    assert all(np.isfinite(v) for v in features.values())
