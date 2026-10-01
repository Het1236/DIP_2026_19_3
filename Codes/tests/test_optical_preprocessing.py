import numpy as np
import pytest

from preprocessing.optical import build_cloud_free_composite, build_valid_mask
from features.optical_indices import compute_ndvi, compute_evi, compute_gndvi, compute_ndwi, summarize_field_indices


def test_build_valid_mask_flags_cloud_classes():
    scl = np.array([[4, 3], [8, 9]])
    mask = build_valid_mask(scl)
    np.testing.assert_array_equal(mask, [[True, False], [False, False]])


def test_build_valid_mask_excludes_nodata_class_zero():
    # a real field raster is rectangular around a non-rectangular field
    # boundary, so class 0 (no data) shows up on every real field's corner
    # pixels and needs to be excluded, not just the cloud classes
    scl = np.array([[0, 4], [5, 0]])
    mask = build_valid_mask(scl)
    np.testing.assert_array_equal(mask, [[False, True], [True, False]])


def test_composite_uses_clear_date_when_other_date_is_fully_cloudy():
    size = 4
    clear_optical = np.full((4, size, size), 0.3, dtype=np.float32)
    clear_scl = np.full((size, size), 4, dtype=np.uint8)

    cloudy_optical = np.full((4, size, size), 0.9, dtype=np.float32)
    cloudy_scl = np.full((size, size), 9, dtype=np.uint8)

    composite, valid_count, n_kept = build_cloud_free_composite(
        [clear_optical, cloudy_optical], [clear_scl, cloudy_scl]
    )

    assert n_kept == 1
    np.testing.assert_allclose(composite, clear_optical)
    np.testing.assert_array_equal(valid_count, np.full((size, size), 1))


def test_composite_takes_median_of_valid_pixels_only():
    size = 2
    scl_all_clear = np.full((size, size), 4, dtype=np.uint8)

    frame_a = np.full((4, size, size), 0.2, dtype=np.float32)
    frame_b = np.full((4, size, size), 0.4, dtype=np.float32)
    frame_c = np.full((4, size, size), 0.6, dtype=np.float32)

    composite, valid_count, n_kept = build_cloud_free_composite(
        [frame_a, frame_b, frame_c], [scl_all_clear, scl_all_clear, scl_all_clear]
    )

    assert n_kept == 3
    np.testing.assert_allclose(composite, 0.4)
    np.testing.assert_array_equal(valid_count, np.full((size, size), 3))


def test_composite_raises_if_every_date_too_cloudy():
    size = 2
    optical = np.full((4, size, size), 0.5, dtype=np.float32)
    scl_cloud = np.full((size, size), 9, dtype=np.uint8)

    with pytest.raises(ValueError):
        build_cloud_free_composite([optical], [scl_cloud])


def test_ndvi_known_values():
    # band order: blue, green, red, nir
    composite = np.zeros((4, 1, 1), dtype=np.float32)
    composite[2, 0, 0] = 0.1  # red
    composite[3, 0, 0] = 0.5  # nir
    ndvi = compute_ndvi(composite)
    assert ndvi[0, 0] == pytest.approx((0.5 - 0.1) / (0.5 + 0.1), rel=1e-4)


def test_indices_stay_in_expected_range_on_random_data():
    rng = np.random.default_rng(0)
    composite = rng.uniform(0.01, 0.6, size=(4, 8, 8)).astype(np.float32)

    for fn in (compute_ndvi, compute_gndvi, compute_ndwi):
        values = fn(composite)
        assert np.all(values >= -1.01) and np.all(values <= 1.01)

    summary = summarize_field_indices(composite)
    assert set(summary.keys()) == {"mean_ndvi", "mean_evi", "mean_gndvi", "mean_ndwi"}
    assert all(np.isfinite(v) for v in summary.values())
