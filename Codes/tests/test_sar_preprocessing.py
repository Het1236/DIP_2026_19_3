import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from preprocessing.sar import backscatter_stats, coregister_to_reference, lee_filter


def test_lee_filter_reduces_variance_on_speckled_constant_field():
    rng = np.random.default_rng(0)
    true_value = -10.0
    looks = 3
    speckle_db = 10 * np.log10(rng.gamma(shape=looks, scale=1.0 / looks, size=(64, 64)))
    noisy = true_value + speckle_db

    filtered = lee_filter(noisy, window=5)

    assert filtered.var() < noisy.var()
    # the filter should not drag the field mean far from the true value
    assert abs(filtered.mean() - true_value) < abs(noisy.mean() - true_value) + 0.5


def test_lee_filter_preserves_shape():
    band = np.random.default_rng(1).normal(-12, 1, size=(32, 40)).astype(np.float32)
    filtered = lee_filter(band, window=3)
    assert filtered.shape == band.shape


def _write_raster(path, array, transform, crs="EPSG:4326"):
    count = array.shape[0]
    with rasterio.open(
        path, "w", driver="GTiff", height=array.shape[1], width=array.shape[2],
        count=count, dtype=array.dtype, crs=crs, transform=transform,
    ) as dst:
        dst.write(array)


def test_coregister_to_reference_matches_reference_grid(tmp_path):
    ref_transform = from_origin(-60.0, -30.0, 0.0001, 0.0001)
    ref_array = np.zeros((1, 20, 20), dtype=np.float32)
    ref_path = tmp_path / "reference.tif"
    _write_raster(ref_path, ref_array, ref_transform)

    # source grid deliberately different resolution and origin offset
    src_transform = from_origin(-60.0005, -29.9995, 0.00015, 0.00015)
    src_array = np.full((1, 15, 15), 7.0, dtype=np.float32)
    src_path = tmp_path / "source.tif"
    _write_raster(src_path, src_array, src_transform)

    out_path = coregister_to_reference(src_path, ref_path, tmp_path / "coregistered.tif")

    with rasterio.open(out_path) as out, rasterio.open(ref_path) as ref:
        assert out.transform == ref.transform
        assert out.width == ref.width
        assert out.height == ref.height
        data = out.read(1)

    # source was a uniform 7.0 field, reprojecting onto a different grid
    # should still leave most interior pixels close to 7.0
    interior = data[3:-3, 3:-3]
    assert np.nanmean(interior) == pytest.approx(7.0, abs=0.5)


def test_backscatter_stats_known_values():
    vv = np.array([-8.0, -10.0, -12.0])
    vh = np.array([-14.0, -16.0, -18.0])
    stats = backscatter_stats(vv, vh)

    assert stats["mean_vv_db"] == pytest.approx(-10.0)
    assert stats["mean_vh_db"] == pytest.approx(-16.0)
    assert stats["mean_vv_minus_vh_db"] == pytest.approx(6.0)
