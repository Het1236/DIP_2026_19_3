import numpy as np
import rasterio

from data.synthetic_timeseries import generate_synthetic_timeseries_dataset


def test_generates_expected_folder_shape(tmp_path):
    records = generate_synthetic_timeseries_dataset(
        out_dir=tmp_path, crop_types=["corn", "wheat"], n_fields_per_crop=1, n_dates=5, size=16,
    )
    assert len(records) == 2

    for record in records:
        s2_files = sorted((record["field_root"] / "s2_images").glob("*.tif"))
        scl_files = sorted((record["field_root"] / "scl_masks").glob("*.tif"))
        assert len(s2_files) == 5
        assert len(scl_files) == 5

        with rasterio.open(s2_files[0]) as src:
            assert src.read().shape == (4, 16, 16)
        with rasterio.open(scl_files[0]) as src:
            assert src.read(1).shape == (16, 16)


def test_some_dates_are_synthetically_cloudy(tmp_path):
    # with enough dates, at least one should end up flagged cloudy somewhere
    records = generate_synthetic_timeseries_dataset(
        out_dir=tmp_path, crop_types=["soybean"], n_fields_per_crop=1, n_dates=8, size=16,
    )
    field_root = records[0]["field_root"]
    scl_files = sorted((field_root / "scl_masks").glob("*.tif"))

    any_cloud_pixel = False
    for path in scl_files:
        with rasterio.open(path) as src:
            scl = src.read(1)
        if np.any(scl == 9):
            any_cloud_pixel = True
    assert any_cloud_pixel
