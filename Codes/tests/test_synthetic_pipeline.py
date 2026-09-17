"""Week 2 sanity tests.

These don't test anything about real crops. They test that our own
environment and I/O plumbing (rasterio read/write, manifest, synthetic
generator) actually works, on a clean checkout, before anyone builds
real feature extraction on top of it.
"""

import numpy as np
import rasterio

from data.manifest import load_manifest
from data.synthetic import generate_synthetic_dataset


def test_generate_synthetic_dataset_writes_readable_geotiffs(tmp_path):
    records = generate_synthetic_dataset(
        out_dir=tmp_path,
        countries=["argentina"],
        crop_types=["corn", "wheat"],
        n_fields_per_country_per_crop=1,
        size=16,
    )

    assert len(records) == 2

    for record in records:
        with rasterio.open(record.optical_path) as src:
            optical = src.read()
        assert optical.shape == (4, 16, 16)
        assert np.isfinite(optical).all()

        with rasterio.open(record.sar_path) as src:
            sar = src.read()
        assert sar.shape == (2, 16, 16)
        assert np.isfinite(sar).all()


def test_generate_synthetic_dataset_is_reproducible(tmp_path):
    kwargs = dict(
        countries=["brazil"],
        crop_types=["soybean"],
        n_fields_per_country_per_crop=1,
        size=16,
    )
    records_a = generate_synthetic_dataset(out_dir=tmp_path / "a", **kwargs)
    records_b = generate_synthetic_dataset(out_dir=tmp_path / "b", **kwargs)

    with rasterio.open(records_a[0].optical_path) as src:
        a = src.read()
    with rasterio.open(records_b[0].optical_path) as src:
        b = src.read()

    np.testing.assert_array_equal(a, b)


def test_manifest_round_trip(tmp_path):
    records = generate_synthetic_dataset(
        out_dir=tmp_path,
        countries=["argentina", "brazil"],
        crop_types=["corn"],
        n_fields_per_country_per_crop=1,
        size=8,
    )
    loaded = load_manifest(tmp_path / "manifest.csv")
    assert len(loaded) == len(records)
    assert {r.field_id for r in loaded} == {r.field_id for r in records}


def test_crop_types_are_optically_separable_on_average(tmp_path):
    # not a claim about real crops, just a check that our synthetic
    # signal is not degenerate (constant / identical across crop types)
    records = generate_synthetic_dataset(
        out_dir=tmp_path,
        countries=["argentina"],
        crop_types=["corn", "wheat"],
        n_fields_per_country_per_crop=3,
        size=16,
    )

    def mean_nir(record):
        with rasterio.open(record.optical_path) as src:
            return src.read(4).mean()

    corn_nir = np.mean([mean_nir(r) for r in records if r.crop_type == "corn"])
    wheat_nir = np.mean([mean_nir(r) for r in records if r.crop_type == "wheat"])
    assert abs(corn_nir - wheat_nir) > 0.05
