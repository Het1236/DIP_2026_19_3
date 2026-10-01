"""Validation against the actual downloaded YieldSAT dataset, not a fixture.

Skips automatically if dataset/ isn't present, e.g. on a teammate's machine
before they've downloaded it, or in any environment without the ~15GB of
real data. This is the test that matters most: everything else in
test_yieldsat_loader.py only proves the adapter works against a fixture we
built to match the documented shape, this proves it works against real
bytes.
"""

import numpy as np
import pytest
import rasterio

from data.config import COUNTRIES, DATASET_ROOT
from data.yieldsat_loader import (
    S2_BAND_ORDER,
    S2_NODATA_VALUE,
    get_yield_mask_path,
    iter_country_field_dirs,
    list_s2_image_paths,
    list_scl_mask_paths,
    load_field_metadata,
    raw_country_dir,
)

pytestmark = pytest.mark.skipif(
    not DATASET_ROOT.exists(), reason="real YieldSAT dataset not present at dataset/"
)


def _first_field_dir(country="Argentina"):
    return next(iter_country_field_dirs(DATASET_ROOT, country))


def test_expected_field_counts_close_to_paper():
    # paper reports 751 Argentina fields, 551 Brazil fields (Table 2)
    for country, expected in [("Argentina", 751), ("Brazil", 551)]:
        n_fields = sum(1 for _ in iter_country_field_dirs(DATASET_ROOT, country))
        assert abs(n_fields - expected) <= 5, f"{country}: got {n_fields}, expected close to {expected}"


def test_metadata_matches_folder_name():
    field_dir = _first_field_dir()
    meta = load_field_metadata(field_dir)
    assert meta.field_id == field_dir.name
    assert meta.country == "Argentina"
    assert meta.crop in {"corn", "soybean", "wheat"}
    assert meta.projected_crs_epsg > 0


def test_s2_and_scl_dates_align_for_a_real_field():
    field_dir = _first_field_dir()
    s2_paths = list_s2_image_paths(field_dir)
    scl_paths = list_scl_mask_paths(field_dir)
    assert len(s2_paths) > 0
    assert len(s2_paths) == len(scl_paths)

    s2_dates = [p.name.replace("S2_L2A_", "") for p in s2_paths]
    scl_dates = [p.name.replace("S2_L2A_SCL_", "") for p in scl_paths]
    assert s2_dates == scl_dates


def test_s2_image_band_order_and_shape():
    field_dir = _first_field_dir()
    s2_path = list_s2_image_paths(field_dir)[0]
    with rasterio.open(s2_path) as src:
        assert src.count == len(S2_BAND_ORDER)
        assert src.nodata == S2_NODATA_VALUE
        data = src.read()
    assert data.dtype == np.uint16
    # every band should have some real, non-nodata reflectance values
    assert (data > 0).any()


def test_yield_mask_exists_and_is_same_grid_as_s2():
    field_dir = _first_field_dir()
    s2_path = list_s2_image_paths(field_dir)[0]
    yield_path = get_yield_mask_path(field_dir)
    assert yield_path.exists()

    with rasterio.open(s2_path) as s2, rasterio.open(yield_path) as yld:
        assert s2.width == yld.width
        assert s2.height == yld.height
        assert s2.crs == yld.crs


def test_all_expected_countries_present():
    for country in COUNTRIES:
        assert raw_country_dir(DATASET_ROOT, country).is_dir()
