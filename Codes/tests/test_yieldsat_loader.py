"""Tests for the YieldSAT raw-release adapter, against a fake field folder
built to match the documented structure exactly (see yieldsat_loader.py's
module docstring). This does not prove the adapter works on real YieldSAT
files, we don't have any yet, it only proves the parsing logic is correct
against the structure YieldSAT's own documentation describes.
"""

import json

import numpy as np

from data.yieldsat_loader import (
    S2_BAND_ORDER,
    get_yield_mask_path,
    iter_country_field_dirs,
    list_s2_image_paths,
    list_scl_mask_paths,
    load_field_metadata,
    select_bgrn_bands,
    YIELD_MASK_FILENAME,
)

FIELD_ID = "Argentina_TEST_farm1_field001_soybean_2021"

METADATA = {
    "adm_units": {"adm_1": "Buenos Aires", "adm_2": "Test District", "country": "Argentina"},
    "area_calculated": 42.0,
    "area_ground_truth": 43.5,
    "centroid_latitude_wgs84": -34.5,
    "centroid_longitude_wgs84": -62.4,
    "crop": "soybean",
    "data_provider": "TEST",
    "farm_identifier": "farm1",
    "field_shared_name": FIELD_ID,
    "harvesting_date": "01.04.2021",
    "max_yield_per_hectare": 6,
    "min_yield_per_hectare": 0,
    "projected_crs_epsg": 32720,
    "quality_density": "good",
    "seeding_date": "15.11.2020",
    "seeding_date_type": "provided_by_farmer",
    "standard_moisture": 13,
    "year": 2021,
    "yield_ground_truth": 3.1,
    "yieldmap_quality": "Good",
}


def _build_fake_field(tmp_path):
    field_root = tmp_path / "Argentina" / "Argentina_raw" / FIELD_ID
    (field_root / "s2_images").mkdir(parents=True)
    (field_root / "scl_masks").mkdir(parents=True)
    (field_root / "yield_masks").mkdir(parents=True)
    (field_root / "dem").mkdir(parents=True)
    (field_root / "soil").mkdir(parents=True)
    (field_root / "weather").mkdir(parents=True)

    (field_root / f"metadata-{FIELD_ID}.json").write_text(json.dumps(METADATA))

    for date in ["20201201", "20201216", "20210105"]:
        (field_root / "s2_images" / f"S2_L2A_{date}.tif").write_bytes(b"")
        (field_root / "scl_masks" / f"S2_L2A_SCL_{date}.tif").write_bytes(b"")

    (field_root / "yield_masks" / YIELD_MASK_FILENAME).write_bytes(b"")
    (field_root / "weather" / f"{FIELD_ID}.csv").write_text("Date,Temp_mean\n2020-12-01,290\n")

    return field_root


def test_load_field_metadata(tmp_path):
    field_root = _build_fake_field(tmp_path)
    meta = load_field_metadata(field_root)

    assert meta.field_id == FIELD_ID
    assert meta.country == "Argentina"
    assert meta.crop == "soybean"
    assert meta.projected_crs_epsg == 32720
    assert meta.seeding_date == "15.11.2020"
    assert meta.harvesting_date == "01.04.2021"


def test_list_s2_and_scl_paths_are_date_aligned(tmp_path):
    field_root = _build_fake_field(tmp_path)
    s2_paths = list_s2_image_paths(field_root)
    scl_paths = list_scl_mask_paths(field_root)

    assert len(s2_paths) == 3
    assert len(scl_paths) == 3

    s2_dates = [p.name.replace("S2_L2A_", "").replace(".tif", "") for p in s2_paths]
    scl_dates = [p.name.replace("S2_L2A_SCL_", "").replace(".tif", "") for p in scl_paths]
    assert s2_dates == scl_dates


def test_yield_mask_path(tmp_path):
    field_root = _build_fake_field(tmp_path)
    path = get_yield_mask_path(field_root)
    assert path.exists()
    assert path.name == YIELD_MASK_FILENAME


def test_select_bgrn_bands_picks_correct_indices():
    # build a fake 12-band stack where each band's value equals its index,
    # so picking band i should just return an array full of the value i
    n_bands = len(S2_BAND_ORDER)
    composite = np.zeros((n_bands, 2, 2), dtype=np.float32)
    for i in range(n_bands):
        composite[i] = i

    bgrn = select_bgrn_bands(composite)
    assert bgrn.shape == (4, 2, 2)
    expected = [S2_BAND_ORDER.index(b) for b in ["B02", "B03", "B04", "B08"]]
    for out_index, band_index in enumerate(expected):
        np.testing.assert_array_equal(bgrn[out_index], band_index)


def test_iter_country_field_dirs(tmp_path):
    _build_fake_field(tmp_path)
    (tmp_path / "Argentina" / "Argentina_raw" / "Argentina_TEST_farm1_field002_corn_2021").mkdir(parents=True)

    field_dirs = list(iter_country_field_dirs(tmp_path, "Argentina"))
    assert len(field_dirs) == 2
    assert all(d.is_dir() for d in field_dirs)
