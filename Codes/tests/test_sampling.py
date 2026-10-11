import json

from data.sampling import select_sample_fields

BASE_METADATA = {
    "adm_units": {"country": "Argentina"},
    "area_calculated": 40.0,
    "centroid_latitude_wgs84": -33.0,
    "centroid_longitude_wgs84": -62.0,
    "farm_identifier": "farm1",
    "harvesting_date": "01.04.2021",
    "projected_crs_epsg": 32720,
    "seeding_date": "15.11.2020",
    "year": 2021,
}


def _make_field(root, field_id, crop, quality):
    field_dir = root / "Argentina" / "Argentina_raw" / field_id
    field_dir.mkdir(parents=True)
    meta = dict(BASE_METADATA, crop=crop, field_shared_name=field_id, yieldmap_quality=quality)
    (field_dir / f"metadata-{field_id}.json").write_text(json.dumps(meta))
    return field_dir


def test_select_sample_fields_respects_group_cap_and_quality(tmp_path):
    for i in range(5):
        _make_field(tmp_path, f"field_corn_good_{i}", "corn", "Good")
    for i in range(3):
        _make_field(tmp_path, f"field_corn_bad_{i}", "corn", "Bad")
    for i in range(2):
        _make_field(tmp_path, f"field_wheat_avg_{i}", "wheat", "Average")

    selected = select_sample_fields(
        dataset_root=tmp_path, countries=["Argentina"], crops=["corn", "wheat"],
        n_per_group=3, quality_allowed={"Good", "Average"},
    )
    selected_names = {p.name for p in selected}

    assert len(selected) == 5  # 3 corn (capped from 5 good) + 2 wheat (all available)
    assert all("bad" not in name for name in selected_names)


def test_select_sample_fields_is_reproducible_with_same_seed(tmp_path):
    for i in range(10):
        _make_field(tmp_path, f"field_soybean_{i}", "soybean", "Good")

    first = select_sample_fields(tmp_path, ["Argentina"], ["soybean"], n_per_group=4, seed=7)
    second = select_sample_fields(tmp_path, ["Argentina"], ["soybean"], n_per_group=4, seed=7)

    assert [p.name for p in first] == [p.name for p in second]
