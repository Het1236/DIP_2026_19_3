"""The farm of a field is read from its field_id by scripts/farm_grouped_cv.py.
The report's "42 farms" comes from this, so it must parse every field in the
committed feature table and must never mix two countries into one farm."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

CODES = Path(__file__).resolve().parents[1]
SCRIPT = CODES / "scripts" / "farm_grouped_cv.py"
TABLE = CODES.parent / "Results" / "features" / "real_field_features.csv"


@pytest.fixture(scope="module")
def farm_module():
    spec = importlib.util.spec_from_file_location("farm_grouped_cv", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_farm_of_reads_country_code_and_farm_number(farm_module):
    assert farm_module.farm_of("Argentina_DUP1_farm14_field117_corn_2023") == "Argentina_DUP1_farm14"
    assert farm_module.farm_of("Brazil_DUP2_farm6_field355_wheat_2017") == "Brazil_DUP2_farm6"


def test_same_farm_number_in_two_countries_is_two_farms(farm_module):
    a = farm_module.farm_of("Argentina_DUP1_farm5_field1_corn_2020")
    b = farm_module.farm_of("Brazil_DUP2_farm5_field1_corn_2020")
    assert a != b


def test_farm_of_rejects_a_field_id_it_cannot_read(farm_module):
    with pytest.raises(ValueError):
        farm_module.farm_of("not_a_field_id")


@pytest.mark.skipif(not TABLE.exists(), reason="feature table not present")
def test_every_field_in_the_committed_table_has_a_farm(farm_module):
    df = pd.read_csv(TABLE)
    df = df[df["status"] == "ok"]
    farms = df["field_id"].map(farm_module.farm_of)
    assert len(df) == 270
    assert farms.nunique() == 42  # the number quoted in the report
    assert farms.value_counts().max() == 43  # the largest farm, also quoted in the report
