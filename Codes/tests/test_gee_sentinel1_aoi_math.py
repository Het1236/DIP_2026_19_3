import pytest

from data.gee_sentinel1 import square_bbox_from_centroid


def test_square_bbox_at_equator_is_symmetric():
    # 1 sq km field at the equator, side = 1000 m
    min_lon, min_lat, max_lon, max_lat = square_bbox_from_centroid(lat=0.0, lon=0.0, area_ha=100)

    half_lat_deg = max_lat
    half_lon_deg = max_lon
    # at the equator, cos(lat) == 1, so degrees-per-metre is the same in both directions
    assert half_lat_deg == pytest.approx(half_lon_deg, rel=1e-6)
    assert half_lat_deg == pytest.approx(500 / 111320, rel=1e-3)


def test_square_bbox_shrinks_in_longitude_away_from_equator():
    bbox_equator = square_bbox_from_centroid(lat=0.0, lon=0.0, area_ha=100)
    bbox_high_lat = square_bbox_from_centroid(lat=60.0, lon=0.0, area_ha=100)

    lon_span_equator = bbox_equator[2] - bbox_equator[0]
    lon_span_high_lat = bbox_high_lat[2] - bbox_high_lat[0]

    # same physical width in metres needs a wider degree span further from the equator
    assert lon_span_high_lat > lon_span_equator


def test_square_bbox_is_centered_on_the_given_point():
    lat, lon = -34.5, -62.4
    min_lon, min_lat, max_lon, max_lat = square_bbox_from_centroid(lat, lon, area_ha=42)

    assert (min_lon + max_lon) / 2 == pytest.approx(lon, abs=1e-9)
    assert (min_lat + max_lat) / 2 == pytest.approx(lat, abs=1e-9)
