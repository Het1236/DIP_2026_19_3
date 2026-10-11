import pytest

from preprocessing.seasonal import peak_window_bounds, select_peak_window_indices


def test_peak_window_bounds_is_between_seeding_and_harvest():
    start, end = peak_window_bounds("01.01.2023", "31.12.2023", start_frac=0.3, end_frac=0.7)
    assert start < end
    assert start.year == 2023
    # roughly 30% and 70% through a 364-day season
    assert (end - start).days == pytest.approx(364 * 0.4, abs=2)


def test_select_peak_window_indices_keeps_only_dates_in_range():
    # season: 01.01.2023 to 31.12.2023, window ~ Apr to Sep
    dates = ["20230115", "20230601", "20230801", "20231215"]
    kept = select_peak_window_indices(dates, "01.01.2023", "31.12.2023", start_frac=0.3, end_frac=0.7)
    kept_dates = [dates[i] for i in kept]
    assert "20230115" not in kept_dates  # too early
    assert "20231215" not in kept_dates  # too late
    assert "20230601" in kept_dates
    assert "20230801" in kept_dates


def test_raises_if_harvest_not_after_seeding():
    with pytest.raises(ValueError):
        peak_window_bounds("01.06.2023", "01.01.2023")
