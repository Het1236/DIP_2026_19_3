"""Restrict a field's Sentinel-2 time series to a peak-growing-season
sub-window instead of using the entire seeding-to-harvest range.

Averaging across the whole season mixes early sparse-canopy growth and
post-harvest stubble with peak-season canopy, diluting the very signal a
classifier needs. We don't have per-crop phenology tables, so this uses a
simple, deliberately generic rule instead: keep only the middle portion of
the season, between start_frac and end_frac of the way from seeding to
harvest. This is a simplifying assumption, not an agronomic model, and
should be revisited if it turns out to hurt classification accuracy later.
"""

from __future__ import annotations

import datetime as dt


def parse_yieldsat_date(date_str: str) -> dt.date:
    """'04.11.2023' -> date(2023, 11, 4)"""
    return dt.datetime.strptime(date_str, "%d.%m.%Y").date()


def parse_acquisition_date(date_str: str) -> dt.date:
    """'20231006' (from an S2_L2A_<date>.tif filename) -> date(2023, 10, 6)"""
    return dt.datetime.strptime(date_str, "%Y%m%d").date()


def peak_window_bounds(seeding_date: str, harvesting_date: str, start_frac: float = 0.3, end_frac: float = 0.7):
    seeding = parse_yieldsat_date(seeding_date)
    harvesting = parse_yieldsat_date(harvesting_date)
    total_days = (harvesting - seeding).days
    if total_days <= 0:
        raise ValueError(f"harvesting_date {harvesting_date} is not after seeding_date {seeding_date}")

    window_start = seeding + dt.timedelta(days=int(total_days * start_frac))
    window_end = seeding + dt.timedelta(days=int(total_days * end_frac))
    return window_start, window_end


def select_peak_window_indices(
    acquisition_dates: list[str],
    seeding_date: str,
    harvesting_date: str,
    start_frac: float = 0.3,
    end_frac: float = 0.7,
) -> list[int]:
    """Indices into acquisition_dates (S2_L2A_<date>.tif style strings)
    that fall inside the peak-season window.
    """
    window_start, window_end = peak_window_bounds(seeding_date, harvesting_date, start_frac, end_frac)
    kept = []
    for i, date_str in enumerate(acquisition_dates):
        acq_date = parse_acquisition_date(date_str)
        if window_start <= acq_date <= window_end:
            kept.append(i)
    return kept
