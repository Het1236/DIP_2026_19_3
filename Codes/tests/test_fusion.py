import pandas as pd
import pytest

from features.fusion import build_fused_feature_list, correlation_filter


def test_build_fused_feature_list_concatenates_in_order():
    optical = ["ndvi", "evi"]
    sar = ["vv", "vh"]
    assert build_fused_feature_list(optical, sar) == ["ndvi", "evi", "vv", "vh"]


def test_correlation_filter_drops_the_later_of_a_perfectly_correlated_pair():
    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [2, 4, 6, 8, 10],  # perfectly correlated with a
        "c": [5, 1, 4, 2, 3],   # unrelated
    })
    kept = correlation_filter(df, ["a", "b", "c"], threshold=0.95)
    assert kept == ["a", "c"]


def test_correlation_filter_keeps_uncorrelated_features():
    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [5, 1, 4, 2, 3],
        "c": [3, 3, 1, 5, 2],
    })
    kept = correlation_filter(df, ["a", "b", "c"], threshold=0.95)
    assert set(kept) == {"a", "b", "c"}


def test_correlation_filter_handles_chain_of_three_correlated_features():
    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [1.01, 2.01, 3.01, 4.01, 5.01],
        "c": [1.02, 2.02, 3.02, 4.02, 5.02],
    })
    kept = correlation_filter(df, ["a", "b", "c"], threshold=0.95)
    assert kept == ["a"]
