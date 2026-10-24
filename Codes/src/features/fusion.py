"""Feature fusion: combine optical and SAR feature columns into one vector.

This is early fusion, also called feature-level fusion, explained in
Learning_Notes/Project_Explained.docx section 5: concatenate the two
feature sets before the model ever sees them. It's the simplest kind of
fusion and the right choice for a classical, explainable pipeline.
"""

from __future__ import annotations

import pandas as pd


def correlation_filter(df: pd.DataFrame, feature_columns: list[str], threshold: float = 0.95) -> list[str]:
    """Drop one column from every pair correlated above `threshold`
    (Pearson, absolute value), keeping whichever one appears first in
    feature_columns. Only look at df's own rows when calling this, fit
    on the training split alone, never on test data, correlations
    computed including test rows would leak information about the test
    set into a decision that shapes the model.

    Why this matters before fusing: two near-duplicate columns (for
    example mean_vv_db and vv_glcm_contrast if they happened to move
    together for this data) don't add real information, just noise and
    extra dimensions, exactly the curse-of-dimensionality risk described
    in Learning_Notes section 5.
    """
    corr = df[feature_columns].corr().abs()
    to_drop = set()
    for i, col_i in enumerate(feature_columns):
        if col_i in to_drop:
            continue
        for col_j in feature_columns[i + 1:]:
            if col_j in to_drop:
                continue
            if corr.loc[col_i, col_j] > threshold:
                to_drop.add(col_j)
    return [c for c in feature_columns if c not in to_drop]


def build_fused_feature_list(optical_features: list[str], sar_features: list[str]) -> list[str]:
    """x_fused = [x_optical || x_sar], the concatenation from
    Learning_Notes' eq_fusion. Kept as a tiny named function rather than
    just writing `optical + sar` inline everywhere, so every script fuses
    the same way and it's obvious at a glance what "fused" means here.
    """
    return list(optical_features) + list(sar_features)
