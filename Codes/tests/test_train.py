import numpy as np
import pandas as pd
import pytest

from models.train import build_model, feature_importances, train_and_evaluate, train_test_split_fields, tune_and_evaluate


def _make_separable_df(n_per_class=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    # two features, three classes, well separated in feature space
    centers = {"corn": (0, 0), "soybean": (5, 5), "wheat": (-5, 5)}
    for crop, (cx, cy) in centers.items():
        for _ in range(n_per_class):
            rows.append({
                "crop": crop,
                "feat_a": cx + rng.normal(0, 0.3),
                "feat_b": cy + rng.normal(0, 0.3),
            })
    return pd.DataFrame(rows)


def test_build_model_known_names():
    assert build_model("random_forest") is not None
    assert build_model("svm") is not None
    with pytest.raises(ValueError):
        build_model("not_a_real_model")


def test_train_test_split_fields_is_stratified():
    df = _make_separable_df(n_per_class=20)
    train_df, test_df = train_test_split_fields(df, test_size=0.2)

    assert len(train_df) + len(test_df) == len(df)
    # every class should still appear in both halves
    assert set(train_df["crop"]) == set(df["crop"])
    assert set(test_df["crop"]) == set(df["crop"])


def test_random_forest_achieves_high_accuracy_on_well_separated_data():
    df = _make_separable_df(n_per_class=40)
    train_df, test_df = train_test_split_fields(df, test_size=0.25)

    result = train_and_evaluate(train_df, test_df, ["feat_a", "feat_b"], model_name="random_forest")
    assert result["accuracy"] > 0.9
    assert result["confusion_matrix"].shape == (3, 3)
    assert result["n_train"] == len(train_df)
    assert result["n_test"] == len(test_df)


def test_svm_achieves_high_accuracy_on_well_separated_data():
    df = _make_separable_df(n_per_class=40)
    train_df, test_df = train_test_split_fields(df, test_size=0.25)

    result = train_and_evaluate(train_df, test_df, ["feat_a", "feat_b"], model_name="svm")
    assert result["accuracy"] > 0.9


def test_tune_and_evaluate_returns_best_params_and_cv_score():
    df = _make_separable_df(n_per_class=40)
    train_df, test_df = train_test_split_fields(df, test_size=0.25)

    result = tune_and_evaluate(train_df, test_df, ["feat_a", "feat_b"], model_name="random_forest", cv=3)
    assert result["accuracy"] > 0.9
    assert "best_params" in result
    assert set(result["best_params"].keys()) == {"n_estimators", "max_depth", "min_samples_leaf"}
    assert 0.0 <= result["cv_macro_f1"] <= 1.0


def test_tune_and_evaluate_works_for_svm_pipeline():
    df = _make_separable_df(n_per_class=40)
    train_df, test_df = train_test_split_fields(df, test_size=0.25)

    result = tune_and_evaluate(train_df, test_df, ["feat_a", "feat_b"], model_name="svm", cv=3)
    assert result["accuracy"] > 0.9
    assert set(result["best_params"].keys()) == {"svc__C", "svc__gamma"}


def test_feature_importances_present_for_random_forest_not_svm():
    df = _make_separable_df(n_per_class=20)
    train_df, test_df = train_test_split_fields(df, test_size=0.25)

    rf_result = train_and_evaluate(train_df, test_df, ["feat_a", "feat_b"], model_name="random_forest")
    svm_result = train_and_evaluate(train_df, test_df, ["feat_a", "feat_b"], model_name="svm")

    rf_importances = feature_importances(rf_result)
    assert set(rf_importances.keys()) == {"feat_a", "feat_b"}
    assert feature_importances(svm_result) == {}
