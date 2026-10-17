"""Train and evaluate a classical model (Random Forest or SVM) on a given
set of feature columns. Used identically for optical-only, SAR-only, and
(from Week 7) fused feature sets, so the three settings are compared fairly
under the exact same split, metrics, and model configuration.

Why Random Forest and SVM specifically, and how each one works, is
explained in Learning_Notes/Project_Explained.docx section 6.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data.config import RANDOM_SEED
from evaluation.metrics import evaluate_predictions

MODEL_BUILDERS = {
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_SEED, class_weight="balanced",
    ),
    "svm": lambda: make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=1.0, class_weight="balanced", random_state=RANDOM_SEED),
    ),
}

# Kept modest on purpose: 216 training rows is not a lot, a huge grid would
# mostly be tuning noise, not signal. Random Forest keys are unprefixed
# (bare estimator), SVM keys need the "svc__" pipeline prefix since it's
# wrapped in a StandardScaler pipeline (see MODEL_BUILDERS above).
PARAM_GRIDS = {
    "random_forest": {
        "n_estimators": [100, 200, 400],
        "max_depth": [None, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4],
    },
    "svm": {
        "svc__C": [0.1, 1, 10, 100],
        "svc__gamma": ["scale", "auto", 0.01, 0.1],
    },
}


def build_model(model_name: str):
    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"unknown model {model_name!r}, expected one of {list(MODEL_BUILDERS)}")
    return MODEL_BUILDERS[model_name]()


def train_test_split_fields(
    df: pd.DataFrame,
    label_column: str = "crop",
    test_size: float = 0.2,
    seed: int = RANDOM_SEED,
):
    """Stratified split so each crop keeps roughly the same proportion in
    train and test, not just a random slice that could leave a class
    under-represented in one side by chance.
    """
    return train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df[label_column],
    )


def train_and_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    label_column: str = "crop",
    model_name: str = "random_forest",
) -> dict:
    labels = sorted(train_df[label_column].unique())

    X_train, y_train = train_df[feature_columns].values, train_df[label_column].values
    X_test, y_test = test_df[feature_columns].values, test_df[label_column].values

    model = build_model(model_name)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    result = evaluate_predictions(y_test, y_pred, labels)
    result["labels"] = labels
    result["model"] = model
    result["model_name"] = model_name
    result["n_train"] = len(train_df)
    result["n_test"] = len(test_df)
    result["feature_columns"] = feature_columns
    return result


def tune_and_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    label_column: str = "crop",
    model_name: str = "random_forest",
    cv: int = 5,
    seed: int = RANDOM_SEED,
) -> dict:
    """Same as train_and_evaluate, but picks hyperparameters via grid
    search with stratified cross-validation on the training split only,
    scored by macro F1 (fair across imbalanced classes, matches how we
    report results). Test data is never touched until the very end, one
    prediction pass with the already-chosen best model.
    """
    labels = sorted(train_df[label_column].unique())
    X_train, y_train = train_df[feature_columns].values, train_df[label_column].values
    X_test, y_test = test_df[feature_columns].values, test_df[label_column].values

    base_model = build_model(model_name)
    param_grid = PARAM_GRIDS[model_name]
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)

    search = GridSearchCV(base_model, param_grid, cv=cv_splitter, scoring="f1_macro", n_jobs=-1)
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)

    result = evaluate_predictions(y_test, y_pred, labels)
    result["labels"] = labels
    result["model"] = best_model
    result["model_name"] = model_name
    result["n_train"] = len(train_df)
    result["n_test"] = len(test_df)
    result["feature_columns"] = feature_columns
    result["best_params"] = search.best_params_
    result["cv_macro_f1"] = float(search.best_score_)
    return result


def feature_importances(result: dict) -> dict:
    """Only meaningful for Random Forest, SVM with an rbf kernel has no
    single per-feature importance score the way a tree ensemble does.
    """
    model = result["model"]
    if not hasattr(model, "feature_importances_"):
        return {}
    return dict(zip(result["feature_columns"], model.feature_importances_.tolist()))
