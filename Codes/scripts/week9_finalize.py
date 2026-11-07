"""Week 9 deliverable: lock the final experiment. Same data, same split, same
feature selection and same tuning as Weeks 6 to 8, so nothing about the
method changes here. This script only re-runs it cleanly end to end and saves
everything the end-sem report needs in one place, Results/final/:

  final_results.csv          test-set results, 3 settings x 2 models
  final_cv_results.csv       5-fold nested cross-validation over all fields
  models/*.joblib            the final trained models
  final_comparison_bars.png  accuracy and macro F1 by modality
  final_confusion_matrices.png  best model of each setting
  final_feature_importance.png  fused Random Forest, SAR vs optical

The nested cross-validation is the one addition. With only 54 test fields,
one field moves accuracy by about 2 percent, so a single split is noisy.
Nested CV tunes inside each training fold and tests on the held-out fold,
so every one of the 270 fields is tested once and the test data never
influences tuning or feature selection.

Run from Codes/ (no GEE needed):
    python scripts/week9_finalize.py
    python scripts/week9_finalize.py --skip-nested     # faster, test-set part only
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from data.config import RANDOM_SEED, RESULTS_DIR
from features.feature_lists import OPTICAL_FEATURES, SAR_FEATURES
from features.fusion import build_fused_feature_list, correlation_filter
from models.train import feature_importances, train_test_split_fields, tune_and_evaluate

FEATURE_TABLE_PATH = RESULTS_DIR / "features" / "real_field_features.csv"
OUT_DIR = RESULTS_DIR / "final"
MODEL_DIR = OUT_DIR / "models"
SETTINGS = ["optical_only", "sar_only", "fused"]
MODEL_NAMES = ["random_forest", "svm"]
COLORS = {"random_forest": "#4C72B0", "svm": "#DD8452"}
SETTING_LABELS = {"optical_only": "Optical only", "sar_only": "SAR only", "fused": "Fused"}
MODEL_LABELS = {"random_forest": "Random Forest", "svm": "SVM"}


def feature_columns_for(setting: str, train_df: pd.DataFrame) -> list[str]:
    """Feature list for a setting. The correlation filter for the fused set is
    fitted on the training rows passed in, never on test rows."""
    if setting == "optical_only":
        return list(OPTICAL_FEATURES)
    if setting == "sar_only":
        return list(SAR_FEATURES)
    raw = build_fused_feature_list(OPTICAL_FEATURES, SAR_FEATURES)
    return correlation_filter(train_df, raw, threshold=0.95)


def run_test_split(df: pd.DataFrame, n_jobs: int) -> tuple[pd.DataFrame, dict]:
    train_df, test_df = train_test_split_fields(df, test_size=0.2)
    print(f"Train: {len(train_df)}, Test: {len(test_df)} (same split as Weeks 6 to 8)")

    rows, results = [], {}
    for setting in SETTINGS:
        features = feature_columns_for(setting, train_df)
        for model_name in MODEL_NAMES:
            res = tune_and_evaluate(train_df, test_df, features, model_name=model_name, cv=5, n_jobs=n_jobs)
            results[(setting, model_name)] = res
            print(f"{setting:13s} {model_name:14s} acc={res['accuracy']:.3f} "
                  f"macroF1={res['macro_f1']:.3f} cvF1={res['cv_macro_f1']:.3f}")
            rows.append({
                "setting": setting, "model": model_name, "n_features": len(features),
                "n_train": res["n_train"], "n_test": res["n_test"],
                "accuracy": res["accuracy"], "macro_f1": res["macro_f1"],
                "weighted_f1": res["weighted_f1"], "cv_macro_f1": res["cv_macro_f1"],
                "best_params": json.dumps(res["best_params"]),
            })
            joblib.dump(
                {"model": res["model"], "features": features, "labels": res["labels"],
                 "best_params": res["best_params"]},
                MODEL_DIR / f"{setting}_{model_name}.joblib",
            )
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "final_results.csv", index=False)
    return table, results


def run_nested_cv(df: pd.DataFrame, n_jobs: int, n_splits: int = 5) -> pd.DataFrame:
    """Outer 5-fold stratified CV over all fields. Inside each outer training
    fold: correlation filter (fused only) and grid search, both on that
    fold's training rows alone."""
    outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    scores = {(s, m): {"accuracy": [], "macro_f1": []} for s in SETTINGS for m in MODEL_NAMES}
    start = time.time()
    for fold, (tr_idx, te_idx) in enumerate(outer.split(df, df["crop"]), start=1):
        tr, te = df.iloc[tr_idx], df.iloc[te_idx]
        for setting in SETTINGS:
            features = feature_columns_for(setting, tr)
            for model_name in MODEL_NAMES:
                res = tune_and_evaluate(tr, te, features, model_name=model_name, cv=5, n_jobs=n_jobs)
                scores[(setting, model_name)]["accuracy"].append(res["accuracy"])
                scores[(setting, model_name)]["macro_f1"].append(res["macro_f1"])
        print(f"  outer fold {fold}/{n_splits} done, {time.time() - start:.0f}s elapsed")

    rows = []
    for (setting, model_name), sc in scores.items():
        rows.append({
            "setting": setting, "model": model_name, "n_folds": n_splits,
            "accuracy_mean": np.mean(sc["accuracy"]), "accuracy_std": np.std(sc["accuracy"], ddof=1),
            "macro_f1_mean": np.mean(sc["macro_f1"]), "macro_f1_std": np.std(sc["macro_f1"], ddof=1),
        })
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "final_cv_results.csv", index=False)
    return table


def plot_comparison_bars(test_table: pd.DataFrame, cv_table: pd.DataFrame | None) -> None:
    n_panels = 2 if cv_table is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 4.8), squeeze=False)
    x = np.arange(len(SETTINGS))
    width = 0.36

    def draw(ax, table, mean_col, std_col, title):
        for i, model in enumerate(MODEL_NAMES):
            vals = [table[(table.setting == s) & (table.model == model)][mean_col].iloc[0] for s in SETTINGS]
            errs = None
            if std_col:
                errs = [table[(table.setting == s) & (table.model == model)][std_col].iloc[0] for s in SETTINGS]
            bars = ax.bar(x + (i - 0.5) * width, vals, width, yerr=errs, capsize=3,
                          label=MODEL_LABELS[model], color=COLORS[model])
            for k, (b, v) in enumerate(zip(bars, vals)):
                top = v + (errs[k] if errs else 0) + 0.015
                ax.text(b.get_x() + b.get_width() / 2, top, f"{v:.2f}", ha="center", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([SETTING_LABELS[s] for s in SETTINGS])
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Macro F1")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper left")

    n_test = int(test_table["n_test"].iloc[0])
    draw(axes[0, 0], test_table, "macro_f1", None, f"Held-out test set ({n_test} fields)")
    if cv_table is not None:
        draw(axes[0, 1], cv_table, "macro_f1_mean", "macro_f1_std", "Nested 5-fold CV (all fields, mean and std)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "final_comparison_bars.png", dpi=150)
    plt.close(fig)


def plot_confusion_panels(test_table: pd.DataFrame, results: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, setting in zip(axes, SETTINGS):
        best_model = test_table[test_table.setting == setting].sort_values("macro_f1", ascending=False).iloc[0]["model"]
        res = results[(setting, best_model)]
        cm, labels = res["confusion_matrix"], res["labels"]
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(f"{setting.replace('_', ' ').capitalize()}, {best_model.replace('_', ' ')}\n"
                     f"accuracy={res['accuracy']:.2f}, macro F1={res['macro_f1']:.2f}")
        threshold = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > threshold else "black")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "final_confusion_matrices.png", dpi=150)
    plt.close(fig)


def plot_feature_importance(results: dict) -> None:
    res = results[("fused", "random_forest")]
    importances = pd.Series(feature_importances(res)).sort_values()
    colors = ["#4C72B0" if f in SAR_FEATURES else "#DD8452" for f in importances.index]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(importances.index, importances.values, color=colors)
    ax.set_title("Final fused Random Forest: feature importance\n(blue = SAR, orange = optical)")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "final_feature_importance.png", dpi=150)
    plt.close(fig)
    sar = sum(v for k, v in importances.items() if k in SAR_FEATURES)
    opt = sum(v for k, v in importances.items() if k in OPTICAL_FEATURES)
    print(f"Importance share: SAR = {sar:.2f}, optical = {opt:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-nested", action="store_true", help="skip the nested cross-validation")
    parser.add_argument("--n-jobs", type=int, default=2, help="worker processes for grid search (default 2)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FEATURE_TABLE_PATH)
    df = df[df["status"] == "ok"].reset_index(drop=True)
    print(f"Loaded {len(df)} real fields: {df['crop'].value_counts().to_dict()}")

    print("\n=== Final run on the held-out test split ===")
    test_table, results = run_test_split(df, args.n_jobs)

    cv_table = None
    saved_cv = OUT_DIR / "final_cv_results.csv"
    if args.skip_nested and saved_cv.exists():
        cv_table = pd.read_csv(saved_cv)
        print(f"Using saved nested CV results from {saved_cv.name}")
    if not args.skip_nested:
        print("\n=== Nested 5-fold cross-validation over all fields (this takes a while) ===")
        cv_table = run_nested_cv(df, args.n_jobs)
        print(cv_table.round(3).to_string(index=False))

    plot_comparison_bars(test_table, cv_table)
    plot_confusion_panels(test_table, results)
    plot_feature_importance(results)
    print(f"\nSaved everything to {OUT_DIR}")


if __name__ == "__main__":
    main()
