"""Week 7 deliverable: does fusion actually help? Builds the fused feature
set (optical + SAR, with a correlation filter to drop redundant columns),
then re-tunes hyperparameters for optical-only, SAR-only, AND fused under
the exact same grid search, so the comparison is fair, not "tuned fusion vs
untuned baselines".

Uses the same RANDOM_SEED and split settings as week6_baseline_models.py,
so the train/test split here is identical to Week 6's, the two results
tables are directly comparable.

Run from Codes/ (no GEE needed):
    python scripts/week7_fusion_and_tuning.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from data.config import RESULTS_DIR
from evaluation.metrics import plot_confusion_matrix
from features.feature_lists import OPTICAL_FEATURES, SAR_FEATURES
from features.fusion import build_fused_feature_list, correlation_filter
from models.train import feature_importances, train_test_split_fields, tune_and_evaluate

FEATURE_TABLE_PATH = RESULTS_DIR / "features" / "real_field_features.csv"
OUT_DIR = RESULTS_DIR / "models"
MODEL_NAMES = ["random_forest", "svm"]


def main() -> None:
    df = pd.read_csv(FEATURE_TABLE_PATH)
    df = df[df["status"] == "ok"].reset_index(drop=True)
    train_df, test_df = train_test_split_fields(df, test_size=0.2)
    print(f"Train: {len(train_df)}, Test: {len(test_df)} (same split as Week 6)")

    fused_raw = build_fused_feature_list(OPTICAL_FEATURES, SAR_FEATURES)
    fused_selected = correlation_filter(train_df, fused_raw, threshold=0.95)
    dropped = sorted(set(fused_raw) - set(fused_selected))
    print(f"\nFused feature set: {len(fused_raw)} raw -> {len(fused_selected)} after correlation filter")
    if dropped:
        print(f"Dropped as redundant (correlation > 0.95 with an earlier feature): {dropped}")

    settings = [
        ("optical_only", OPTICAL_FEATURES),
        ("sar_only", SAR_FEATURES),
        ("fused", fused_selected),
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for setting_name, feature_columns in settings:
        for model_name in MODEL_NAMES:
            result = tune_and_evaluate(train_df, test_df, feature_columns, model_name=model_name, cv=5)
            print(f"\n{setting_name} / {model_name} (tuned): "
                  f"accuracy={result['accuracy']:.3f}, macro F1={result['macro_f1']:.3f}, "
                  f"cv macro F1={result['cv_macro_f1']:.3f}")
            print(f"best params: {result['best_params']}")
            print(pd.DataFrame(
                result["confusion_matrix"], index=result["labels"], columns=result["labels"],
            ))

            cm_path = OUT_DIR / f"confusion_matrix_tuned_{setting_name}_{model_name}.png"
            plot_confusion_matrix(
                result["confusion_matrix"], result["labels"],
                f"{setting_name.replace('_', ' ')}, {model_name.replace('_', ' ')} (tuned)\n"
                f"accuracy={result['accuracy']:.2f}, macro F1={result['macro_f1']:.2f}",
                cm_path,
            )

            rows.append({
                "setting": setting_name, "model": model_name,
                "n_features": len(feature_columns),
                "n_train": result["n_train"], "n_test": result["n_test"],
                "accuracy": result["accuracy"],
                "macro_f1": result["macro_f1"],
                "weighted_f1": result["weighted_f1"],
                "cv_macro_f1": result["cv_macro_f1"],
                "best_params": result["best_params"],
            })

            if model_name == "random_forest":
                importances = feature_importances(result)
                top8 = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:8]
                print(f"top 8 features: {top8}")

    comparison = pd.DataFrame(rows)
    comparison_path = OUT_DIR / "tuned_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    print(f"\nSaved {comparison_path}")
    print(comparison[["setting", "model", "n_features", "accuracy", "macro_f1", "cv_macro_f1"]].to_string(index=False))

    best_row = comparison.loc[comparison["macro_f1"].idxmax()]
    print(f"\nBest setting overall: {best_row['setting']} / {best_row['model']}, macro F1 = {best_row['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
