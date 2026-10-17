"""Week 6 deliverable: baseline classical models, optical-only and SAR-only,
on the real 270-field feature table. Fusion and hyperparameter tuning are
Week 7, this script deliberately keeps optical and SAR separate to answer
"how good is each modality alone" before combining them.

Run from Codes/ (no GEE needed, everything here reads the already-built
feature table):
    python scripts/week6_baseline_models.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from data.config import RESULTS_DIR
from evaluation.metrics import plot_confusion_matrix
from features.feature_lists import OPTICAL_FEATURES, SAR_FEATURES
from models.train import feature_importances, train_and_evaluate, train_test_split_fields

FEATURE_TABLE_PATH = RESULTS_DIR / "features" / "real_field_features.csv"
OUT_DIR = RESULTS_DIR / "models"
MODEL_NAMES = ["random_forest", "svm"]


def main() -> None:
    df = pd.read_csv(FEATURE_TABLE_PATH)
    df = df[df["status"] == "ok"].reset_index(drop=True)
    print(f"Loaded {len(df)} real fields with usable features.")
    print(df["crop"].value_counts().to_string())

    train_df, test_df = train_test_split_fields(df, test_size=0.2)
    print(f"\nTrain: {len(train_df)}, Test: {len(test_df)} (stratified by crop)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for setting_name, feature_columns in [("optical_only", OPTICAL_FEATURES), ("sar_only", SAR_FEATURES)]:
        for model_name in MODEL_NAMES:
            result = train_and_evaluate(train_df, test_df, feature_columns, model_name=model_name)
            print(f"\n{setting_name} / {model_name}: "
                  f"accuracy={result['accuracy']:.3f}, macro F1={result['macro_f1']:.3f}")
            print(pd.DataFrame(
                result["confusion_matrix"], index=result["labels"], columns=result["labels"],
            ))

            cm_path = OUT_DIR / f"confusion_matrix_{setting_name}_{model_name}.png"
            plot_confusion_matrix(
                result["confusion_matrix"], result["labels"],
                f"{setting_name.replace('_', ' ')}, {model_name.replace('_', ' ')}\n"
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
            })

            if model_name == "random_forest":
                importances = feature_importances(result)
                top5 = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:5]
                print(f"top 5 features: {top5}")

    comparison = pd.DataFrame(rows)
    comparison_path = OUT_DIR / "baseline_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    print(f"\nSaved {comparison_path}")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
