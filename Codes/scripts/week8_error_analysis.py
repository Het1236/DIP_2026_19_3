"""Week 8 deliverable: error analysis on the fused model, not just the
headline accuracy number. Per-crop breakdown, per-country breakdown, full
feature importance, and a cross-country generalization check (train on one
country, test on the other), which is directly analogous to the YieldSAT
paper's own Leave-One-Region-Out experiments (Learning_Notes/Dataset_Tutorial.md
mentions distribution shift as an open challenge the dataset's own authors
flagged, this is us checking whether our fused model suffers from it too).

Run from Codes/ (no GEE needed):
    python scripts/week8_error_analysis.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report

from data.config import RESULTS_DIR
from evaluation.metrics import plot_confusion_matrix
from features.feature_lists import OPTICAL_FEATURES, SAR_FEATURES
from features.fusion import build_fused_feature_list, correlation_filter
from models.train import feature_importances, train_test_split_fields, tune_and_evaluate

FEATURE_TABLE_PATH = RESULTS_DIR / "features" / "real_field_features.csv"
OUT_DIR = RESULTS_DIR / "analysis"


def main() -> None:
    df = pd.read_csv(FEATURE_TABLE_PATH)
    df = df[df["status"] == "ok"].reset_index(drop=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Same split, same feature selection as Week 6/7, so this is analyzing
    # exactly the model already reported, not a fresh one.
    train_df, test_df = train_test_split_fields(df, test_size=0.2)
    fused_raw = build_fused_feature_list(OPTICAL_FEATURES, SAR_FEATURES)
    fused_features = correlation_filter(train_df, fused_raw, threshold=0.95)

    result = tune_and_evaluate(train_df, test_df, fused_features, model_name="random_forest", cv=5)
    model, labels = result["model"], result["labels"]
    print(f"Fused Random Forest (Week 7's best setting): "
          f"accuracy={result['accuracy']:.3f}, macro F1={result['macro_f1']:.3f}")

    test_df = test_df.copy()
    test_df["pred"] = model.predict(test_df[fused_features].values)

    # --- 1. per-crop breakdown ---
    print("\n=== Per-crop breakdown ===")
    report = classification_report(test_df["crop"], test_df["pred"], labels=labels, output_dict=True, zero_division=0)
    per_crop_df = pd.DataFrame(report).T.loc[labels]
    per_crop_df.to_csv(OUT_DIR / "per_crop_metrics.csv")
    print(per_crop_df.round(3))

    # --- 2. per-country breakdown ---
    print("\n=== Per-country breakdown (same test set, split by country) ===")
    per_country_rows = []
    for country in sorted(test_df["country"].unique()):
        subset = test_df[test_df["country"] == country]
        acc = (subset["crop"] == subset["pred"]).mean()
        per_country_rows.append({"country": country, "n_test": len(subset), "accuracy": float(acc)})
    per_country_df = pd.DataFrame(per_country_rows)
    per_country_df.to_csv(OUT_DIR / "per_country_accuracy.csv", index=False)
    print(per_country_df.round(3).to_string(index=False))

    # --- 3. full feature importance, both modalities on one chart ---
    print("\n=== Feature importance (fused Random Forest) ===")
    importances = feature_importances(result)
    imp_series = pd.Series(importances).sort_values()
    colors = ["#4c72b0" if f in SAR_FEATURES else "#dd8452" for f in imp_series.index]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(imp_series.index, imp_series.values, color=colors)
    ax.set_title("Fused Random Forest: every feature's importance\n(blue = SAR, orange = optical)")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "feature_importance_full.png", dpi=130)
    plt.close(fig)

    optical_total = sum(v for k, v in importances.items() if k in OPTICAL_FEATURES)
    sar_total = sum(v for k, v in importances.items() if k in SAR_FEATURES)
    print(f"Total importance mass: optical features = {optical_total:.3f}, SAR features = {sar_total:.3f}")

    # --- 4. cross-country generalization: train on one, test on the other ---
    print("\n=== Cross-country generalization (train on one country, test on the other) ===")
    cross_rows = []
    for train_country, test_country in [("Argentina", "Brazil"), ("Brazil", "Argentina")]:
        train_c = df[df["country"] == train_country]
        test_c = df[df["country"] == test_country]
        res = tune_and_evaluate(train_c, test_c, fused_features, model_name="random_forest", cv=5)
        cross_rows.append({
            "train_country": train_country, "test_country": test_country,
            "n_train": len(train_c), "n_test": len(test_c),
            "accuracy": res["accuracy"], "macro_f1": res["macro_f1"],
        })
        plot_confusion_matrix(
            res["confusion_matrix"], res["labels"],
            f"Train on {train_country}, test on {test_country}\n"
            f"accuracy={res['accuracy']:.2f}, macro F1={res['macro_f1']:.2f}",
            OUT_DIR / f"confusion_matrix_cross_{train_country}_to_{test_country}.png",
        )
        print(f"train {train_country} -> test {test_country}: "
              f"accuracy={res['accuracy']:.3f}, macro F1={res['macro_f1']:.3f}")

    cross_df = pd.DataFrame(cross_rows)
    cross_df.to_csv(OUT_DIR / "cross_country_generalization.csv", index=False)

    same_country_f1 = result["macro_f1"]
    cross_country_avg_f1 = cross_df["macro_f1"].mean()
    print(f"\nSame-distribution (mixed-country) test macro F1: {same_country_f1:.3f}")
    print(f"Cross-country average macro F1: {cross_country_avg_f1:.3f}")
    print(f"Drop from distribution shift: {same_country_f1 - cross_country_avg_f1:.3f}")


if __name__ == "__main__":
    main()
