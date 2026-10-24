"""Bar chart of macro F1 for optical-only, SAR-only and fused features,
read from the Week 7 results table (Results/models/tuned_comparison.csv).

Run from Codes/ after week7_fusion_and_tuning.py:
    python scripts/plot_week7_comparison.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.config import RESULTS_DIR

MODELS_DIR = RESULTS_DIR / "models"
SETTINGS = ["optical_only", "sar_only", "fused"]
MODELS = ["random_forest", "svm"]
COLORS = {"random_forest": "#4C72B0", "svm": "#DD8452"}


def main() -> None:
    df = pd.read_csv(MODELS_DIR / "tuned_comparison.csv")
    n_test = int(df["n_test"].iloc[0])
    x = np.arange(len(SETTINGS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for i, model in enumerate(MODELS):
        values = [df[(df.setting == s) & (df.model == model)]["macro_f1"].iloc[0] for s in SETTINGS]
        bars = ax.bar(x + (i - 0.5) * width, values, width, label=model.replace("_", " ").title(), color=COLORS[model])
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", " ").capitalize() for s in SETTINGS])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Macro F1 (test set)")
    ax.set_title(f"Week 7: does fusion help? Macro F1 by modality and model\n(tuned, same {n_test}-field test set)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    out = MODELS_DIR / "week7_fusion_comparison.png"
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
