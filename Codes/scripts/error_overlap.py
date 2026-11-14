"""Week 10 check: do the optical-only and SAR-only models make the same mistakes?

If both sensors failed on the same fields, fusion could not help. If they fail on
different fields, fusion has something to repair. This script counts it on the
54 test fields, using the saved best model of each feature set (Results/final/models).

Run from Codes/ (no GEE needed, needs python scripts/week9_finalize.py to have run once):
    python scripts/error_overlap.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix

from data.config import RESULTS_DIR
from models.train import train_test_split_fields

FEATURE_TABLE_PATH = RESULTS_DIR / "features" / "real_field_features.csv"
FINAL_DIR = RESULTS_DIR / "final"
OUT_PATH = FINAL_DIR / "error_overlap.txt"


def main() -> None:
    df = pd.read_csv(FEATURE_TABLE_PATH)
    df = df[df["status"] == "ok"].reset_index(drop=True)
    _train, test = train_test_split_fields(df, test_size=0.2)  # same split as Weeks 6 to 9

    results = pd.read_csv(FINAL_DIR / "final_results.csv")
    lines, wrong = [], {}
    for setting in ["optical_only", "sar_only", "fused"]:
        best = results[results.setting == setting].sort_values("macro_f1", ascending=False).iloc[0]["model"]
        saved = joblib.load(FINAL_DIR / "models" / f"{setting}_{best}.joblib")
        pred = saved["model"].predict(test[saved["features"]].values)
        wrong[setting] = pred != test["crop"].values
        cm = confusion_matrix(test["crop"], pred, labels=saved["labels"])
        lines += [f"{setting} ({best}): {int(wrong[setting].sum())} of {len(test)} wrong",
                  f"  rows = true {saved['labels']}, columns = predicted", *[f"  {row.tolist()}" for row in cm], ""]

    o, s, f = wrong["optical_only"], wrong["sar_only"], wrong["fused"]
    lines += [
        f"Wrong in both optical-only and SAR-only : {int((o & s).sum())}",
        f"Wrong in exactly one of the two         : {int((o ^ s).sum())}",
        f"  of these, fused model gets right      : {int(((o ^ s) & ~f).sum())}",
        f"Wrong in both, fused model still wrong  : {int((o & s & f).sum())}",
        f"Right in both, fused model wrong        : {int((~o & ~s & f).sum())}",
    ]
    text = "\n".join(lines)
    OUT_PATH.write_text(text)
    print(text)
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
