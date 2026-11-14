"""Week 10 check: does the result survive when fields of one farm are kept apart?

The Week 9 nested cross validation splits fields at random. Fields of one farm
can then be in both the training and the test part, and they look alike, so the
scores could be a little high. Here the outer folds never split a farm. Everything
else is the same as week9_finalize.run_nested_cv: the correlation filter and the
grid search use the training folds only, the same grids, the same seed.

The farm of a field is read from its field_id, for example
"Argentina_DUP1_farm14_field117_corn_2023" belongs to farm "Argentina_DUP1_farm14".
So this script needs only the committed feature table, not the YieldSAT download.

Run from Codes/ (no GEE needed):
    python scripts/farm_grouped_cv.py
    python scripts/farm_grouped_cv.py --n-jobs 1      # if memory is low
"""

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import week9_finalize as w9
from data.config import RANDOM_SEED, RESULTS_DIR
from models.train import tune_and_evaluate

FEATURE_TABLE_PATH = RESULTS_DIR / "features" / "real_field_features.csv"
OUT_PATH = RESULTS_DIR / "final" / "farm_grouped_cv.csv"
FARM_PATTERN = re.compile(r"^(?P<farm>[A-Za-z]+_DUP\d+_farm\d+)_field\d+_")


def farm_of(field_id: str) -> str:
    match = FARM_PATTERN.match(field_id)
    if not match:
        raise ValueError(f"cannot read the farm from field_id {field_id!r}")
    return match.group("farm")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=2, help="worker processes for grid search (default 2)")
    args = parser.parse_args()

    df = pd.read_csv(FEATURE_TABLE_PATH)
    df = df[df["status"] == "ok"].reset_index(drop=True)
    groups = df["field_id"].map(farm_of).values
    sizes = pd.Series(groups).value_counts()
    print(f"{len(df)} fields from {len(sizes)} farms "
          f"(fields per farm: median {int(sizes.median())}, largest {int(sizes.max())})")

    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    scores = {(s, m): {"accuracy": [], "macro_f1": []} for s in w9.SETTINGS for m in w9.MODEL_NAMES}
    start = time.time()
    for fold, (tr_idx, te_idx) in enumerate(outer.split(df, df["crop"], groups), start=1):
        tr, te = df.iloc[tr_idx], df.iloc[te_idx]
        shared = set(groups[tr_idx]) & set(groups[te_idx])
        assert not shared, f"farm(s) on both sides of fold {fold}: {sorted(shared)[:3]}"
        for setting in w9.SETTINGS:
            features = w9.feature_columns_for(setting, tr)
            for model_name in w9.MODEL_NAMES:
                res = tune_and_evaluate(tr, te, features, model_name=model_name, cv=5, n_jobs=args.n_jobs)
                scores[(setting, model_name)]["accuracy"].append(res["accuracy"])
                scores[(setting, model_name)]["macro_f1"].append(res["macro_f1"])
        print(f"  fold {fold}/5 done: {len(te)} test fields from {len(set(groups[te_idx]))} farms, "
              f"{time.time() - start:.0f}s elapsed", flush=True)

    rows = []
    for (setting, model_name), sc in scores.items():
        rows.append({
            "setting": setting, "model": model_name, "n_folds": 5, "n_farms": len(sizes),
            "accuracy_mean": np.mean(sc["accuracy"]), "accuracy_std": np.std(sc["accuracy"], ddof=1),
            "macro_f1_mean": np.mean(sc["macro_f1"]), "macro_f1_std": np.std(sc["macro_f1"], ddof=1),
        })
    table = pd.DataFrame(rows)
    table.to_csv(OUT_PATH, index=False)

    random_cv = pd.read_csv(RESULTS_DIR / "final" / "final_cv_results.csv")
    merged = random_cv.merge(table, on=["setting", "model"], suffixes=("_random", "_by_farm"))
    merged["change"] = merged["macro_f1_mean_by_farm"] - merged["macro_f1_mean_random"]
    print("\nMacro F1, nested 5-fold: random split vs split by farm")
    print(merged[["setting", "model", "macro_f1_mean_random", "macro_f1_mean_by_farm", "change"]]
          .round(3).to_string(index=False))
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
