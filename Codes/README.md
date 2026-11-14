# Codes

Source code for DIP_2026_19_3: SAR and optical feature fusion for crop classification
with YieldSAT (Argentina and Brazil). See the root [README](../README.md) for the project
overview and the main results.

## Setup

Python 3.11 or newer is needed. The results were produced with Python 3.14.2 and the exact
package versions in `requirements.txt`. The saved models in `Results/final/models/` only load
with the same scikit-learn version (1.9.0).

```bash
cd Codes
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Check the install by running the tests. They need no real data. Tests that need the real
YieldSAT download skip themselves when `dataset/` is missing.

```bash
python -m pytest -q
```

## What you can run without the dataset

The feature table `Results/features/real_field_features.csv` is committed (270 real fields).
Everything from the baseline models to the final results reads this table. It needs no
YieldSAT download and no Google Earth Engine.

Run from `Codes/`, in this order:

| Step | Command | What it does | Output |
|---|---|---|---|
| 1 | `python scripts/week6_baseline_models.py` | Optical-only and SAR-only baselines, Random Forest and SVM | `Results/models/baseline_comparison.csv` and confusion matrices |
| 2 | `python scripts/week7_fusion_and_tuning.py` | Fused features, correlation filter, tuned models for all three settings | `Results/models/tuned_comparison.csv` and tuned confusion matrices |
| 3 | `python scripts/plot_week7_comparison.py` | Bar chart of the Week 7 comparison | `Results/models/week7_fusion_comparison.png` |
| 4 | `python scripts/week8_error_analysis.py` | Per-crop and per-country results, feature importance, cross-country test | `Results/analysis/` |
| 5 | `python scripts/week9_finalize.py` | Final run, saved models, nested 5-fold cross validation, final figures | `Results/final/` |
| 6 | `python scripts/error_overlap.py` | Do the optical-only and SAR-only models make the same mistakes? | `Results/final/error_overlap.txt` |
| 7 | `python scripts/farm_grouped_cv.py` | Nested cross validation with whole farms held out (about 20 minutes) | `Results/final/farm_grouped_cv.csv` |

Step 5 takes about 20 minutes because of the nested cross validation. For a quick run
that only repeats the test-split part, use `python scripts/week9_finalize.py --skip-nested`.
All random seeds are 42, so the numbers repeat exactly.

If your machine has little free memory, run the grid searches in one process. For
`week9_finalize.py` use `--n-jobs 1`. For the other scripts set the environment variable
`LOKY_MAX_CPU_COUNT=1` first.

## Building the feature table from the real data

This part needs the YieldSAT download and Google Earth Engine. It was run once to produce
the committed table.

1. Place the YieldSAT data in `dataset/` at the project root. The expected layout is
   `dataset/<Country>/<Country>_raw/<field_id>/...`. The folder is not committed
   (about 15 GB).
2. Authenticate once and set the shared Cloud project (every team member does this with
   their own Google account):

```bash
earthengine authenticate
# PowerShell
$env:GEE_PROJECT_ID = "fusioncrop"
# bash
export GEE_PROJECT_ID=fusioncrop
python -m src.data.gee_auth_check
```

It should print `Earth Engine is ready.`

3. Build the table. It is resumable: after an interruption, run the same command again.

```bash
python -m scripts.build_feature_table
```

Each field needs its own Sentinel-1 download, so this takes a few hours for 300 fields.
`scripts/add_glcm_to_existing_table.py` was a one-time step that added the texture columns to
an earlier table using local caches. It is not needed to reproduce the results.

## Structure

```
Codes/
  src/
    data/            config, YieldSAT loader, field sampling, Google Earth Engine download
    preprocessing/   cloud mask and composite, season window, Lee filter, grid alignment
    features/        vegetation indices, GLCM texture, feature lists, fusion
    models/          Random Forest and SVM training and tuning
    evaluation/      accuracy, F1-score, confusion matrix
  scripts/           runnable entry points, one file does one thing
  tests/             pytest tests
```

## Earlier demos

These were used to check each stage while the pipeline was built. They are not part of the
final results.

- `scripts/week2_demo.py`: synthetic fields, NDVI and a figure (needs no data)
- `scripts/week3_demo.py`: cloud mask and indices on synthetic multi-date data
- `scripts/week4_demo.py`: Lee filter on a real Sentinel-1 composite (needs Google Earth Engine)
- `scripts/real_field_demo.py`: the full pipeline on one real field per crop (needs the dataset and Google Earth Engine)
