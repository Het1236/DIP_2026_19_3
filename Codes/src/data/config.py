"""Central config for paths, countries, and reproducibility settings.

Kept as plain module-level constants so every script imports the same
values instead of hardcoding paths or the random seed in multiple places.
"""

import os
from pathlib import Path

CODES_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = CODES_ROOT.parent
RESULTS_DIR = PROJECT_ROOT / "Results"
DEV_CACHE_DIR = CODES_ROOT / ".dev_cache"

# Real YieldSAT download, local only, never committed (see .gitignore).
# Layout on disk:
#   dataset/<Country>/<Country>_raw/<field_id>/...
#   dataset/<Country>/<Country>_preprocessed/merge_s2-soil-dem-weather-coords.nc
DATASET_ROOT = PROJECT_ROOT / "dataset"

# Matches the real folder names exactly (capitalized), not YieldSAT's own
# lowercase convention used elsewhere in the docs.
COUNTRIES = ["Argentina", "Brazil"]
CROP_TYPES = ["corn", "soybean", "wheat"]  # no rapeseed in our two countries

RANDOM_SEED = 42

# Set your own Google Earth Engine Cloud Project ID as an environment
# variable before running any GEE script:
#   export GEE_PROJECT_ID=your-project-id      (bash)
#   $env:GEE_PROJECT_ID = "your-project-id"    (PowerShell)
GEE_PROJECT_ID = os.environ.get("GEE_PROJECT_ID")
