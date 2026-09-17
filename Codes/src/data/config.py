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

COUNTRIES = ["argentina", "brazil"]
CROP_TYPES = ["corn", "soybean", "wheat"]

RANDOM_SEED = 42

# Set your own Google Earth Engine Cloud Project ID as an environment
# variable before running any GEE script:
#   export GEE_PROJECT_ID=your-project-id      (bash)
#   $env:GEE_PROJECT_ID = "your-project-id"    (PowerShell)
GEE_PROJECT_ID = os.environ.get("GEE_PROJECT_ID")
