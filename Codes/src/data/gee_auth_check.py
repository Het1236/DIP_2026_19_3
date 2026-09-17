"""One-time Google Earth Engine setup check.

GEE authentication needs an interactive browser login tied to a personal
Google account, so it cannot be done for you from this environment. Each
team member who will pull Sentinel-1 data (Week 4 onward) should run this
once on their own machine:

    1. pip install -r requirements.txt   (already includes earthengine-api)
    2. earthengine authenticate          (opens a browser, one-time)
    3. set the GEE_PROJECT_ID environment variable to your GEE Cloud project id
    4. python src/data/gee_auth_check.py

If step 4 prints "Earth Engine is ready.", you're set up correctly and
the actual Sentinel-1 pull script (added in Week 4) will work for you.
"""

import sys

from .config import GEE_PROJECT_ID


def check() -> bool:
    import ee

    if not GEE_PROJECT_ID:
        print("GEE_PROJECT_ID is not set. Set it to your Earth Engine Cloud project id.")
        return False

    try:
        ee.Initialize(project=GEE_PROJECT_ID)
        # a trivial call that only succeeds if auth + project are both valid
        ee.Number(1).getInfo()
    except Exception as exc:  # noqa: BLE001 - we want to surface any auth error as-is
        print(f"Earth Engine is NOT ready: {exc}")
        return False

    print("Earth Engine is ready.")
    return True


if __name__ == "__main__":
    sys.exit(0 if check() else 1)
