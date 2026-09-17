"""Week 2 wrap-up: prove GEE access works against real Sentinel-1 data,
not just a trivial API call.

Filters the Sentinel-1 GRD collection to a small area of interest in
Argentina over a short date range and prints how many scenes are
available. Does not download or export anything, this is a connectivity
and collection-access check only. The real per-field pull comes in Week 4.

Run from Codes/ with GEE_PROJECT_ID set:
    python -m src.data.gee_smoke_test
"""

import sys

from .config import GEE_PROJECT_ID

# a small bounding box inside Argentina's Pampas crop belt, picked only
# because it is a real agricultural area, not tied to any specific
# YieldSAT field yet (we don't have real field boundaries until access
# is confirmed)
AOI_BBOX = [-62.5, -34.5, -62.3, -34.3]
START_DATE = "2024-01-01"
END_DATE = "2024-02-01"


def run() -> bool:
    import ee

    if not GEE_PROJECT_ID:
        print("GEE_PROJECT_ID is not set.")
        return False

    try:
        ee.Initialize(project=GEE_PROJECT_ID)
        aoi = ee.Geometry.Rectangle(AOI_BBOX)
        collection = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(aoi)
            .filterDate(START_DATE, END_DATE)
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        )
        count = collection.size().getInfo()
    except Exception as exc:  # noqa: BLE001
        print(f"GEE smoke test FAILED: {exc}")
        return False

    print(f"Sentinel-1 GRD scenes found over the sample AOI ({START_DATE} to {END_DATE}): {count}")
    if count == 0:
        print("Query worked but returned 0 scenes, try widening the date range.")
    else:
        print("GEE data access confirmed working.")
    return True


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
