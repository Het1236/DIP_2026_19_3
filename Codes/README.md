# Codes

Source code for DIP_2026_19_3 (SAR–Optical Feature Fusion). See the root
[README](../README.md) for the project overview and [PROJECT_PLAN.md](../PROJECT_PLAN.md)
for the week-by-week plan.

## Setup

```bash
cd Codes
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Structure

```
Codes/
  src/
    data/           # config, manifest, data loading/acquisition
    preprocessing/   # cloud masking, speckle filtering, co-registration (Week 3-4)
    features/        # spectral indices, GLCM texture, fusion (Week 5, 7)
    models/          # training (Week 6-7)
    evaluation/       # metrics, confusion matrices (Week 6)
  scripts/          # runnable entry points, one file = one thing you can run
  notebooks/        # exploration only, nothing here is part of the final pipeline
  tests/            # pytest sanity tests
```

## Week 2: verify your setup

We don't have confirmed access to the real YieldSAT files yet, so `src/data/synthetic.py`
generates small fake-but-structured fields (same GeoTIFF format, same manifest shape)
so the rest of the pipeline can be built and tested before real data arrives. See
[DECISIONS.md](../DECISIONS.md) for why.

Run the tests:

```bash
python -m pytest -q
```

Run the demo (generates synthetic fields, reads them back, computes NDVI, saves a figure):

```bash
python scripts/week2_demo.py
```

Output goes to `../Results/dev/week2_demo.png`. This is synthetic data, not a project result.

## Google Earth Engine (needed from Week 4 onward)

We share one GEE Cloud project across the team: **`fusioncrop`**. Each member still
needs to authenticate their own Google account once (this step cannot be shared):

```bash
earthengine authenticate
```

Then set `GEE_PROJECT_ID` to `fusioncrop` (not a personal project) and run the check:

```bash
# PowerShell
$env:GEE_PROJECT_ID = "fusioncrop"
python -m src.data.gee_auth_check

# bash
GEE_PROJECT_ID=fusioncrop python -m src.data.gee_auth_check
```

It should print `Earth Engine is ready.` Confirmed working as of 17 Sep 2026, see
[DECISIONS.md](../DECISIONS.md) entry 10.
