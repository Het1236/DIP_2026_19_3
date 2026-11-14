# DIP_2026_19_3 — SAR–Optical Feature Fusion

ECE 501 (Digital Image Processing) course project, Ahmedabad University.

## Problem Statement

Given SAR (Sentinel-1) and optical (Sentinel-2) field crops with an associated crop-type or
yield-derived label, extract classical hand-crafted features from each modality, fuse them, and
train a classical machine learning model to compare optical-only, SAR-only, and fused-feature
classification performance.

- **Input:** Sentinel-1 and Sentinel-2 field crops with crop-type labels and per-pixel yield masks,
  for the group's assigned countries (YieldSAT-derived; Group A: Germany + Uruguay, Group B:
  Argentina + Brazil).
- **Expected output:** Extracted feature vectors per modality; trained models for optical-only,
  SAR-only, and fused-feature settings; accuracy, F1-score, and confusion matrix per setting.

## Team

Group 3, Project 19, Ahmedabad University.

| Name | Enrolment No. |
|---|---|
| Het Patel | AU2440266 |
| Dhairya Shah | AU2440128 |
| Jay Daftari | AU2440271 |
| Hiya Soni | AU2440084 |

## Main Results

Dataset: YieldSAT, Argentina and Brazil. Features were built for 270 real fields
(133 Argentina, 137 Brazil; 91 corn, 88 soybean, 91 wheat). Classical features only, no
deep learning. Macro F1-score:

| Features | Random Forest | SVM |
|---|---|---|
| Optical only (12 features) | 0.63 | 0.71 |
| SAR only (13 features) | 0.81 | 0.80 |
| Fused (21 features) | 0.86 | **0.91** |

These are means from a nested 5-fold cross validation over all 270 fields. On one held-out
test split (54 fields) the fused Random Forest reaches 0.93. The 270 fields come from only
42 farms, so we also held out whole farms (`Codes/scripts/farm_grouped_cv.py`). Then the
fused SVM scores 0.87 and the fused Random Forest 0.82, against 0.81 for the best single
sensor, so the fusion gain is modest and the Random Forest gain is not shown. A model trained
on one country and tested on the other falls to a macro F1-score of about 0.48. The full
tables and figures are in `Results/final/`, and the discussion is in the end-semester report.

## Repository Structure

- `Mid_Sem_Report/` — 2-page, 2-column IEEE-style mid-semester report
- `Mid_Sem_Presentation/` — mid-semester slides and speaker notes
- `End_Sem_Report/` — 4-page, 2-column IEEE-style end-semester report
- `End_Sem_Presentation/` — end-semester slides and speaker notes
- `Results/` — graphs, images, tables (`Results/final/` holds the final results and models)
- `Codes/` — project source code

## Reproducing the Results

The feature table is committed, so the models and figures can be reproduced without the
YieldSAT download. See [Codes/README.md](Codes/README.md) for the setup and the order in
which to run the scripts. Building the feature table from the raw data needs the dataset
(about 15 GB, not committed) and a Google Earth Engine account.

## Timeline

- Weekly commit: every **Saturday, 5:00 PM** (except midterm exam week)
- Mid-semester presentation: ~**12 October** (report + 10 min presentation)
- End-semester presentation: ~**16 November** (report + 10 min presentation)
