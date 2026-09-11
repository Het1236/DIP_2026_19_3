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

## Repository Structure

- `Mid_Sem_Report/` — 2-page, 2-column IEEE-style mid-semester report
- `End_Sem_Report/` — 4-page, 2-column IEEE-style end-semester report
- `Results/` — graphs, images, tables
- `Codes/` — project source code

## Timeline

- Weekly commit: every **Saturday, 5:00 PM** (except midterm exam week)
- Mid-semester presentation: ~**12 October** (report + 10 min presentation)
- End-semester presentation: ~**16 November** (report + 10 min presentation)
