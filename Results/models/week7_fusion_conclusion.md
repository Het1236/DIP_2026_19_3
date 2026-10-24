# Week 7: first draft conclusion on fusion

Setup: 270 real fields (Argentina and Brazil), 216 train and 54 test, the same split as Week 6. Models are tuned with 5-fold cross validation on the training data. Scores below are on the test set.

| Setting | Features | Random Forest macro F1 | SVM macro F1 |
|---|---|---|---|
| Optical only | 12 | 0.63 | 0.76 |
| SAR only | 13 | 0.79 | 0.76 |
| Fused | 21 | 0.93 | 0.91 |

Fusion gave the best result for both models. Random Forest on fused features got 50 of 54 test fields right. The best single-modality model (SAR, Random Forest) got 43 of 54 right.

The cross validation scores point the same way, with a smaller gap. Fused cross validation macro F1 is 0.87 for Random Forest and 0.89 for SVM. The best single-modality value is 0.83 (SAR, SVM).

The 25 raw fused features were reduced to 21 by a correlation filter at 0.95. The dropped features are mean_gndvi, mean_ndwi, mean_vh_db and ndvi_glcm_energy.

Limits:
- The test set has 54 fields, so one field changes accuracy by about 2 percent.
- This is one split. The size of the gain may change on another split.
- Tuning helped optical and SVM models but SAR Random Forest got slightly worse than its untuned Week 6 value (0.81 to 0.79).

Draft conclusion: in this first test, combining optical and SAR features gave a clear gain over either alone. The cross validation gain is smaller than the test gain, so the exact size needs more checking in the next weeks.
