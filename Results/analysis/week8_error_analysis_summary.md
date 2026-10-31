# Week 8: error analysis of the fused Random Forest

Model: the best Week 7 setting (fused features, tuned Random Forest). Same split as Weeks 6 and 7: 216 train and 54 test fields.

## Per crop (18 test fields each)

| Crop | Precision | Recall | F1 |
|---|---|---|---|
| Corn | 0.94 | 0.89 | 0.91 |
| Soybean | 0.89 | 0.94 | 0.92 |
| Wheat | 0.94 | 0.94 | 0.94 |

No crop is clearly weak. Corn has the lowest recall.

## Per country (same test set, split by country)

| Country | Test fields | Accuracy |
|---|---|---|
| Argentina | 24 | 0.875 (21 correct) |
| Brazil | 30 | 0.967 (29 correct) |

The gap is 2 fields out of a very small test set, so we do not treat it as a real difference.

## Feature importance

- The two most important features are both SAR: mean VV (about 0.25) and the VV minus VH difference (about 0.14).
- The next one is optical: mean EVI (about 0.09).
- In the top 10, 4 features are SAR and 6 are optical.
- Simple mean values matter more than texture. The best texture feature is the NIR correlation at about 0.06.

## Cross-country generalization (train on one country, test on the other)

| Train | Test | Accuracy | Macro F1 |
|---|---|---|---|
| Argentina | Brazil | 0.52 | 0.51 |
| Brazil | Argentina | 0.45 | 0.44 |

The mixed-country test macro F1 is 0.93, and the cross-country average is 0.48. This is a drop of about 0.45. With 3 crops, random guessing gives about 0.33.

## What this means

- Inside the mixed data the fused model works well and no crop or country stands out as a problem.
- The model does not carry over from one country to the other. The Week 7 gain from fusion should be read as "fusion helps when training and test data come from the same mix of countries".
- A likely reason is that the two countries have different seasons, climate and farming practice, so the same crop looks different in the features. We have not tested this explanation yet.

## Limits

- Test sets are small: 54 fields in total, 24 and 30 per country, 18 per crop.
- One split only.
- The cross-country test trains on only 133 or 137 fields.
