# End-Sem Presentation: Speaker Notes

Total talk about 9.5 minutes. The same text is in the notes pane of the pptx.

## Slide 1: Title

TIME: 20 seconds. Total talk: about 9.5 minutes, leaving 30 seconds of slack.

SAY:
Good morning. We are Group 3 of Project 19 in ECE 501, Digital Image Processing. I am Het Patel, and with me are Dhairya Shah, Jay Daftari and Hiya Soni.

Our project asks one question: can we tell which crop grows in a field if we use optical and radar satellite images together, and does using both beat using only one? At mid-sem we showed the data pipeline. Today we show the full result.

TRANSITION: First, why this matters.

## Slide 2: Motivation

TIME: 1 minute

SAY:
1. Why crop type. Knowing which crop grows where helps with farm planning and food security. Satellites are the only practical way to do this over large areas.

2. Two kinds of satellite. Optical satellites such as Sentinel-2 are passive. They measure reflected sunlight, so they show greenness. Radar satellites such as Sentinel-1 are active. They send a microwave pulse and measure what comes back, so they need no sunlight.

3. The problem with optical, from our own data. This chart is from three real fields, one per crop. Grey is the number of dates we had to drop because clouds or shadow covered too much of the field. Corn keeps 52 of 96 dates, soybean 47 of 81 and wheat 49 of 93. Together that is 148 of 270, about 55 percent kept, so about 45 percent lost. These are only three fields, so treat it as an indication.

4. Radar passes through clouds, so it is not affected by the clouds that remove the optical dates. But radar images carry speckle noise, so we filter them.

INFERENCE: Optical is rich but incomplete in time. Radar is not blocked by clouds but is noisy. That is why we ask whether fusing them helps.

TRANSITION: So the exact problem is the following.

## Slide 3: Problem Definition

TIME: 40 seconds

SAY:
Given Sentinel-2 and Sentinel-1 images of the same field, predict whether it grows corn, soybean or wheat. And compare three settings: optical-only, SAR-only and fused, using classical machine learning.

Words to explain:
- Hand-crafted features: numbers we compute by a known formula, such as a vegetation index. Not deep learning. The brief asks for explainable features.
- Fused: the optical and SAR feature numbers joined into one vector per field.
- Classical machine learning: Random Forest and SVM.

The three cards are what we did. A pipeline for both image types. Features and fusion. And three tests. First, a train and test split with 5-fold cross validation. Second, a stricter cross validation that holds out whole farms. Third, a test where we train on one country and test on the other. Our group was assigned Argentina and Brazil.

TRANSITION: What does existing work give us?

## Slide 4: Literature Review

TIME: 45 seconds

SAY: six sources, each used in one part of the project.
[1] Miranda and colleagues, YieldSAT, CVPR 2026. Our dataset: Sentinel-2 images and crop labels for many fields, no radar.
[2] Torres and colleagues describe the Sentinel-1 mission. It is our radar source, VV and VH polarisation.
[3] Gorelick and colleagues describe Google Earth Engine, which we use to download Sentinel-1.
[4] Lee, 1980. The speckle filter. It uses the local mean and variance in a window. We use 5 by 5.
[5] Rouse and colleagues, 1974. The standard reference for the vegetation index NDVI. We also use EVI, GNDVI and NDWI.
[6] Haralick and colleagues, 1973. Texture features from the gray level co-occurrence matrix. We use contrast, homogeneity, energy and correlation.

THE GAP: YieldSAT has no radar, so the radar half of the project is built by us.

IF ASKED about other fusion papers: our reports cite these six. A wider survey is outside what we completed.

TRANSITION: Now the method.

## Slide 5: Method

TIME: 1.5 minutes (core slide, go slowly)

OPTICAL BRANCH:
1. Sentinel-2 images for each date, with the scene classification layer, SCL, which gives a class for every pixel.
2. Cloud mask: keep SCL classes 4, 5, 6 and 7, which are vegetation, bare soil, water and unclassified. Drop cloud, shadow, snow and no data. If fewer than 20 percent of a date's field pixels are usable, drop the date.
3. Median composite over the valid dates, using only the middle of the season, from 30 to 70 percent of the way between seeding and harvest. Why: the whole-season composite mixes early growth and stubble, which gave low NDVI of 0.41 to 0.52. With the window the mean NDVI is about 0.80 for corn, 0.78 for soybean and 0.77 for wheat. Why median: one missed cloud pixel moves the mean but barely moves the median. One rule for all crops, because we have no crop calendar.
4. Features: means of NDVI, EVI, GNDVI and NDWI, plus four texture values on NIR and four on NDVI. That is 12 features.

SAR BRANCH:
1. Sentinel-1 scenes, interferometric wide mode, VV and VH, from Google Earth Engine for the same field and season. Median of the scenes, in decibels, at 10 metres.
2. Lee filter, 5 by 5, to reduce speckle. On a real test composite it cut the VV standard deviation from 1.35 to 1.08 dB, about 20 percent, and kept the mean at minus 12.87 dB.
3. Resample onto the optical grid, so one optical pixel and one SAR pixel cover the same ground.
4. Features: mean and standard deviation of VV and VH, the VV minus VH difference, and texture on VV and VH. That is 13 features.

FUSION: early fusion. Join the 25 features into one vector, then drop one of each pair correlated above 0.95, using training data only. This removes 4 features: mean GNDVI, mean NDWI, mean VH and the NDVI texture energy. 21 features remain. We tune Random Forest and SVM with 5-fold cross validation on the training data.

DATA (bottom card): we sampled 50 fields per country and crop with Good or Average quality, 300 in total. Features were built for 270. The 30 that failed: 16 had no Sentinel-1 scenes in their dates, 11 were too cloudy, 2 had no Sentinel-2 date in the window, 1 was a server error. Split: 216 train and 54 test, same crop mix in both. The 270 fields come from only 42 farms, and the largest farm has 43 of them. This matters for the farm test on the next slide.

TRANSITION: Now the results.

## Slide 6: Results: Fusion

TIME: 1.5 minutes

SAY:
The chart shows macro F1-score from nested 5-fold cross validation over all 270 fields. Macro F1 gives each crop the same weight. Nested means the tuning happens inside each training fold, so the test fold never influences the model.

How to read the bars. There are four per group. Dark bars are random folds. Light bars are the stricter version where whole farms are held out. Why we added the light bars: fields of one farm look alike. In random folds a farm can appear in both training and test, and that can flatter the score. In the light bars every farm stays on one side.

NUMBERS (random folds, then farms held out):
- Optical only: Random Forest 0.63 then 0.57. SVM 0.71 then 0.67.
- SAR only: Random Forest 0.81 then 0.76. SVM 0.80 then 0.81.
- Fused: Random Forest 0.86 then 0.82. SVM 0.91 then 0.87.
The spread between folds is about 0.03 to 0.10.

INFERENCE:
1. The order is the same in every case: optical is lowest, SAR is higher, fused is highest.
2. When farms are held out, five of the six scores fall, by 0.04 to 0.06. So the random-fold numbers were a little optimistic. The honest range for the best model, the fused SVM, is 0.87 to 0.91.
3. The gain over the best single sensor, which is 0.81 in both cases. For the SVM it is about 0.10 with random folds and 0.055 with farms held out. For the Random Forest it is about 0.05 and then only about 0.01, which is inside the spread. So we claim a real but modest gain for the SVM, and we do not claim it for the Random Forest.
4. Adding SAR to optical is worth much more than adding optical to SAR: about 0.20 for the SVM and 0.23 to 0.25 for the Random Forest, in both cases.

On our single 54-field test split the fused Random Forest got 0.93 and 50 of 54 right, against 43 of 54 for the best single type. That split is probably on the high side. One field changes accuracy by about 2 percent there. Also the best fused model differs: Random Forest on the test split, SVM in cross validation. So we do not name one best model from the split.

KPIs: macro F1, accuracy, and the fold-to-fold standard deviation.

IF ASKED "why is SAR better than optical": a possible reason is that radar is not blocked by clouds, so the SAR features use more of the season. We have not tested this.

IF ASKED "so which number do you trust": the farm-held-out one, 0.87.

TRANSITION: Why does fusion help? Let us look at the errors.

## Slide 7: Results: Error Analysis

TIME: 1 minute

SAY:
These are test-set confusion matrices for the best model of each feature set. Rows are the true crop, columns the predicted crop, and the diagonal is correct. Correct fields out of 54: optical-only SVM 41, SAR-only Random Forest 43, fused Random Forest 50.

THE KEY POINT: the two single sensors fail in different places.
- SAR gets all 18 wheat fields right, but confuses corn and soybean: 6 corn called soybean and 3 soybean called corn.
- Optical is weaker on wheat, 4 of 18 wrong. Its 6 wrong corn fields split evenly: 3 called soybean and 3 called wheat.
- Count by field. Optical is wrong on 13 test fields and SAR on 11. Only 5 are wrong in both. 14 fields are wrong in exactly one of them, and the fused model fixes 12 of those 14. It makes no new mistake on a field that both single-sensor models get right.

WHY (likely, not tested): radar responds to structure and water in the crop, optical responds to leaf colour and greenness. One sensor often holds the clue that the other lacks. The SAR errors also suggest corn and soybean look alike to radar in this data.

OTHER NUMBERS (fused Random Forest, 18 test fields per crop):
- F1 per crop: corn 0.91, soybean 0.92, wheat 0.94. No crop is clearly weak.
- Per country: Argentina 21 of 24 correct, Brazil 29 of 30. The gap is 2 fields on a small test set, so we do not treat it as real.
- Feature importance: the top two features are both SAR, the mean VV at about 0.26 and the VV minus VH difference at about 0.14. The third is the mean EVI at about 0.09. SAR features hold 65 percent of the total importance and optical 35 percent.
- Six simple statistics hold 61 percent of the importance and the 15 texture features share the other 39 percent. So one simple statistic is worth much more than one texture value.

INFERENCE: fusion works because the sensors are complementary, not because one is simply better. A better radar feature for corn versus soybean would help more than another texture value.

TRANSITION: One more test, and it is the important one.

## Slide 8: Results: Cross-Country Test

TIME: 1.25 minutes

SAY:
So far training and test data came from the same mix of Argentina and Brazil. In practice a model must work on a new region. So we trained on one country and tested on the other.

NUMBERS (fused Random Forest):
- Train on Argentina, test on Brazil: accuracy 0.52, macro F1 0.51.
- Train on Brazil, test on Argentina: accuracy 0.45, macro F1 0.44.
- The mixed test was 0.93. The average across countries is 0.48, a drop of about 0.45.
- Random guessing among three crops gives about 0.33. So the model is better than guessing but not useful.

INFERENCE: the model works inside the mixed data and does not carry over to a new country. The YieldSAT paper also warns that models lose accuracy on new regions and years.

A LIKELY REASON: the two countries have different seasons, climate and farming, so the same crop gives different feature values. We have NOT tested this. Another factor: these runs train on only 133 or 137 fields. The two sizes are almost equal, so training size does not explain the drop.

HONEST MESSAGE: our fusion result holds for data from the same mix of countries. It should not be read as a model that works anywhere.

IF ASKED "what would fix it": more countries in training, features that are standardised per country, or adapting the features to the new country. We list it as future work and did not include it in the reported results.

TRANSITION: To conclude.

## Slide 9: Conclusion

TIME: 45 seconds

SAY:
What we found:
- Fused features gave the best result. The fused SVM reached a macro F1 of 0.91 in nested cross validation and 0.87 when whole farms were held out, against 0.81 for the best single sensor.
- The sensors are complementary: only 5 of 54 test fields are wrong in both.
- SAR features carry about 65 percent of the importance, and the mean VV is the strongest feature.
- Across countries the macro F1 falls to about 0.48.

Limits, which we state ourselves:
- We used 270 of the 1302 fields, and they come from only 42 farms.
- The test set has only 54 fields, which is why we added cross validation.
- The Random Forest gain over the best single sensor almost disappears when farms are held out, so we do not claim it.
- We have not tested why the countries differ.

Next steps: use all 1302 fields and more farms, test why the countries differ, try adapting the features to a new country.

TRANSITION: Finally, who did what.

## Slide 10: Individual Contribution

TIME: 40 seconds

SAY (each member says their own line, or one person reads it):
- Het Patel: project lead, the Sentinel-1 download through Google Earth Engine, the Lee filter and the alignment of SAR to the optical grid.
- Dhairya Shah: the optical pipeline, which is the cloud mask, the composite and the vegetation indices.
- Jay Daftari: the texture features, the feature fusion and the feature table.
- Hiya Soni: the models and tuning, the evaluation and the result figures.

NOTE FOR THE TEAM BEFORE PRESENTING: this split follows the roles in our project plan. Edit the slide so it matches what each person actually did.

CLOSING: Thank you. We are happy to take questions.

LIKELY QUESTIONS AND SHORT ANSWERS
- Why only 270 of 1302 fields? Each field needs its own Sentinel-1 download, which is slow. We took a balanced sample of 50 per country and crop and kept the full run as future work. We checked that all 1302 fields load.
- Why only Argentina and Brazil? Those are the countries assigned to our group.
- Why Random Forest and SVM? The brief asks for classical, explainable models and no deep learning.
- Why macro F1? It weights each crop equally. Soybean is the largest class.
- Is 0.91 overfitted? Tuning and the feature filter use only training data in every fold. But random folds can put the same farm on both sides, so we also held out whole farms. Then the fused SVM scores 0.87. So the honest range is 0.87 to 0.91.
- Which number do you trust most? The farm-held-out one, 0.87 for the fused SVM.
- Why is the test split different from cross validation? The test set has 54 fields, so one field is about 2 percent. Cross validation uses all 270 fields.
- Why did you use the middle 30 to 70 percent of the season? The whole-season composite mixed early growth and stubble and gave low NDVI. We have no crop calendar, so we used one simple rule for all crops. We did not tune the window.
- Did you use yield? No. YieldSAT is a yield dataset but we used only the crop labels.
- Could fields from the same farm leak between train and test? In random folds yes, so we tested it. With whole farms held out the fused SVM falls from 0.91 to 0.87 and the fused Random Forest from 0.86 to 0.82. The gain over the best single sensor shrinks to about 0.055 for the SVM and about 0.01 for the Random Forest. We still have only 42 farms.
- Does fusion really help then? Yes for the SVM, by about 0.05 even with farms held out. For the Random Forest the gain is not shown. The sensors also fail on different fields, which is why fusion helps.
- Does it work on a new country? No, macro F1 falls to about 0.48.
