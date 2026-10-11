# Mid-Sem Presentation: Speaker Notes

Total time about 10 minutes. The same text is in the notes pane of the pptx.

## Slide 1: Title

TIME: 30 seconds

SAY:
Good morning. We are Group 3 of Project 19 in ECE 501, Digital Image Processing. I am Het Patel, and with me are Dhairya Shah, Jay Daftari and Hiya Soni.

Our project is about one question: can we tell which crop grows in a field if we look at it with two different kinds of satellite image together? The title says it in short. SAR is radar imaging. Optical is normal sunlight imaging. Fusion means combining what we learn from both. Classification means the output is a label, here corn, soybean or wheat. YieldSAT is the dataset we use.

Today we cover only the first half of the semester, which is the data and the processing pipeline. The classifier comes in the second half.

TRANSITION: Let me start with why this matters.

## Slide 2: Motivation

TIME: 1.5 minutes

SAY (flow: why crop type, why one sensor fails, why radar helps, what we ask):
1. Why crop type. Knowing which crop grows where helps with farm planning and food security. A crop map tells planners what is planted and how much, over areas too large to visit on foot. Satellites are the only practical way to do this at scale.

2. Two kinds of satellite. Optical satellites such as Sentinel-2 are passive. They measure sunlight reflected from the ground, in colours our eyes see and in near-infrared. Healthy plants reflect strongly in near-infrared, so optical images tell us a lot about greenness. Radar satellites such as Sentinel-1 are active. They send their own microwave pulse and measure what comes back. They do not need sunlight.

3. The problem with optical, shown by our own data. This chart is from three real fields, one per crop. The grey part is the number of dates we had to drop because clouds or cloud shadow covered too much of the field. Corn keeps 52 of 96 dates, soybean 47 of 81, wheat 49 of 93. Added together that is 148 kept out of 270, about 55 percent kept, so about 45 percent lost. Note these are three fields only, so treat 55 percent as an indication, not a dataset-wide number.

4. Why radar helps. Radar waves pass through clouds, so a radar image exists even on the dates optical is blocked. It also responds to different things: the structure and moisture of the plants, not their colour. So the two sensors give two different views of the same field.

5. The catch. Radar images carry speckle, a grainy noise. We cannot use them raw, which is why a filter is part of our pipeline.

INFERENCE: Neither sensor is enough alone. Optical is rich but incomplete in time. Radar is complete in time but noisy. That is the reason to fuse them.

KPI THIS SLIDE SUPPORTS: fraction of dates kept by the cloud mask, about 55 percent.

IF ASKED: "Why not just wait for a clear day?" A crop grows on a schedule, and the clear days may fall outside the growth stage that separates crops.

TRANSITION: So the exact problem we are solving is the following.

## Slide 3: Problem Definition

TIME: 1 minute

SAY:
Our problem in two sentences. Given Sentinel-2 and Sentinel-1 images of the same field, predict whether it grows corn, soybean or wheat. And compare three settings: optical-only, SAR-only and fused features, using classical machine learning.

Words to explain:
- Hand-crafted features: numbers we compute from the images by a known formula, for example a vegetation index or the mean radar backscatter. This is different from deep learning, where a network learns features by itself. We do not use deep learning, because the course asks for explainable features.
- Classical machine learning: models such as Random Forest and SVM, which take a feature table and output a class.
- Fused: the optical and SAR feature numbers joined into one vector per field.

The right-hand card is our scope today. For mid-sem we only built and tested the data pipeline, on three real fields from Argentina, plus a check of the whole dataset. The left card is what comes next. Our group was assigned Argentina and Brazil.

HONEST POINT: We have not trained a classifier yet. So we do not claim that fusion is better. That is what the second half will measure.

TRANSITION: Before the method, what does existing work give us?

## Slide 4: Literature Review

TIME: 1 minute

SAY:
We rely on six sources, and each one maps to a part of our pipeline. I will go through the table by its last column.

[1] Miranda and colleagues, YieldSAT, CVPR 2026. This is our dataset. It is a multimodal benchmark with Sentinel-2 images and crop labels for many fields. Importantly, it has no radar data.
[2] Torres and colleagues describe the Sentinel-1 mission. That is our radar source, in VV and VH polarisation.
[3] Gorelick and colleagues describe Google Earth Engine, the cloud platform we use to search and download Sentinel-1.
[4] Lee, 1980. The speckle filter. It uses the mean and variance in a local window. We use a 5 by 5 window.
[5] Rouse and colleagues, 1974. They introduced NDVI, the vegetation index. We compute NDVI and three more indices.
[6] Haralick and colleagues, 1973. Gray level co-occurrence matrix texture features. This is planned for the second half.

THE GAP, the line at the bottom: YieldSAT does not contain radar, so the radar half of the project is built by us. That is a real piece of work, not just reading data that already exists.

IF ASKED about other fusion papers: our mid-sem report cites only these six, so I will say that a wider survey of fusion methods is part of the end-sem report.

TRANSITION: Now the data itself.

## Slide 5: Dataset Check

TIME: 1 minute

SAY:
First KPI: coverage. We scanned every field folder in our two countries. All 751 Argentina fields and all 551 Brazil fields loaded without errors, 1302 in total. The crop counts also match the YieldSAT paper. Argentina has 185 corn, 440 soybean and 126 wheat fields. Brazil has 118 corn, 293 soybean and 140 wheat.

What the chart tells us: the classes are not balanced. Soybean is the biggest class in both countries. So later we cannot trust accuracy alone. A model that always says soybean would already look decent. That is why we will also report F1-score and confusion matrices.

Dataset details: each field is a folder with Sentinel-2 images for every date between seeding and harvest. Each image has 12 bands at 10 metre resolution. Values are 16-bit integers, so we divide by 10000 to get reflectance. The value 0 means no data. Each date also has an SCL file, the scene classification layer, which gives a class for every pixel.

The issue we found: the image is a rectangle but the field is not. Corners outside the field have SCL class 0. Our first mask let these pixels through. We fixed it by keeping only an allow-list of usable classes. I mention it because it shows why we checked every field and not a sample.

TRANSITION: Next, how we turn these raw images into clean inputs.

## Slide 6: Processing Pipeline

TIME: 1.5 minutes (this is the core method slide, go slowly)

SAY: two branches, same field, same grid.

OPTICAL BRANCH, left to right:
1. Input: Sentinel-2 images for each date with their SCL layer.
2. Cloud mask: we keep SCL classes 4, 5, 6 and 7, which are vegetation, bare soil, water and unclassified. We drop cloud, cloud shadow, snow and no data. If fewer than 20 percent of a date's field pixels are usable, we drop the whole date.
3. Median composite: for each pixel and each band we take the median over the valid dates. Formula on the report: C of x, y, b equals the median over valid dates of the image value. Why median and not mean: one cloud pixel that the mask missed pulls the mean a lot, but barely moves the median.
4. Four indices. NDVI is near-infrared minus red over near-infrared plus red. Higher means greener. EVI is a corrected version that uses blue to reduce atmosphere and soil effects. GNDVI uses green instead of red and responds to chlorophyll. NDWI uses green and near-infrared and responds to water.

SAR BRANCH:
1. Input: Sentinel-1 GRD scenes in interferometric wide mode, with VV and VH. VV means sent vertical and received vertical. VH means sent vertical and received horizontal. We query them in Google Earth Engine over the exact extent of the field and between its seeding and harvest dates.
2. Median of all scenes, downloaded at 10 metres in decibels. Decibel is a log scale, ten times log base ten of the power, which is why values are negative.
3. Lee filter, 5 by 5. Speckle comes from radar waves interfering with each other. The filter replaces each pixel with a mix of the local mean and the pixel. In flat areas it moves toward the mean. Near edges it keeps the pixel, so edges are not blurred.
4. Bilinear resampling onto the pixel grid of the YieldSAT field.

WHY THE LAST BOX MATTERS: after alignment, one optical pixel and one SAR pixel cover the same ground. Without this, we could not build one feature vector per field.

TRANSITION: Let us see what the pipeline produced on real fields.

## Slide 7: Results: Three Real Fields

TIME: 1.5 minutes

SAY:
We ran the full pipeline on one real Argentina field per crop. The figure shows the corn field in four panels. A is the cloud-free optical composite. B is NDVI, where green means higher values. C is the Sentinel-1 VV image as downloaded, and D is the same after the Lee filter and alignment. Look at C and D. The grain in C is speckle. In D the field texture is smoother but the dark patch at the bottom left is still there, so we removed noise and kept structure.

Now the table, with the KPIs:
- Dates kept: 52 of 96 for corn, 47 of 81 for soybean, 49 of 93 for wheat. About 55 percent each.
- Sentinel-1 scenes used: 15, 12 and 14.
- NDVI: 0.52 for corn, 0.41 for soybean, 0.48 for wheat.
- VV: minus 10.23, minus 9.44 and minus 14.90 decibels.
- VH: minus 17.50, minus 16.81 and minus 22.22 decibels.

INFERENCES, said carefully:
1. The VV minus VH gap is about 7.3 decibels for all three fields (7.27 to 7.37). That consistency tells us the radar download and filtering behave sensibly.
2. The wheat field has a much lower VV, about 4.7 to 5.5 decibels below the others. It is also from a different season: planted in June and harvested in November 2023, while corn and soybean were planted in October and November and harvested in March and April. The season or the crop structure may explain it.
3. NDVI values look low because the composite averages the whole season, including early growth. A peak-season composite should give higher values, and that is our next step.

LIMIT: one field per crop, one country. We cannot conclude that wheat always has lower VV. These are pipeline checks, not classification results.

TRANSITION: One more check, on the filter itself.

## Slide 8: Results: Speckle Filtering

TIME: 1 minute

SAY:
How do we know the Lee filter works and does not just blur everything? We tested it on a real Sentinel-1 composite of a farmland area in Argentina, 112 by 113 pixels, built from 5 scenes. The figure shows VV and VH, raw on the left and filtered on the right.

Three numbers answer the question:
1. The standard deviation of VV dropped from 1.35 to 1.08 decibels. That is a fall of about 20 percent, so the pixel-to-pixel noise is lower.
2. The VV mean stayed at minus 12.87 decibels. This is the important check. A good despeckling filter reduces noise and does not shift the signal, because the mean backscatter is exactly what we will use as a feature.
3. Visually, the large structures remain and the grain is smoothed.

Why the standard deviation is a fair measure: speckle is random pixel-to-pixel variation, so reducing it lowers the standard deviation. The caution is that a field with real texture also has variation, so the drop is a proxy and not a perfect noise measure.

IF ASKED "why not a stronger filter or a bigger window": a larger window smooths more but blurs field borders. We picked 5 by 5 as the standard choice and may test other sizes if time allows. We have not yet compared filters.

TRANSITION: So what is next?

## Slide 9: Next Plan

TIME: 1 minute

SAY:
Three steps lead to a classifier.

1. Features. We switch from a whole-season composite to a peak-season window, because that should give higher and more separable NDVI values. We also add GLCM texture features, which describe how gray levels are arranged in the image, such as contrast and homogeneity. This follows Haralick, reference 6.
2. Feature table. We scale the pipeline from three fields to many fields, and store optical, SAR and fused features in one table.
3. Models. We train Random Forest and SVM. Random Forest is a set of decision trees that vote. SVM finds a boundary that separates classes with the largest margin.

Evaluation. We compare optical-only, SAR-only and fused settings. The KPIs are accuracy, F1-score and the confusion matrix. Accuracy is the share of correct predictions. F1-score balances precision and recall per class, which matters because soybean dominates. The confusion matrix shows which crops get mixed up.

TARGET: end-sem presentation is around 16 November.

HONEST FRAMING: we do not yet know whether fusion beats optical-only. A possible outcome is that it does not help much. Either result is a valid finding.

TRANSITION: Finally, who did what.

## Slide 10: Individual Contribution

TIME: 45 seconds

SAY (each member introduces their own line, or one person reads it):
- Het Patel: Sentinel-1 download through Google Earth Engine, the Lee filter, and alignment of SAR to the optical grid.
- Dhairya Shah: the optical pipeline, which is the SCL cloud mask, the median composite and the four vegetation indices.
- Jay Daftari: the scan of all 1302 fields, the crop count check and the fix for the mask problem.
- Hiya Soni: the speckle test, the result figures and the field summary table.

NOTE FOR THE TEAM BEFORE PRESENTING: this split is a draft. Edit the slide so that it matches what each person actually did.

CLOSING:
To summarise: the optical branch gives a clean composite and four indices, the SAR branch gives a filtered image on the same grid, and both ran on real data from all 1302 fields in the dataset check and on three fields end to end. Next is texture, a feature table and the classifiers. Thank you. We are happy to take questions.

LIKELY QUESTIONS:
- Why median? Robust to missed cloud pixels.
- Why dB? Radar power spans a large range, a log scale makes it manageable.
- Is 55 percent typical? Only three fields. We will report the dataset-wide figure later.
- Did you use deep learning? No, the project is limited to hand-crafted features and classical models.

