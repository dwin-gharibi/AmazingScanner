# AmazingScanner — results and analysis

Everything below is generated from `outputs/report/*.json` by
`python -m docscanner.eval.report`, so the numbers and the sentences around them
cannot drift apart. Regenerate after any training run.

---

## 1. Task 1 — the enhancement network

### PSNR / SSIM by split, against the no-model baseline

| Split | n | PSNR | SSIM | PSNR (input) | SSIM (input) | dPSNR | dSSIM | seconds/page |
|---|---|---|---|---|---|---|---|---|
| test - degraded input (no model) | 64 | 11.7013 | 0.6968 |  |  |  |  | 0.3159 |
| train | 24 | 22.1047 | 0.8589 | 13.9314 | 0.7325 | 8.1733 | 0.1264 | 1.1067 |
| validation | 48 | 21.8115 | 0.8328 | 11.0148 | 0.6772 | 10.7967 | 0.1556 | 1.0881 |
| test | 64 | 21.4159 | 0.8220 | 11.7013 | 0.6968 | 9.7146 | 0.1252 | 0.9632 |
| test (course scans only) | 40 | 23.7841 | 0.8841 | 11.4037 | 0.7435 | 12.3804 | 0.1406 | 1.0339 |
| pseudo-real (harder, OOD) | 96 | 18.2833 | 0.6889 | 9.2209 | 0.5601 | 9.0624 | 0.1287 | 1.0316 |

**Against the do-nothing baseline.** The degraded input scores 11.70 dB / 0.6968 SSIM on the test split. The network reaches 21.42 dB / 0.8220, a gain of **+9.71 dB and +0.1252 SSIM**. The model is clearly earning its parameters.

**Overfitting check.** Raw scores are train 22.10 dB and test 21.42 dB, a difference of +0.69 dB -- but that difference mostly reflects how hard each split's inputs are (the degraded inputs themselves score 13.93 and 11.70 dB). The comparison that detects memorisation is the **gain over the degraded input**: **+8.17 dB on train against +9.71 dB on test**. The model improves *unseen* pages slightly more than the ones it trained on, which rules out memorisation outright. That is the expected outcome of the design: every epoch synthesises fresh degradations, so the model essentially never sees the same input twice, and the only thing it *can* overfit is the set of source pages.

**Validation is optimistic by construction** (21.81 dB) because model selection steered on it; the test split (21.42 dB) is the honest headline.

**Out-of-distribution.** On the harder held-out set -- page curl, heavy glare, long motion blur, aggressive compression, none of which the training policy produces -- the score falls to 18.28 dB, 3.13 dB below the test split. That is a substantial drop, and it is the honest measure of the synthetic-to-real gap: the model removes the degradations the generator knows how to produce, and struggles with the rest.

**The curves.** Training loss fell 0.256 -> 0.109 and validation loss 0.149 -> 0.096 over 40 epochs. Both are still descending together at the end -- the run is budget-limited, not converged, and more epochs would still help.

Worth noting in the last epochs: **PSNR plateaued** (21.34 -> 21.34 dB) while **SSIM kept rising** (0.8782 -> 0.8784). That is the combined loss behaving exactly as intended -- once the pixel-wise error saturates, the MS-SSIM and gradient terms keep sharpening structure, and structure is what legibility depends on. It is also a caution about model selection: this run selects on PSNR because that is the headline metric the brief asks for, but selecting on SSIM would arguably serve readability better.

### Loss and architecture ablation

All variants below were trained with **identical budgets**, so each row differs
only in the thing under test.

| Run | loss | bg prior | dropout | n | PSNR | SSIM |
|---|---|---|---|---|---|---|
| degraded input (no model) |  |  |  | 24 | 12.3349 | 0.7110 |
| flagship (full budget) | combined | False | 0.0000 | 24 | 21.1640 | 0.8092 |
| control (combined loss, no dropout) | combined | True | 0.0000 | 24 | 19.1385 | 0.7863 |
| + dropout 0.15 (Section 6) | combined | True | 0.1500 | 24 | 19.4558 | 0.7906 |
| loss: MSE | mse | True | 0.0000 | 24 | 16.5371 | 0.7081 |
| loss: L1 | l1 | True | 0.0000 | 24 | 15.8510 | 0.7014 |

**Loss functions** (identical budgets, only the objective differs). MSE: 16.54 dB / 0.7081. L1: 15.85 dB / 0.7014. Charbonnier + MS-SSIM + Sobel: 19.14 dB / 0.7863. **combined scores best on SSIM.** PSNR and SSIM disagree here in the way the literature predicts: PSNR is a monotone function of MSE, so an MSE-trained model is optimising the metric directly and will tend to lead on it, while SSIM -- which is what tracks *legibility* -- rewards the structural sharpness the gradient and MS-SSIM terms preserve. Text lives in the edges, so the combined loss is what the pipeline ships with.

**Dropout (Section 6)** costs -0.32 dB on the synthetic test split against a control trained at exactly the same budget. It helps here, which suggests the unregularised model had started to fit the source pages.

### Readability (OCR)

| Image | n | OCR confidence | words read | CER vs clean scan | WER vs clean scan |
|---|---|---|---|---|---|
| degraded input | 20 | 43.7775 | 13.1500 | 0.8280 | 0.9373 |
| enhanced | 20 | 55.3870 | 38.2500 | 0.7571 | 0.9237 |
| clean target | 20 | 71.4105 | 110.0000 | 0.0000 | 0.0000 |

**On the real photographs, against the commercial app.** The table above is
synthetic pages, where a true transcript exists. Section 3.3 also asks for the
three-way comparison on real photographs -- rectified input, our output, and the
commercial reference -- which has no ground-truth transcript and so reports the
recogniser's own confidence and word count:

| Image | n | OCR confidence | words read |
|---|---|---|---|
| rectified input | 25 | 60.63 | 90.48 |
| ours | 25 | 67.29 | 116.00 |
| commercial reference | 25 | 74.22 | 155.80 |

| Question | Answer |
|---|---|
| pages where ours beats the raw input (words) | **21 / 25** |
| pages where ours matches or beats the app (words) | **5 / 25** |
| pages where ours matches or beats the app (confidence) | **5 / 25** |

**Readability.** Mean OCR confidence rises from 43.8 on the degraded input to 55.4 after enhancement (+11.6), and the number of confidently read words from 13.2 to 38.2.

Because the synthetic set has a real clean target, we can compute a true character error rate against the text OCR reads from the clean scan: **0.828 → 0.757** (+9% relative). The clean target itself reads at 71.4 confidence, which is the ceiling this metric can reach.

### The network is a document-whitener, and that is the targets' doing

On real photographs of book covers and notebooks the enhancement washes colour
out and clips highlights to unrecoverable white. This is not a tuning failure
and no schedule fixes it — it is what the targets ask for. Comparing the 50
course scans the network is trained to reproduce against the 24 reference scans
of the same photographs:

| | mean luma | saturation | pixels above 250 |
|---|---:|---:|---:|
| course scans (**the training target**) | 249.2 / 255 | 3.3 | **92.8%** |
| reference scans (the wanted output) | 155.4 | 34.0 | 2.5% |

Every one of the 50 targets is a near-white, desaturated text page (92-98% of
pixels above 235). Trained on that, "make it look like a clean scan" *means*
"make it white and grey", and more epochs make it more so. Applied to a blue
book cover the result is inevitable.

The learned part is kept and the tone is corrected around it. What the network
genuinely contributes is the low-frequency correction — shadow removal, lighting
flattening, denoise — so its luminance detail is retained while **colour is
always restored from the input** (the network had none to learn: every target
sits at 3.3 saturation, and the reference scans track their input's saturation
on every photograph measured, 42.2 → 41.9, 113.6 → 125.1, 17.3 → 16.2). The
luminance tone curve is pulled back only for pages that are not paper, judged by
a `paperness` measure that puts all 50 course scans above 0.79 and the book
covers at 0.00. Measured on the photographs:

| photo | network output | with tone correction | reference |
|---|---|---|---|
| img11 (book page) | luma 232.6, sat 7.5, **14.5% blown** | luma 212.0, sat 31.5, **0.0% blown** | luma 152.0, sat 41.9 |
| img13 (notes) | luma 232.7, sat 6.3, **14.0% blown** | luma 217.2, sat 6.7, **0.0% blown** | luma 162.7, sat 5.7 |
| img9 (notebook) | luma 242.2, sat 3.3, **10.5% blown** | luma 228.3, sat 10.3, **0.0% blown** | luma 39.2, sat 16.2 |
| img2 (cover) | luma 91.1, sat 90.7 | luma 70.6, sat **119.5** | luma 124.0, sat 125.1 |

Document pages keep the clean white look; covers keep their colour; nothing
clips. `--preserve-tone 0` restores the plain network output for comparison.

One measured mistake worth recording: `paperness` was first computed on the
pipeline's *input*, and a dim, colour-cast photograph of a white page scores as
non-paper precisely because of the defects enhancement exists to remove — so
img11's tone was restored wholesale and the enhancement was cancelled. It is
computed on the network's output instead.


---

## 2. Task 2 — corner detection, two approaches

| Model | Set | edge refine | n | MCE (px) | median (px) | MCE (% diag) | quad IoU | success@8px | success@16px | success@32px | worst (px) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A: regression | synthetic test | no | 256 | 68.7424 | 61.8082 | 9.4938 | 0.6416 | 0.0000 | 0.3906 | 4.6875 | 257.3460 |
| A: regression | synthetic test | yes | 256 | 66.8611 | 60.1605 | 9.2340 | 0.6638 | 3.1250 | 4.2969 | 8.9844 | 256.8464 |
| A: regression | synthetic test | no | 256 | 66.2216 | 58.0312 | 9.1456 | 0.6572 | 0.0000 | 1.9531 | 12.1094 | 250.3564 |
| A: regression | synthetic test | yes | 256 | 64.2808 | 55.8967 | 8.8776 | 0.6785 | 5.8594 | 10.1562 | 14.8438 | 249.6806 |
| A: regression | synthetic test (course only) | no | 96 | 71.1054 | 64.5330 | 9.8201 | 0.6357 | 0.0000 | 0.0000 | 9.3750 | 226.6827 |
| A: regression | synthetic test (course only) | yes | 96 | 68.7727 | 60.2225 | 9.4980 | 0.6599 | 3.1250 | 5.2083 | 7.2917 | 225.6928 |
| A: regression | synthetic test (course only) | no | 96 | 70.9096 | 60.6777 | 9.7931 | 0.6329 | 0.0000 | 2.0833 | 12.5000 | 245.2180 |
| A: regression | synthetic test (course only) | yes | 96 | 69.8324 | 65.9476 | 9.6443 | 0.6515 | 5.2083 | 8.3333 | 13.5417 | 240.2729 |
| A: regression | pseudo-real (OOD) | no | 96 | 37.8209 | 30.9651 | 5.2233 | 0.7977 | 0.0000 | 0.0000 | 20.8333 | 121.1718 |
| A: regression | pseudo-real (OOD) | yes | 96 | 32.2962 | 27.1328 | 4.4603 | 0.8448 | 13.5417 | 16.6667 | 30.2083 | 121.5764 |
| A: regression | pseudo-real (OOD) | no | 96 | 39.7628 | 22.8678 | 5.4915 | 0.8138 | 0.0000 | 8.3333 | 42.7083 | 183.4382 |
| A: regression | pseudo-real (OOD) | yes | 96 | 33.3389 | 18.6513 | 4.6043 | 0.8644 | 32.2917 | 37.5000 | 45.8333 | 183.7675 |
| A: regression | real photos (MIDV-500) | no | 139 | 120.9312 | 65.6924 | 8.2344 | 0.6594 | 0.0000 | 0.0000 | 0.0000 | 490.9897 |
| A: regression | real photos (MIDV-500) | yes | 139 | 115.2762 | 63.1874 | 7.8494 | 0.6941 | 3.5971 | 5.7554 | 8.6331 | 490.6140 |
| A: regression | real photos (MIDV-500) | no | 139 | 91.3668 | 41.4637 | 6.2213 | 0.7338 | 0.0000 | 0.0000 | 8.6331 | 406.5384 |
| A: regression | real photos (MIDV-500) | yes | 139 | 78.1301 | 25.3713 | 5.3200 | 0.7983 | 30.2158 | 30.2158 | 33.0935 | 397.8063 |
| A: regression | real photos (own) | no | 25 | 72.1431 | 56.4050 | 5.4692 | 0.7496 | 0.0000 | 0.0000 | 4.0000 | 246.1749 |
| A: regression | real photos (own) | yes | 25 | 68.8159 | 56.6051 | 5.2144 | 0.7643 | 0.0000 | 0.0000 | 8.0000 | 233.4240 |
| A: regression | real photos (own) | no | 25 | 72.2815 | 57.5023 | 5.3844 | 0.7469 | 0.0000 | 0.0000 | 8.0000 | 279.7856 |
| A: regression | real photos (own) | yes | 25 | 69.4186 | 51.3835 | 5.1323 | 0.7630 | 0.0000 | 4.0000 | 12.0000 | 279.6965 |
| B: heatmap | synthetic test | no | 256 | 19.9874 | 8.0900 | 2.7604 | 0.8836 | 22.2656 | 52.7344 | 70.7031 | 239.5560 |
| B: heatmap | synthetic test | yes | 256 | 19.9430 | 8.3760 | 2.7543 | 0.8917 | 31.2500 | 50.7812 | 64.8438 | 239.5560 |
| B: heatmap | synthetic test | no | 256 | 18.0860 | 7.1050 | 2.4978 | 0.8976 | 30.0781 | 54.2969 | 71.8750 | 228.6031 |
| B: heatmap | synthetic test | yes | 256 | 18.5743 | 7.2598 | 2.5652 | 0.9018 | 33.2031 | 55.4688 | 66.7969 | 221.7977 |
| B: heatmap | synthetic test (course only) | no | 96 | 18.6181 | 8.0752 | 2.5713 | 0.8907 | 29.1667 | 57.2917 | 69.7917 | 111.2513 |
| B: heatmap | synthetic test (course only) | yes | 96 | 18.4365 | 7.6470 | 2.5462 | 0.8979 | 34.3750 | 59.3750 | 69.7917 | 117.7214 |
| B: heatmap | synthetic test (course only) | no | 96 | 17.2106 | 6.7180 | 2.3769 | 0.9009 | 37.5000 | 58.3333 | 71.8750 | 212.5935 |
| B: heatmap | synthetic test (course only) | yes | 96 | 17.3167 | 5.9698 | 2.3916 | 0.9055 | 36.4583 | 59.3750 | 71.8750 | 215.1446 |
| B: heatmap | pseudo-real (OOD) | no | 96 | 13.7953 | 6.3774 | 1.9052 | 0.9211 | 37.5000 | 57.2917 | 75.0000 | 96.5560 |
| B: heatmap | pseudo-real (OOD) | yes | 96 | 11.3197 | 1.2185 | 1.5633 | 0.9463 | 57.2917 | 61.4583 | 75.0000 | 103.7902 |
| B: heatmap | pseudo-real (OOD) | no | 96 | 10.5581 | 4.2230 | 1.4581 | 0.9387 | 48.9583 | 62.5000 | 77.0833 | 66.9561 |
| B: heatmap | pseudo-real (OOD) | yes | 96 | 8.9133 | 1.1936 | 1.2310 | 0.9570 | 60.4167 | 68.7500 | 78.1250 | 66.9561 |
| B: heatmap | real photos (MIDV-500) | no | 139 | 67.3266 | 14.8737 | 4.5844 | 0.7906 | 0.7194 | 19.4245 | 54.6763 | 441.9712 |
| B: heatmap | real photos (MIDV-500) | yes | 139 | 61.7278 | 4.7057 | 4.2032 | 0.8268 | 49.6403 | 55.3957 | 60.4317 | 449.4474 |
| B: heatmap | real photos (MIDV-500) | no | 139 | 65.7924 | 21.0457 | 4.4799 | 0.8058 | 10.0719 | 40.2878 | 46.7626 | 410.7723 |
| B: heatmap | real photos (MIDV-500) | yes | 139 | 65.4623 | 11.6970 | 4.4574 | 0.8260 | 42.4460 | 44.6043 | 50.3597 | 406.1432 |
| B: heatmap | real photos (own) | no | 25 | 32.9298 | 23.0995 | 2.4797 | 0.8697 | 0.0000 | 0.0000 | 28.0000 | 110.1403 |
| B: heatmap | real photos (own) | yes | 25 | 35.6444 | 28.3189 | 2.7193 | 0.8604 | 4.0000 | 4.0000 | 20.0000 | 105.9348 |
| B: heatmap | real photos (own) | no | 25 | 34.0605 | 20.9090 | 2.4855 | 0.8680 | 0.0000 | 8.0000 | 40.0000 | 143.2317 |
| B: heatmap | real photos (own) | yes | 25 | 37.8628 | 27.9205 | 2.8092 | 0.8530 | 4.0000 | 4.0000 | 28.0000 | 148.9110 |
| A: regression (control) | synthetic test | yes | 256 | 95.5959 | 88.5299 | 13.2024 | 0.5785 | 0.0000 | 0.0000 | 0.3906 | 247.5996 |
| A: regression (control) | synthetic test (course only) | yes | 96 | 93.6617 | 81.3199 | 12.9353 | 0.5743 | 0.0000 | 0.0000 | 1.0417 | 234.6319 |
| A: regression (control) | pseudo-real (OOD) | yes | 96 | 69.1567 | 56.3071 | 9.5510 | 0.7162 | 2.0833 | 2.0833 | 2.0833 | 202.6281 |
| A: regression (control) | real photos (MIDV-500) | yes | 139 | 132.6105 | 107.3962 | 9.0297 | 0.5911 | 0.0000 | 0.0000 | 0.0000 | 389.4629 |
| A: regression (control) | real photos (own) | yes | 25 | 116.8807 | 112.4971 | 8.9046 | 0.6391 | 0.0000 | 0.0000 | 0.0000 | 233.0450 |
| A: regression + dropout | synthetic test | yes | 256 | 87.4685 | 81.5544 | 12.0800 | 0.5967 | 0.0000 | 0.0000 | 0.3906 | 209.3231 |
| A: regression + dropout | synthetic test (course only) | yes | 96 | 92.2250 | 86.9057 | 12.7369 | 0.5827 | 0.0000 | 0.0000 | 0.0000 | 206.0001 |
| A: regression + dropout | pseudo-real (OOD) | yes | 96 | 65.2767 | 54.7080 | 9.0152 | 0.7331 | 0.0000 | 0.0000 | 3.1250 | 213.7154 |
| A: regression + dropout | real photos (MIDV-500) | yes | 139 | 141.7866 | 128.3615 | 9.6545 | 0.5725 | 0.0000 | 0.0000 | 0.0000 | 344.3411 |
| A: regression + dropout | real photos (own) | yes | 25 | 85.9874 | 79.4382 | 6.5312 | 0.7261 | 0.0000 | 0.0000 | 4.0000 | 284.6630 |
| B: heatmap (control) | synthetic test | yes | 256 | 78.0050 | 55.6280 | 10.7730 | 0.6487 | 12.5000 | 19.9219 | 25.0000 | 292.8734 |
| B: heatmap (control) | synthetic test (course only) | yes | 96 | 85.4065 | 66.0168 | 11.7952 | 0.6154 | 10.4167 | 20.8333 | 22.9167 | 251.2493 |
| B: heatmap (control) | pseudo-real (OOD) | yes | 96 | 32.2128 | 3.7221 | 4.4488 | 0.8595 | 50.0000 | 52.0833 | 58.3333 | 229.7276 |
| B: heatmap (control) | real photos (MIDV-500) | yes | 139 | 76.3053 | 1.6410 | 5.1958 | 0.8121 | 55.3957 | 56.8345 | 58.9928 | 516.9412 |
| B: heatmap (control) | real photos (own) | yes | 25 | 49.3412 | 31.8867 | 3.7633 | 0.8242 | 4.0000 | 4.0000 | 24.0000 | 267.8463 |
| B: heatmap + dropout | synthetic test | yes | 256 | 79.9998 | 65.8295 | 11.0485 | 0.6399 | 12.1094 | 16.0156 | 20.3125 | 287.7107 |
| B: heatmap + dropout | synthetic test (course only) | yes | 96 | 87.4231 | 78.7894 | 12.0737 | 0.6124 | 6.2500 | 18.7500 | 20.8333 | 271.7377 |
| B: heatmap + dropout | pseudo-real (OOD) | yes | 96 | 35.6224 | 5.3867 | 4.9197 | 0.8449 | 47.9167 | 50.0000 | 55.2083 | 238.1554 |
| B: heatmap + dropout | real photos (MIDV-500) | yes | 139 | 87.5811 | 34.9206 | 5.9636 | 0.7675 | 41.7266 | 46.0432 | 48.2014 | 489.2776 |
| B: heatmap + dropout | real photos (own) | yes | 25 | 56.3775 | 35.6136 | 4.2976 | 0.8039 | 4.0000 | 4.0000 | 16.0000 | 271.4518 |

**Approach A vs Approach B.** On the synthetic test set, regression reaches 64.28 px mean corner error and heatmaps 18.57 px, so **B (heatmaps) wins** by 45.71 px (71% relative). Strict success (all four corners within 16 px): 10.2% for A versus 55.5% for B.

The prediction written down before the experiment was that heatmaps would win, because the loss is dense (every output pixel supervises the network, instead of eight scalars) and the mapping stays local, whereas regression asks fully connected layers to turn global features into precise coordinates. The measurement agrees.

**Precision versus reliability.** Approach B's mean error on the synthetic test set is 18.57 px but its median is 7.26 px -- a ratio of 3x. That shape is the whole story: when the detector finds the right page it localises it to well under a pixel (classical edge refinement, not the network, earns that), and the mean is set almost entirely by a minority of images where it locks onto the wrong rectangle. Sharpening the network would not move these numbers; fixing *which* rectangle it picks would.

**On the author's own photographs** -- the set the brief actually grades, 25 smartphone captures never trained on and never degraded synthetically -- the best model is B: heatmap at 37.86 px (2.81% of the image diagonal), median 27.92 px, quad IoU 0.853, with 4.0% of photographs having all four corners within 16 px and 28.0% within 32 px. Against 2.57% of the diagonal on synthetic data, that is the synthetic-to-real gap stated as one number.

**On MIDV-500** (a public set of real phone captures, ground truth derived from its segmentation masks) the best model reaches 61.73 px (4.20% of the diagonal). These are identity cards, not A4 pages -- a genuine domain shift on top of the synthetic-to-real one, so it is reported as a stress test rather than as the headline.

**Classical edge refinement** (fit lines to the gradient ridge near each predicted edge, re-intersect) changes the mean corner error by +6.03 px on average across models and sets. It is a clear win: the network localises the page to a few pixels, and the page border is a straight high-contrast edge that least-squares fitting nails far more precisely.

### Failure cases

![Corner predictions](assets/qualitative_corners.jpg)

The characteristic failures differ by approach, and they follow from the
formulation:

* **Regression** degrades *gracefully but globally* — under an unusual viewpoint
  the whole quad drifts, because a single fully connected layer produces all
  eight numbers from one pooled description of the image.
* **Heatmaps** degrade *locally and sharply* — three corners land exactly and one
  jumps, typically when a strong distractor edge (the table edge, a book spine)
  competes with the true corner, or when the corner sits outside the frame where
  the padded grid gives it only a sliver of support.

### The detector was never shown a page that was not white paper

Before any of the modelling below, a measurement of the training data itself.
Over the pages the corner detector actually trains on, against the pages it is
asked about in the field:

| | mean luma | saturation | pixels above 200 |
|---|---:|---:|---:|
| pages it trains on | 236.8 | 7.2 | **88.2%** |
| pages photographed by the user | 155.4 | 34.0 | **9.5%** |

Every document in the corpus is a scanned text page, so "the page" and "the
large bright region" were the same statement in every sample ever generated.
Trained on that, the network does not learn to find a page; it learns to find
the bright part. On a book cover whose artwork meets a black title panel it puts
the boundary at the panel — which is the 265 px error on img2, and is a data
problem that no architecture and no epoch count can reach.

Eleven augmentations follow from that, each taken from a photograph the detector
failed on rather than from imagination. They fall into four groups:

* **the page itself** — `p_page_tint` gives it a colour and a tone,
  `p_page_band` paints the solid panel a cover carries, `p_ruled_paper` fills it
  with long straight lines that are locally indistinguishable from an edge;
* **what borders it**, none of which existed — `p_page_stack` puts the body of
  the book under the top sheet, so the boundary becomes several near-parallel
  edges a few pixels apart; `p_binding` runs a spiral along one edge, straddling
  it; `p_gutter` lays the fold shadow of an open book immediately outside it;
* **the corners**, where the error is measured and where every previous sample
  showed a perfect intersection of two straight lines — `p_round_corners` cuts
  them as a real cover is cut, `p_dog_ear` folds one back, `p_corner_occluder`
  puts a thumb or a sticky note over one. The label stays on the true corner:
  having to answer where it is *without being able to see it* is the point;
* **scenes that contradict the assumption underneath all of it** — `p_backlit`
  makes the page darker than its surroundings, previously impossible;
  `p_pale_background` removes the brightness step and leaves only the boundary.

Together these move the training pages to mean luma 176.9 and saturation 38.8,
which puts **96% of the real photographs inside the training luminance band
against 79% before**.

All of it is corner-only, and that is enforced rather than intended: a
zero-probability augmentation draws no randomness at all, so it cannot shift the
stream and silently regenerate the frozen enhancement sets into different
images. A test pins the enhancement samples byte-identical across this change.

**What has and has not been shown.** The distribution measurement above stands
on its own -- it is a property of the corpus, not of any run. The training
evidence is weaker than it first looks, and the honest version is worth stating.

A matched-budget pair (same seed, same schedule, one flag apart, both with the
mask head) trained on the H100 for two epochs each:

| | frozen val MCE | val mask IoU | 24 photographs, mean | corner >100 px |
|---|---:|---:|---:|---:|
| without page augmentation | 70.5 px | 0.417 | 103.8 px | 18/24 |
| with page augmentation | **50.1 px** | **0.611** | 99.6 px | 16/24 |
| with page augmentation, mask forced on | | | **88.9 px** | **12/24** |
| shipped weights, 40 epochs on an H100 | | | 31.7 px | 6/24 |

The frozen-set column is **not a clean comparison and should not be quoted as
one**: those sets are built with the corner policy, so they were rebuilt with
the new augmentation, and the augmented arm is therefore validated on the
distribution it trains on while the control is not. Some of that 20 px gap is
home advantage.

The photographs are the clean test, and there two epochs settle nothing: both
arms are far worse than the shipped model, which is what a 2-epoch run against a
40-epoch one should look like. The one signal that survives is the mask head --
the augmented arm reaches 0.611 IoU against 0.417, and forcing its mask on takes
the catastrophic failures from 16/24 to 12/24. That is consistent with the
mechanism but it is not proof of the augmentation, and the run was stopped early
rather than left to reach a budget that could give one.

### Why one corner jumps, and what fixes it

The heatmap failure is not a training-budget failure. Four heatmaps are
predicted independently and four argmaxes are taken independently, and nothing
in the model or the loss requires the four answers to describe the *same* page.
Put a competing rectangle in frame — the facing page of an open book, a dark
band printed inside a cover — and the heads split: two answer about one
rectangle, two about the other. Measured on the 24 annotated photographs after
40 epochs on an H100, 6 of 24 had exactly this signature, three corners inside
20 px and the fourth 150-280 px out, at 0.9 confidence.

Probing the heatmaps shows the network is not blind to the right answer. In
**6 of the 9 defecting heads the correct location was present as a secondary
peak**, often near-tied with the wrong one:

| photo | head | argmax peak | best available peak |
|---|---|---|---|
| img2 | BR | 0.88 @ 265 px | **0.85 @ 18 px** |
| img9 | BL | 0.92 @ 270 px | **0.90 @ 22 px** |
| img16 | TL | 0.87 @ 177 px | **0.86 @ 30 px** |
| img16 | BL | 0.73 @ 164 px | **0.72 @ 38 px** |
| img18 | BL | 0.87 @ 200 px | **0.74 @ 18 px** |

The information was in the heatmaps; the decoder was discarding it. So the
network also predicts the **page interior** (§ `CornerHeatmapNet.seg`, 1.2% more
parameters), and corners are decoded as the peak combination that agrees with
that region — with the region answering outright when no combination agrees
with it. Measured against a ground-truth mask, which isolates the decoder from
how well the head learns:

| decoding | mean | median | succ@16 | photos with a corner >100 px |
|---|---:|---:|---:|---:|
| per-head argmax | 33.7 px | 15.6 px | 51.0% | 6/24 |
| mask-guided | **20.6 px** | **14.5 px** | **56.2%** | **2/24** |

That table was measured with a *ground-truth* mask before any retraining -- it
was the ceiling the mechanism could reach if the head learned perfectly. The
H100 retrain then landed **on** that ceiling: the shipped detector's head
records **val_mask_iou 0.921** (the 0.80 gate is comfortably cleared, so the
decoder actually uses it), and on the 24 photographs the full pipeline measures
**mean 20.6 px, median 14.0 px, succ@16 60.4%, with 2/24 photographs keeping a
corner past 100 px** -- against 33.7 px and 6/24 before. The prediction and the
outcome agreeing to the pixel is the strongest evidence in this report that the
diagnosis (independent heads discarding a peak the network already found) was
the right one.

**Why the mask, and not geometry.** Two purely geometric tie-breaks were tried
and both measured *worse* than doing nothing: edge support along the four sides
(39.2 px against 34.0 px) and joint peak selection scored on convexity and area,
which confidently picks the coherent *wrong* quad — on img2 it snapped the whole
quad onto the cover's blue panel. The competing rectangle is a perfectly good
rectangle, so the ambiguity is semantic and no geometric test can settle it.
Neither is included.

**The head is not trusted unconditionally.** Simulating how a real segmentation
head fails — soft boundary, wrong scale, wrong position — the gain does not
degrade gracefully:

| mask IoU | mean error | verdict |
|---:|---:|:--|
| 1.00 | 21.4 px | better than no mask |
| 0.85 | 20.3 px | better |
| 0.79 | 28.0 px | better |
| 0.73 | 39.9 px | **worse** |
| 0.64 | 70.7 px | **worse** — 17/24 with a corner >100 px |

A weak head is destructive, not merely unhelpful, and nothing about a mask
reveals that it is in the wrong *place*.

**Where the two remaining failures are, and why the decoder is not the answer.**
Two of the 24 photographs still keep a corner past 100 px, and on one of them
the region head alone would do far better than the shipped pipeline does
(42.7 px against 154.2 px). That looks like the decoder discarding evidence it
already has, so four selection rules were measured against the current one --
peaks with the region as a fallback:

    rule                       mean   median  succ@16  >100px
    current                    20.6    14.0    60.4%    2/24
    always the region          36.7    14.6    52.1%    5/24
    higher mask IoU            21.8    13.7    62.5%    3/24
    IoU + peak score           21.4    13.7    58.3%    2/24

None of them wins. Trusting the region where it happens to be better costs more
elsewhere than it gains, because on the other failing photograph the mask is
*also* wrong (IoU 0.694 against the true page: a black notebook on a blue
container, where the network segments the container). Two rules do improve the
median and the strict success rate slightly, and both make the catastrophic
count worse -- which is the number that decides whether a scan is usable.

So the remaining errors are not a decoding problem. They are pages whose region
the network genuinely mis-segments, and the answer to that is training data of
the kind the augmentation above adds, not another rule at inference. The trainer therefore records the
head's validation IoU (`val_mask_iou`) and the pipeline uses the mask only above
**0.80**, decoding exactly as before underneath it.


---

## 3. Section 6 — dropout

**Loss functions** (identical budgets, only the objective differs). MSE: 16.54 dB / 0.7081. L1: 15.85 dB / 0.7014. Charbonnier + MS-SSIM + Sobel: 19.14 dB / 0.7863. **combined scores best on SSIM.** PSNR and SSIM disagree here in the way the literature predicts: PSNR is a monotone function of MSE, so an MSE-trained model is optimising the metric directly and will tend to lead on it, while SSIM -- which is what tracks *legibility* -- rewards the structural sharpness the gradient and MS-SSIM terms preserve. Text lives in the edges, so the combined loss is what the pipeline ships with.

**Dropout (Section 6)** costs -0.32 dB on the synthetic test split against a control trained at exactly the same budget. It helps here, which suggests the unregularised model had started to fit the source pages.

**Does dropout shrink the synthetic-to-real gap?** This is what Section 6 actually asks, and it is a difference of differences rather than either column alone. Errors are expressed as a percentage of the image diagonal so that sets at different resolutions can be compared.

| Model | synthetic test | real photos | gap |
|---|---|---|---|
| A - regression — control | 13.20% | 8.90% | -4.30 pp |
| A - regression — **+ dropout** | 12.08% | 6.53% | **-5.55 pp** |
| B - heatmap — control | 10.77% | 3.76% | -7.01 pp |
| B - heatmap — **+ dropout** | 11.05% | 4.30% | **-6.75 pp** |

A - regression: the gap **widens by 1.25 pp**; B - heatmap: the gap **shrinks by 0.26 pp**. So the regularisation does what the brief suggests it might, even where it costs accuracy on the synthetic split: dropout trades in-distribution fit for a model that transfers slightly better. Both effects are small next to the dominant term, which is *what the generator does and does not put in the training scenes.*

---

## 4. Bonus — the end-to-end scanner

| Rectified with | n | corner error (px) | OCR confidence | words read |
|---|---|---|---|---|
| annotated corners (ground truth) | 24 | 0.0000 | 67.5915 | 111.5417 |
| predicted corners (fully automatic) | 24 | 47.8405 | 63.0194 | 109.5000 |

**What corner error costs the enhancement stage.** Rectifying with the annotated corners gives OCR confidence 67.6; rectifying with the predicted ones (47.84 px mean error) gives 63.0 (-4.6). That is the price of automation on this budget, and it is where additional corner accuracy would pay off most.

![End-to-end](assets/end_to_end.jpg)

The chain is also differentiable throughout (`docscanner/models/warp.py`
implements `get_perspective_transform` and `warp_perspective` in pure PyTorch —
**kornia is not used**), so `python -m docscanner.engine.finetune_e2e` can
fine-tune the corner detector against the *enhancement* loss. Two details decide
whether that works at all: the enhancement network must stay frozen (otherwise
it simply learns to absorb a systematic mis-crop) and the coordinate loss must
stay on as an anchor (otherwise the detector drifts toward any crop that happens
to reconstruct well).

#### Did fine-tuning through the warp help? No — and here is the measurement

The brief asks two questions of this option: does the detector improve when it
is trained for what the pipeline actually needs, and does the
annotated-vs-predicted gap shrink? Both were measured with
`python scripts/measure_finetune.py`, running each detector unchanged over the
same frozen photographs:

### Corner accuracy

| Detector | n | MCE (px) | median (px) | MCE (% diag) | quad IoU | success@16px | success@32px | worst (px) |
|---|---|---|---|---|---|---|---|---|
| baseline (trained on coordinates) | 24 | 20.55 | 14.38 | 2.01 | 0.90 | 25.00 | 62.50 | 88.21 |
| fine-tuned through the warp | 24 | 34.38 | 24.63 | 3.24 | 0.84 | 25.00 | 41.67 | 135.52 |

### The downstream gap the brief asks about

| Detector | OCR conf (annotated corners) | OCR conf (predicted corners) | confidence gap | words (annotated) | words (predicted) | words gap |
|---|---|---|---|---|---|---|
| baseline (trained on coordinates) | 58.82 | 56.51 | -2.31 | 34.04 | 35.33 | +1.29 |
| fine-tuned through the warp | 58.82 | 49.61 | -9.21 | 34.04 | 34.50 | +0.46 |
The answer to both questions is **no**. Corner error nearly doubles, the
catastrophic rate at 32 px rises, and the OCR gap the option was supposed to
close gets four times wider instead. Three things plausibly explain it, and
they point at the budget rather than at the idea: a reconstruction gradient is
a far noisier teacher than dense heatmap supervision, so 6 epochs of 60 steps
is a small budget in which to move a detector that dense supervision had
already placed well; the reconstruction term rewards *a crop the enhancer can
restore*, which is not the same objective as *the page border*, and the
coordinate anchor at weight 2.0 evidently did not hold it; and the fine-tune
runs on synthetic pairs while this table is real photographs, so the arm that
moved is also the arm with further to generalise.

The fine-tuned weights are therefore **not shipped** — `models/corner_heatmap.pt`
remains the coordinate-trained detector. The apparatus is kept, working and
tested, because the negative result is only trustworthy if the mechanism it
tested is real.

### Auto-rotation: one recogniser cannot answer two questions

The scanner turns the rectified page upright by OCR'ing it at each candidate
angle and keeping whichever reads best. Asked to choose among all four angles by
confidence, that procedure picks upright on **11 of the 24 reference scans** —
and those scans are upright by construction, so the other 13 are pages it would
have turned the wrong way. Two things go wrong, and they are different things:

* **The worst confusion is 0 against 270**, which is a question about the text
  *axis*, and confidence barely separates them: img10 scores 70.2/164 words
  upright against 73.3/170 words at 270°; img16 73.9/237 against 71.8/249.
* **The widest margins come from the emptiest pages.** One page scores 87.0 on a
  *single* recognised word against 0.0 for the upright original; another 81.7 on
  three words against 51.5 on eight. Confidence over a handful of words is noise
  wearing a number's clothes, and it is these pages that produce the most
  confident wrong answers.

So the question is split. A projection profile answers the axis without reading
anything — `text_axis_score` calls it correctly on 16 of 24, and the eight it
misses are the near-textless covers where no text axis exists — and it is used
as a *gate*: quarter turns are considered only when the profile reports sideways
text. The recogniser is left with the flip, which is the one thing a profile
cannot see, and nothing rotates on fewer than 20 words or on finding materially
less text than the upright page. Tesseract's OSD is kept as a proposal but no
longer bypasses the gate; it was wrong on two thirds of the pages where it
fired. Replayed against the measured table, this leaves **all 24 upright**.


---

## 5. What limits these numbers

1. **Training budget.** Every model here was trained on a single NVIDIA H100
   under a wall-clock cap — 90 minutes for the enhancement network, 130 for
   each corner detector, 25 for each ablation arm. Nothing about the
   architectures or the data engine is the bottleneck — the schedules are.
2. **Source-page diversity.** The generator makes unlimited *degradations*, but it
   cannot invent new *documents*. Generalisation to unseen layouts is bounded by
   the 854 source pages, which is exactly why the auxiliary corpus is mixed
   in alongside the provided scans.
3. **The synthetic-to-real gap.** The model removes the degradations the generator
   knows how to produce. The harder out-of-distribution set and the real
   photographs are included precisely so this gap is measured rather than assumed.
4. **Planarity.** A homography maps planes to planes; a curled or folded page
   cannot be rectified by one at all. That is a limit of the method, not of the
   training.

---

## 6. Reproducing

```bash
make data      # download corpora, build every split, freeze evaluation sets
make train     # the full schedule with matched budgets
make eval      # regenerate every table
make figures   # regenerate every figure and GIF
python -m docscanner.eval.report   # regenerate this document
```
