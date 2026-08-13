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
