| Run | loss | bg prior | dropout | n | PSNR | SSIM |
|---|---|---|---|---|---|---|
| degraded input (no model) |  |  |  | 24 | 12.3349 | 0.7110 |
| flagship (full budget) | combined | False | 0.0000 | 24 | 21.1640 | 0.8092 |
| control (combined loss, no dropout) | combined | True | 0.0000 | 24 | 19.1385 | 0.7863 |
| + dropout 0.15 (Section 6) | combined | True | 0.1500 | 24 | 19.4558 | 0.7906 |
| loss: MSE | mse | True | 0.0000 | 24 | 16.5371 | 0.7081 |
| loss: L1 | l1 | True | 0.0000 | 24 | 15.8510 | 0.7014 |