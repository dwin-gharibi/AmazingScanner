| Model | Set | edge refine | TTA | n | MCE (px) | median (px) | MCE (% diag) | quad IoU | success@8px | success@16px | success@32px | worst (px) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A: regression (control) | synthetic test | yes | no | 256 | 95.5959 | 88.5299 | 13.2024 | 0.5785 | 0.0000 | 0.0000 | 0.3906 | 247.5996 |
| A: regression (control) | synthetic test (course only) | yes | no | 96 | 93.6617 | 81.3199 | 12.9353 | 0.5743 | 0.0000 | 0.0000 | 1.0417 | 234.6319 |
| A: regression (control) | pseudo-real (OOD) | yes | no | 96 | 69.1567 | 56.3071 | 9.5510 | 0.7162 | 2.0833 | 2.0833 | 2.0833 | 202.6281 |
| A: regression (control) | real photos (MIDV-500) | yes | no | 139 | 132.6105 | 107.3962 | 9.0297 | 0.5911 | 0.0000 | 0.0000 | 0.0000 | 389.4629 |
| A: regression (control) | real photos (own) | yes | no | 25 | 116.8807 | 112.4971 | 8.9046 | 0.6391 | 0.0000 | 0.0000 | 0.0000 | 233.0450 |
| A: regression + dropout | synthetic test | yes | no | 256 | 87.4685 | 81.5544 | 12.0800 | 0.5967 | 0.0000 | 0.0000 | 0.3906 | 209.3231 |
| A: regression + dropout | synthetic test (course only) | yes | no | 96 | 92.2250 | 86.9057 | 12.7369 | 0.5827 | 0.0000 | 0.0000 | 0.0000 | 206.0001 |
| A: regression + dropout | pseudo-real (OOD) | yes | no | 96 | 65.2767 | 54.7080 | 9.0152 | 0.7331 | 0.0000 | 0.0000 | 3.1250 | 213.7154 |
| A: regression + dropout | real photos (MIDV-500) | yes | no | 139 | 141.7866 | 128.3615 | 9.6545 | 0.5725 | 0.0000 | 0.0000 | 0.0000 | 344.3411 |
| A: regression + dropout | real photos (own) | yes | no | 25 | 85.9874 | 79.4382 | 6.5312 | 0.7261 | 0.0000 | 0.0000 | 4.0000 | 284.6630 |
| B: heatmap (control) | synthetic test | yes | no | 256 | 78.0050 | 55.6280 | 10.7730 | 0.6487 | 12.5000 | 19.9219 | 25.0000 | 292.8734 |
| B: heatmap (control) | synthetic test (course only) | yes | no | 96 | 85.4065 | 66.0168 | 11.7952 | 0.6154 | 10.4167 | 20.8333 | 22.9167 | 251.2493 |
| B: heatmap (control) | pseudo-real (OOD) | yes | no | 96 | 32.2128 | 3.7221 | 4.4488 | 0.8595 | 50.0000 | 52.0833 | 58.3333 | 229.7276 |
| B: heatmap (control) | real photos (MIDV-500) | yes | no | 139 | 76.3053 | 1.6410 | 5.1958 | 0.8121 | 55.3957 | 56.8345 | 58.9928 | 516.9412 |
| B: heatmap (control) | real photos (own) | yes | no | 25 | 49.3412 | 31.8867 | 3.7633 | 0.8242 | 4.0000 | 4.0000 | 24.0000 | 267.8463 |
| B: heatmap + dropout | synthetic test | yes | no | 256 | 79.9998 | 65.8295 | 11.0485 | 0.6399 | 12.1094 | 16.0156 | 20.3125 | 287.7107 |
| B: heatmap + dropout | synthetic test (course only) | yes | no | 96 | 87.4231 | 78.7894 | 12.0737 | 0.6124 | 6.2500 | 18.7500 | 20.8333 | 271.7377 |
| B: heatmap + dropout | pseudo-real (OOD) | yes | no | 96 | 35.6224 | 5.3867 | 4.9197 | 0.8449 | 47.9167 | 50.0000 | 55.2083 | 238.1554 |
| B: heatmap + dropout | real photos (MIDV-500) | yes | no | 139 | 87.5811 | 34.9206 | 5.9636 | 0.7675 | 41.7266 | 46.0432 | 48.2014 | 489.2776 |
| B: heatmap + dropout | real photos (own) | yes | no | 25 | 56.3775 | 35.6136 | 4.2976 | 0.8039 | 4.0000 | 4.0000 | 16.0000 | 271.4518 |