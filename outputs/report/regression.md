| Measurement | Baseline | This run | Change | |
|---|---:|---:|---:|:--|
| `corners.heatmap.real_own.mce_px` | 61.154 | 25.035 | -36.119 | 🟢 improved |
| `corners.heatmap.real_own.median_px` | 43.063 | 20.195 | -22.868 | 🟢 improved |
| `corners.heatmap.real_own.iou` | 0.757 | 0.886 | +0.129 | 🟢 improved |
| `corners.heatmap.synthetic.mce_px` | 14.478 | 18.507 | +4.029 | 🔴 regressed |
| `corners.heatmap.synthetic.median_px` | 0.574 | 7.170 | +6.597 | 🔴 regressed |
| `corners.heatmap.synthetic.iou` | 0.930 | 0.902 | -0.028 | ⚪ unchanged |
| `corners.heatmap.synthetic_course.mce_px` | 10.130 | 17.319 | +7.189 | 🔴 regressed |
| `corners.heatmap.synthetic_course.median_px` | 0.566 | 5.988 | +5.422 | 🔴 regressed |
| `corners.heatmap.synthetic_course.iou` | 0.954 | 0.906 | -0.048 | ⚪ unchanged |
| `corners.regression.real_own.mce_px` | 71.778 | 37.280 | -34.498 | 🟢 improved |
| `corners.regression.real_own.median_px` | 73.477 | 33.453 | -40.024 | 🟢 improved |
| `corners.regression.real_own.iou` | 0.724 | 0.838 | +0.114 | 🟢 improved |
| `corners.regression.synthetic.mce_px` | 34.467 | 35.589 | +1.122 | ⚪ unchanged |
| `corners.regression.synthetic.median_px` | 23.403 | 24.915 | +1.512 | ⚪ unchanged |
| `corners.regression.synthetic.iou` | 0.852 | 0.806 | -0.046 | ⚪ unchanged |
| `corners.regression.synthetic_course.mce_px` | 35.367 | 32.167 | -3.200 | 🟢 improved |
| `corners.regression.synthetic_course.median_px` | 26.222 | 21.479 | -4.743 | 🟢 improved |
| `corners.regression.synthetic_course.iou` | 0.856 | 0.821 | -0.034 | ⚪ unchanged |
| `end_to_end.corner_error_px` | 61.154 | 20.553 | -40.600 | 🟢 improved |
| `end_to_end.ocr_confidence` | 50.379 | 59.636 | +9.256 | 🟢 improved |

Bounds are ceilings with a 15% tolerance, held in `scripts/regression_bounds.json`.
Corner errors are in pixels on the frozen sets — the *set* is part of the number, since the same detector measures around a pixel on synthetic pages and tens of pixels on photographs.
