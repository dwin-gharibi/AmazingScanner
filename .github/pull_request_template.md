## What this changes

<!-- One paragraph. What is different afterwards, and why. -->

## Why

<!-- The problem, not the patch. If it fixes a bad scan, link the image. -->

## Evidence

<!--
Numbers, not adjectives. Anything touching the data pipeline, the networks, the
losses or the evaluation code changes the published metrics, so paste the
before/after from `amazingscanner eval --all`:

| Metric                        | Before | After |
|-------------------------------|--------|-------|
| Corner MCE, real photos (px)  |        |       |
| Enhancement PSNR (dB)         |        |       |
| End-to-end success @ 8 px     |        |       |

For a change that cannot move a number (docs, deployment, tooling), say so.
-->

## Checks

- [ ] `task verify` passes locally (lint, manifests, tests)
- [ ] Comparisons are at **matched budgets** — same epochs, steps and batch size
      on both arms, so the difference measured is the change under test
- [ ] No pre-trained weights, no pre-designed architectures, no third-party
      library doing a geometric transform (the brief allows OpenCV only)
- [ ] Real photographs are still **test data only** — nothing from the Roboflow
      set has entered a training split
- [ ] If the numbers moved: `amazingscanner refresh` re-run so the report,
      charts, figures and README describe the model this PR actually ships
