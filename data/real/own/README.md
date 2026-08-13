# Your own real photos

These photos are for **evaluation only** and are never trained on.

## The easy way: the built-in labelling tool

```bash
make label            # http://localhost:7861
```

Drag your photos in, press **Suggest corners**, correct anything that is off,
press **Next**. The tool writes `annotations.json` here after every click, so a
session can be interrupted and resumed. When you are done, press **Validate all
labels** for a contact sheet and a per-photo report.

That is the whole workflow — no external service, no export/import step.

## The alternative: annotate in Roboflow

If you would rather use a commercial annotator, the COCO-keypoints path still
works in both directions:

1. Put 10-15 phone photos of documents in `photos/`.
2. Annotate the four page corners in Roboflow (keypoint project, order:
   top-left, top-right, bottom-right, bottom-left), export as *COCO Keypoints*,
   and unzip it into `export/`.
3. Run:

       python -m docscanner.data.prepare --own

   which converts the export into `annotations.json`; every script and the
   Gradio app pick it up automatically.

The labelling tool exports the same format (**Export manifest + COCO**), so work
started here can be finished there, or reviewed there and brought back.

## Reference scans (optional, for the commercial baseline)

Put the matching scanner-app outputs (CamScanner / Adobe Scan / the built-in
phone scanner) in `reference/`, using the **same file stem** as the photo. The
OCR table then gains a column comparing this project against that app.

## Checking the labels

```bash
make check-labels
```

Verifies four finite points, convexity, plausible page coverage and aspect, and
whether the stored order matches canonical TL/TR/BR/BL — then writes a contact
sheet so 15 annotations can be reviewed at a glance. One page labelled in the
wrong order teaches the detector to rotate that page, and nothing downstream
will warn you, so this step is not optional.
