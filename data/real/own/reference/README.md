# Reference scans — drop `imgN_scanned.jpg` here

This directory is the **commercial-scanner baseline**: for each photograph in
`../photos/`, the same document as a scanner app produced it. Everything in the
end-to-end comparison that says "reference" reads from here.

It is empty. Nothing in this repository can regenerate these files — they are
the output of a different program on your originals — so if the comparison
column is blank, that is why.

## What to put here

One image per photograph, sharing its number:

```
photos/img4_main_jpg.rf.<hash>.jpg     ->  reference/img4_scanned.jpg
photos/img12_main_jpg.rf.<hash>.jpg    ->  reference/img12_scanned.jpg
```

The matching is by the number in the stem, so all of these pair with `img4`:

    img4_scanned.jpg   img4_scan.jpg   img4_ref.jpg
    img4_reference.jpg img4_camscanner.jpg   img4.jpg

Roboflow's export mangles filenames into `img4_main_jpg.rf.<sha>.jpg`; that is
handled — see `photo_key()` in `docscanner/data/real.py`.

## Then

```bash
amazingscanner data --own          # re-attach references to the manifest
amazingscanner eval --end-to-end   # the comparison, with the baseline column
amazingscanner stages              # per-photo, per-stage, against the reference
```

`stages` is the one worth looking at: it writes each photograph's input,
detected quad, rectification, enhancement, final output **and the reference**
side by side, so a disagreement can be attributed to a stage instead of argued
about.

## Why this directory is committed, empty, with a README

`.gitignore` used to cover it. That made the failure silent: a fresh clone had
no references, the end-to-end table quietly dropped its baseline column, and
nothing said the inputs were missing. Files you cannot regenerate do not belong
in `.gitignore` — the 50 course scans were ignored for the same reason and it
cost a training run.
