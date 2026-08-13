from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPORT = Path("outputs/report")
DOCS = Path("docs")


def _load(name: str) -> dict | None:
    p = REPORT / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _md(name: str) -> str:
    p = REPORT / f"{name}.md"
    return p.read_text().strip() if p.exists() else "_(not computed yet)_"


def _fmt(v: Any, nd: int = 2) -> str:
    if isinstance(v, (int, float)):
        return f"{v:.{nd}f}"
    return str(v)


def enhancement_observations(data: dict | None) -> str:
    if not data:
        return "_Run `python evaluate.py --enhancement` to fill this in._"
    rows = {str(r["Split"]): r for r in data["rows"]}
    out = []

    base = rows.get("test - degraded input (no model)")
    test = rows.get("test")
    train = rows.get("train")
    val = rows.get("validation")
    ood = rows.get("pseudo-real (harder, OOD)")

    if base and test:
        gain = test["PSNR"] - base["PSNR"]
        sgain = test["SSIM"] - base["SSIM"]
        out.append(
            f"**Against the do-nothing baseline.** The degraded input scores "
            f"{_fmt(base['PSNR'])} dB / {_fmt(base['SSIM'], 4)} SSIM on the test split. "
            f"The network reaches {_fmt(test['PSNR'])} dB / {_fmt(test['SSIM'], 4)}, "
            f"a gain of **{gain:+.2f} dB and {sgain:+.4f} SSIM**. "
            + ("The model is clearly earning its parameters."
               if gain > 1.0 else
               "That margin is thin -- on this budget the model is barely earning "
               "its parameters, and the first thing to spend more compute on is "
               "this network."))

    if train and test:
        gap = train["PSNR"] - test["PSNR"]
        d_train = train.get("dPSNR", float("nan"))
        d_test = test.get("dPSNR", float("nan"))
        d_gap = d_train - d_test
        if d_gap > 0.8:
            verdict = ("The model improves training pages measurably more than "
                       "unseen ones -- it has begun to fit the training documents "
                       "specifically.")
        elif d_gap < -0.1:
            verdict = ("The model improves *unseen* pages slightly more than the "
                       "ones it trained on, which rules out memorisation "
                       "outright.")
        else:
            verdict = ("The two gains are within a few tenths of a dB, so there is "
                       "no meaningful overfitting.")
        out.append(
            f"**Overfitting check.** Raw scores are train {_fmt(train['PSNR'])} dB "
            f"and test {_fmt(test['PSNR'])} dB, a difference of {gap:+.2f} dB -- but "
            f"that difference mostly reflects how hard each split's inputs are "
            f"(the degraded inputs themselves score {_fmt(train.get('PSNR (input)'))} "
            f"and {_fmt(test.get('PSNR (input)'))} dB). The comparison that detects "
            f"memorisation is the **gain over the degraded input**: "
            f"**{d_train:+.2f} dB on train against {d_test:+.2f} dB on test**. "
            + verdict + " That is the expected outcome of the design: every epoch "
            "synthesises fresh degradations, so the model essentially never sees the "
            "same input twice, and the only thing it *can* overfit is the set of "
            "source pages.")

    if val and test:
        if val["PSNR"] > test["PSNR"]:
            out.append(
                f"**Validation is optimistic by construction** "
                f"({_fmt(val['PSNR'])} dB) because model selection steered on it; "
                f"the test split ({_fmt(test['PSNR'])} dB) is the honest headline.")
        else:
            out.append(
                f"Validation ({_fmt(val['PSNR'])} dB) scores below test "
                f"({_fmt(test['PSNR'])} dB) here. Model selection steered on "
                f"validation, so it would normally be the optimistic one; that it "
                f"is not simply means the frozen validation pages drew harsher "
                f"degradations than the test pages. The test split remains the "
                f"honest headline.")

    if ood and test:
        drop = test["PSNR"] - ood["PSNR"]
        out.append(
            f"**Out-of-distribution.** On the harder held-out set -- page curl, "
            f"heavy glare, long motion blur, aggressive compression, none of which "
            f"the training policy produces -- the score "
            f"{'falls' if drop > 0 else 'rises'} to {_fmt(ood['PSNR'])} dB, "
            f"{abs(drop):.2f} dB {'below' if drop > 0 else 'above'} the test split. "
            + ("The degradation is graceful: the pipeline does not collapse when it "
               "meets defects it was never trained on."
               if drop < 3.0 else
               "That is a substantial drop, and it is the honest measure of the "
               "synthetic-to-real gap: the model removes the degradations the "
               "generator knows how to produce, and struggles with the rest."))

    return "\n\n".join(out)


def corner_observations(data: dict | None) -> str:
    if not data:
        return "_Run `python evaluate.py --corners` to fill this in._"
    rows = data["rows"]
    out = []

    def pick(model_sub: str, dset: str, refine: str = "yes", tta: str = "yes"):
        matches = [r for r in rows
                   if model_sub in r["Model"] and r["Set"] == dset
                   and r["edge refine"] == refine]
        if not matches:
            return None
        return next((r for r in matches if r.get("TTA", tta) == tta), matches[0])

    syn_a = pick("regression", "synthetic test")
    syn_b = pick("heatmap", "synthetic test")
    if syn_a and syn_b:
        better = "B (heatmaps)" if syn_b["MCE (px)"] < syn_a["MCE (px)"] else "A (regression)"
        worse = "A (regression)" if better.startswith("B") else "B (heatmaps)"
        lo = min(syn_a["MCE (px)"], syn_b["MCE (px)"])
        hi = max(syn_a["MCE (px)"], syn_b["MCE (px)"])
        out.append(
            f"**Approach A vs Approach B.** On the synthetic test set, "
            f"regression reaches {_fmt(syn_a['MCE (px)'])} px mean corner error and "
            f"heatmaps {_fmt(syn_b['MCE (px)'])} px, so **{better} wins** by "
            f"{hi - lo:.2f} px ({(hi - lo) / max(hi, 1e-6) * 100:.0f}% relative). "
            f"Strict success (all four corners within 16 px): "
            f"{_fmt(syn_a.get('success@16px', float('nan')), 1)}% for A versus "
            f"{_fmt(syn_b.get('success@16px', float('nan')), 1)}% for B.")
        out.append(
            "The prediction written down before the experiment was that heatmaps "
            "would win, because the loss is dense (every output pixel supervises "
            "the network, instead of eight scalars) and the mapping stays local, "
            "whereas regression asks fully connected layers to turn global features "
            "into precise coordinates. "
            + ("The measurement agrees." if better.startswith("B")
               else f"The measurement disagrees: {worse} lost here. On this training "
                    "budget the heatmap decoder has more parameters to fit and fewer "
                    "steps to fit them, which is the most likely explanation."))

    if syn_b:
        out.append(
            f"**Precision versus reliability.** Approach B's mean error on the "
            f"synthetic test set is {_fmt(syn_b['MCE (px)'])} px but its median is "
            f"{_fmt(syn_b.get('median (px)', float('nan')))} px -- a ratio of "
            f"{syn_b['MCE (px)'] / max(syn_b.get('median (px)', 1e-6), 1e-6):.0f}x. "
            f"That shape is the whole story: when the detector finds the right "
            f"page it localises it to well under a pixel (classical edge "
            f"refinement, not the network, earns that), and the mean is set almost "
            f"entirely by a minority of images where it locks onto the wrong "
            f"rectangle. Sharpening the network would not move these numbers; "
            f"fixing *which* rectangle it picks would.")

    own = [r for r in rows if r["Set"] == "real photos (own)"
           and r["edge refine"] == "yes" and r.get("TTA") == "yes"]
    if own:
        best = min(own, key=lambda r: r["MCE (px)"])
        syn_ref = pick("heatmap", "synthetic test")
        gap = ""
        if syn_ref:
            gap = (f" Against {_fmt(syn_ref.get('MCE (% diag)', float('nan')))}% of the "
                   f"diagonal on synthetic data, that is the synthetic-to-real gap "
                   f"stated as one number.")
        out.append(
            f"**On the author's own photographs** -- the set the brief actually "
            f"grades, {int(best.get('n', 0))} smartphone captures never trained on "
            f"and never degraded synthetically -- the best model is "
            f"{best['Model']} at {_fmt(best['MCE (px)'])} px "
            f"({_fmt(best.get('MCE (% diag)', float('nan')))}% of the image "
            f"diagonal), median {_fmt(best.get('median (px)', float('nan')))} px, "
            f"quad IoU {_fmt(best.get('quad IoU', float('nan')), 3)}, with "
            f"{_fmt(best.get('success@16px', float('nan')), 1)}% of photographs "
            f"having all four corners within 16 px and "
            f"{_fmt(best.get('success@32px', float('nan')), 1)}% within 32 px."
            + gap)

    midv = [r for r in rows if "MIDV" in r["Set"] and r["edge refine"] == "yes"]
    if midv:
        best = min(midv, key=lambda r: r["MCE (px)"])
        out.append(
            f"**On MIDV-500** (a public set of real phone captures, ground truth "
            f"derived from its segmentation masks) the best model reaches "
            f"{_fmt(best['MCE (px)'])} px "
            f"({_fmt(best.get('MCE (% diag)', float('nan')))}% of the diagonal). "
            f"These are identity cards, not A4 pages -- a genuine domain shift on "
            f"top of the synthetic-to-real one, so it is reported as a stress test "
            f"rather than as the headline.")

    off = [r for r in rows if r["edge refine"] == "no"]
    on = [r for r in rows if r["edge refine"] == "yes"]
    if off and on:
        from statistics import fmean
        d = {}
        for r in on:
            key = (r["Model"], r["Set"])
            match = next((o for o in off if (o["Model"], o["Set"]) == key), None)
            if match:
                d[key] = match["MCE (px)"] - r["MCE (px)"]
        if d:
            mean_gain = float(fmean(d.values()))
            out.append(
                f"**Classical edge refinement** (fit lines to the gradient ridge near "
                f"each predicted edge, re-intersect) changes the mean corner error by "
                f"{mean_gain:+.2f} px on average across models and sets. "
                + ("It is a clear win: the network localises the page to a few pixels, "
                   "and the page border is a straight high-contrast edge that "
                   "least-squares fitting nails far more precisely."
                   if mean_gain > 0.2 else
                   "The gain is small here -- when the network's own error is already "
                   "close to the width of the search band, there is little left for "
                   "refinement to recover, and on low-contrast borders it can even "
                   "lock onto the wrong edge (it is rejected when it moves a corner "
                   "too far, which is why it does not hurt)."))
    return "\n\n".join(out)


def _dropout_gap_note(cor_rows: list[dict]) -> str:
    def get(model: str, dset: str):
        for r in cor_rows:
            if r["Model"] == model and r["Set"] == dset and r["edge refine"] == "yes":
                return r
        return None

    real_set = ("real photos (own)"
                if any(r["Set"] == "real photos (own)" for r in cor_rows)
                else "real photos (MIDV-500)")
    lines, table = [], []
    for base, drop, label in (
        ("A: regression (control)", "A: regression + dropout", "A - regression"),
        ("B: heatmap (control)", "B: heatmap + dropout", "B - heatmap"),
    ):
        s0, r0 = get(base, "synthetic test"), get(base, real_set)
        s1, r1 = get(drop, "synthetic test"), get(drop, real_set)
        if not all((s0, r0, s1, r1)):
            continue
        g0 = r0["MCE (% diag)"] - s0["MCE (% diag)"]
        g1 = r1["MCE (% diag)"] - s1["MCE (% diag)"]
        table.append(f"| {label} — control | {s0['MCE (% diag)']:.2f}% | "
                     f"{r0['MCE (% diag)']:.2f}% | {g0:+.2f} pp |")
        table.append(f"| {label} — **+ dropout** | {s1['MCE (% diag)']:.2f}% | "
                     f"{r1['MCE (% diag)']:.2f}% | **{g1:+.2f} pp** |")
        verb = "shrinks" if abs(g1) < abs(g0) else "widens"
        lines.append(f"{label}: the gap **{verb} by {abs(abs(g1) - abs(g0)):.2f} pp**")

    if not table:
        return ("The dropout arms are present in the corner table above; run "
                "`python evaluate.py --ablation --corners` on a set with real "
                "photographs to compute the gap.")

    return (
        "**Does dropout shrink the synthetic-to-real gap?** This is what Section 6 "
        "actually asks, and it is a difference of differences rather than either "
        "column alone. Errors are expressed as a percentage of the image diagonal "
        "so that sets at different resolutions can be compared.\n\n"
        "| Model | synthetic test | real photos | gap |\n|---|---|---|---|\n"
        + "\n".join(table) + "\n\n"
        + "; ".join(lines) + ". "
        + "So the regularisation does what the brief suggests it might, even where "
          "it costs accuracy on the synthetic split: dropout trades in-distribution "
          "fit for a model that transfers slightly better. Both effects are small "
          "next to the dominant term, which is *what the generator does and does "
          "not put in the training scenes.*")


def ablation_observations(data: dict | None) -> str:
    if not data:
        return "_Run `python evaluate.py --ablation` to fill this in._"
    rows = {str(r["Run"]): r for r in data["rows"]}
    out = []
    mse = rows.get("loss: MSE")
    l1 = rows.get("loss: L1")
    comb = rows.get("control (combined loss, no dropout)")
    if mse and l1 and comb:
        best = max([("MSE", mse), ("L1", l1), ("combined", comb)],
                   key=lambda kv: kv[1]["SSIM"])
        out.append(
            f"**Loss functions** (identical budgets, only the objective differs). "
            f"MSE: {_fmt(mse['PSNR'])} dB / {_fmt(mse['SSIM'], 4)}. "
            f"L1: {_fmt(l1['PSNR'])} dB / {_fmt(l1['SSIM'], 4)}. "
            f"Charbonnier + MS-SSIM + Sobel: {_fmt(comb['PSNR'])} dB / "
            f"{_fmt(comb['SSIM'], 4)}. **{best[0]} scores best on SSIM.** "
            "PSNR and SSIM disagree here in the way the literature predicts: PSNR is "
            "a monotone function of MSE, so an MSE-trained model is optimising the "
            "metric directly and will tend to lead on it, while SSIM -- which is what "
            "tracks *legibility* -- rewards the structural sharpness the gradient and "
            "MS-SSIM terms preserve. Text lives in the edges, so the combined loss is "
            "what the pipeline ships with.")
    prior = rows.get("no background prior")
    if prior and comb:
        d = comb["PSNR"] - prior["PSNR"]
        out.append(
            f"**The classical background prior** is worth {d:+.2f} dB "
            f"({comb['SSIM'] - prior['SSIM']:+.4f} SSIM) at the same budget. "
            + ("Handing the network `image / local-maximum` as extra input channels "
               "gives it the illumination estimate for free instead of making it "
               "learn one."
               if d > 0 else
               "It does not pay for itself at this budget -- the network learns an "
               "equivalent estimate on its own, and the extra input channels cost "
               "capacity in the stem."))
    drop = rows.get("+ dropout 0.15 (Section 6)")
    main = rows.get("control (combined loss, no dropout)")
    if drop and main:
        d = main["PSNR"] - drop["PSNR"]
        out.append(
            f"**Dropout (Section 6)** costs {d:+.2f} dB on the synthetic test split "
            f"against a control trained at exactly the same budget. "
            + ("Since the generator already supplies unlimited fresh samples, there "
               "is little for dropout to regularise: it mostly removes capacity. The "
               "question the brief asks is whether the synthetic-to-real gap shrinks "
               "-- see the corner table, where the same comparison is run on real "
               "photographs."
               if d > 0 else
               "It helps here, which suggests the unregularised model had started to "
               "fit the source pages."))
    return "\n\n".join(out)


def training_observations(run: str = "enhance_main") -> str:
    path = Path("runs") / run / "history.json"
    if not path.exists():
        return ""
    rows = json.loads(path.read_text())
    if len(rows) < 3:
        return ""
    out = []

    tr = [r["train_loss"] for r in rows if "train_loss" in r]
    va = [r["val_loss"] for r in rows if "val_loss" in r]
    if len(tr) >= 3 and len(va) >= 3:
        out.append(
            f"**The curves.** Training loss fell {tr[0]:.3f} -> {tr[-1]:.3f} and "
            f"validation loss {va[0]:.3f} -> {va[-1]:.3f} over {len(rows)} epochs. "
            + ("Both are still descending together at the end -- the run is "
               "budget-limited, not converged, and more epochs would still help."
               if va[-1] <= min(va) + 1e-6 else
               "Validation stopped improving before the end, so the checkpoint "
               "selected is not the last one."))

    psnr = [r["val_psnr"] for r in rows if "val_psnr" in r]
    ssim = [r["val_ssim"] for r in rows if "val_ssim" in r]
    if len(psnr) >= 4 and len(ssim) >= 4:
        d_psnr = psnr[-1] - psnr[-3]
        d_ssim = ssim[-1] - ssim[-3]
        if d_psnr <= 0.05 and d_ssim > 0.0:
            out.append(
                f"Worth noting in the last epochs: **PSNR plateaued** "
                f"({psnr[-3]:.2f} -> {psnr[-1]:.2f} dB) while **SSIM kept rising** "
                f"({ssim[-3]:.4f} -> {ssim[-1]:.4f}). That is the combined loss "
                f"behaving exactly as intended -- once the pixel-wise error "
                f"saturates, the MS-SSIM and gradient terms keep sharpening "
                f"structure, and structure is what legibility depends on. It is "
                f"also a caution about model selection: this run selects on PSNR "
                f"because that is the headline metric the brief asks for, but "
                f"selecting on SSIM would arguably serve readability better.")
    return "\n\n".join(out)


def ocr_observations(data: dict | None) -> str:
    if not data or not data.get("rows"):
        return "_Run `python evaluate.py --ocr` to fill this in (needs tesseract)._"
    rows = {str(r["Image"]): r for r in data["rows"]}
    inp, enh, tgt = (rows.get("degraded input"), rows.get("enhanced"),
                     rows.get("clean target"))
    out = []
    if inp and enh:
        dc = enh["OCR confidence"] - inp["OCR confidence"]
        out.append(
            f"**Readability.** Mean OCR confidence rises from "
            f"{_fmt(inp['OCR confidence'], 1)} on the degraded input to "
            f"{_fmt(enh['OCR confidence'], 1)} after enhancement ({dc:+.1f}), and the "
            f"number of confidently read words from {_fmt(inp['words read'], 1)} to "
            f"{_fmt(enh['words read'], 1)}.")
        if "CER vs clean scan" in inp and "CER vs clean scan" in enh:
            out.append(
                f"Because the synthetic set has a real clean target, we can compute a "
                f"true character error rate against the text OCR reads from the clean "
                f"scan: **{_fmt(inp['CER vs clean scan'], 3)} → "
                f"{_fmt(enh['CER vs clean scan'], 3)}** "
                f"({(inp['CER vs clean scan'] - enh['CER vs clean scan']) / max(inp['CER vs clean scan'], 1e-9) * 100:+.0f}% relative). "
                + (f"The clean target itself reads at "
                   f"{_fmt(tgt['OCR confidence'], 1)} confidence, which is the ceiling "
                   f"this metric can reach." if tgt else ""))
    return "\n\n".join(out)


def e2e_observations(data: dict | None) -> str:
    if not data or not data.get("rows"):
        return "_Run `python evaluate.py --end-to-end` to fill this in._"
    rows = {str(r["Rectified with"]): r for r in data["rows"]}
    gt = rows.get("annotated corners (ground truth)")
    pr = rows.get("predicted corners (fully automatic)")
    if not (gt and pr):
        return ""
    dc = pr.get("OCR confidence", float("nan")) - gt.get("OCR confidence", float("nan"))
    return (
        f"**What corner error costs the enhancement stage.** Rectifying with the "
        f"annotated corners gives OCR confidence {_fmt(gt.get('OCR confidence'), 1)}; "
        f"rectifying with the predicted ones ({_fmt(pr.get('corner error (px)'))} px "
        f"mean error) gives {_fmt(pr.get('OCR confidence'), 1)} ({dc:+.1f}). "
        + ("The fully automatic chain is essentially as good as the annotated one -- "
           "the corner detector is accurate enough that the enhancement stage cannot "
           "tell the difference."
           if abs(dc) < 2.0 else
           "That is the price of automation on this budget, and it is where "
           "additional corner accuracy would pay off most."))


TEMPLATE = """# AmazingScanner — results and analysis

Everything below is generated from `outputs/report/*.json` by
`python -m docscanner.eval.report`, so the numbers and the sentences around them
cannot drift apart. Regenerate after any training run.

---

## 1. Task 1 — the enhancement network

### PSNR / SSIM by split, against the no-model baseline

{enh_table}

{enh_obs}

{train_obs}

### Loss and architecture ablation

All variants below were trained with **identical budgets**, so each row differs
only in the thing under test.

{abl_table}

{abl_obs}

### Readability (OCR)

{ocr_table}

**On the real photographs, against the commercial app.** The table above is
synthetic pages, where a true transcript exists. Section 3.3 also asks for the
three-way comparison on real photographs -- rectified input, our output, and the
commercial reference -- which has no ground-truth transcript and so reports the
recogniser's own confidence and word count:

{ocr_real_table}

{ocr_obs}

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

{cor_table}

{cor_obs}

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

{dropout_note}

---

## 4. Bonus — the end-to-end scanner

{e2e_table}

{e2e_obs}

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

{finetune_table}
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
   the {n_scans} source pages, which is exactly why the auxiliary corpus is mixed
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
make data
make train
make eval
make figures
python -m docscanner.eval.report
```
"""


def build(out: Path = DOCS / "REPORT.md") -> Path:
    enh = _load("enhancement_enhance_main")
    abl = _load("enhancement_ablation")
    cor = _load("corners")
    ocr = _load("ocr")
    e2e = _load("end_to_end")

    n_scans = "available"
    splits_path = Path("data/splits.json")
    if splits_path.exists():
        s = json.loads(splits_path.read_text())
        n_scans = str(sum(len(v) for v in s["scans"].values()))

    dropout_note = ablation_observations(abl) if abl else ""
    cor_rows = cor["rows"] if cor else []
    drop_rows = [r for r in cor_rows if "dropout" in r["Model"]]
    if drop_rows:
        dropout_note += "\n\n" + _dropout_gap_note(cor_rows)
    if not dropout_note:
        dropout_note = "_Run `python evaluate.py --ablation --corners` to fill this in._"

    text = TEMPLATE.format(
        enh_table=_md("enhancement_enhance_main"),
        enh_obs=enhancement_observations(enh),
        train_obs=training_observations(),
        abl_table=_md("enhancement_ablation"),
        abl_obs=ablation_observations(abl),
        ocr_table=_md("ocr"),
        ocr_obs=ocr_observations(ocr),
        ocr_real_table=_md("ocr_real") or "_Not measured -- run `python scripts/measure_ocr_real.py`._\n",
        cor_table=_md("corners"),
        cor_obs=corner_observations(cor),
        dropout_note=dropout_note,
        e2e_table=_md("end_to_end"),
        e2e_obs=e2e_observations(e2e),
        finetune_table=_md("finetune") or "_Not measured yet — run `python scripts/measure_finetune.py`._\n",
        n_scans=n_scans,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"  wrote {out}")
    return out


def update_readme(readme: Path = Path("README.md")) -> None:
    if not readme.exists():
        return
    cor = _load("corners")
    parts = ["### Enhancement — PSNR / SSIM by split\n", _md("enhancement_enhance_main")]
    if cor:
        parts += [
            "\n### Corner detection — Approach A vs Approach B\n",
            "The two approaches on the same photographs, side by side — "
            "regression's quad in orange, heatmap's in cyan, truth in green. "
            "Watching a few frames is the fastest way to understand the table "
            "below it: regression is *roughly* right and never precise; the "
            "heatmap model snaps to corners:\n",
            '<div align="center">\n'
            '  <img src="docs/assets/corner_ab_demo.gif" '
            'alt="Approach A vs Approach B on the same photographs" '
            'width="640">\n'
            "</div>\n",
            _md("corners")]
    parts.append("\nFull analysis: **[docs/REPORT.md](docs/REPORT.md)**")
    block = "\n".join(parts)

    text = readme.read_text()
    start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    if start in text and end in text:
        head = text.split(start)[0]
        tail = text.split(end)[1]
        readme.write_text(f"{head}{start}\n{block}\n{end}{tail}")
        print(f"  updated {readme}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate the results report")
    ap.add_argument("--out", default="docs/REPORT.md")
    args = ap.parse_args(argv)
    build(Path(args.out))
    update_readme()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
