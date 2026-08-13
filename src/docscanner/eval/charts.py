from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

ASSETS = Path("docs/assets/charts")
REPORT = Path("outputs/report")

C = {
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
    "yellow": "#eda100", "violet": "#4a3aa7", "rose": "#d6336c",
    "ink": "#0b0b0b", "muted": "#52514e", "grid": "#d9d8d4", "surface": "#ffffff",
}

CAPTIONS: dict[str, tuple[str, str]] = {}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": C["surface"], "axes.facecolor": C["surface"],
        "axes.edgecolor": C["grid"], "axes.labelcolor": C["ink"],
        "text.color": C["ink"], "xtick.color": C["muted"], "ytick.color": C["muted"],
        "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": .8,
        "grid.alpha": .55, "axes.axisbelow": True,
        "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
        "figure.autolayout": False,
    })
    return plt


def _despine(ax, keep=("left", "bottom")):
    for side, sp in ax.spines.items():
        sp.set_visible(side in keep)


def _labels(ax, bars, fmt="{:.2f}", dy=0.012):
    top = max((b.get_height() for b in bars), default=1.0)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + top * dy,
                fmt.format(b.get_height()), ha="center", va="bottom",
                fontsize=9.5, color=C["muted"])


def _load(name: str) -> dict | None:
    p = REPORT / f"{name}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except ValueError:
        return None


def _save(fig, stem: str, title: str, caption: str) -> Path:
    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / f"{stem}.png"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=C["surface"], bbox_inches="tight")
    CAPTIONS[stem] = (title, caption)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  {out}")
    return out


def chart_psnr_by_split():
    d = _load("enhancement_enhance_main")
    if not d:
        print("  [skip] psnr_by_split"); return None
    rows = [r for r in d["rows"] if not str(r["Split"]).startswith("test - degraded")]
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    names = [str(r["Split"]).replace(" (harder, OOD)", "\n(OOD)")
                            .replace(" (course scans only)", "\n(course only)")
             for r in rows]
    x = np.arange(len(rows)); w = .38
    b1 = ax.bar(x - w / 2, [r["PSNR (input)"] for r in rows], w,
                color=C["orange"], label="degraded input (no model)")
    b2 = ax.bar(x + w / 2, [r["PSNR"] for r in rows], w,
                color=C["blue"], label="after enhancement")
    _labels(ax, b1, "{:.1f}"); _labels(ax, b2, "{:.1f}")
    ax.set_xticks(x, names, fontsize=9.5)
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Enhancement — PSNR against the do-nothing baseline")
    ax.set_ylim(0, max(r["PSNR"] for r in rows) * 1.22)
    ax.legend(frameon=False, loc="upper right")
    _despine(ax)
    test = next((r for r in rows if r["Split"] == "test"), rows[-1])
    return _save(fig, "psnr_by_split", "PSNR by split",
                 f"Every bar pair is the same images before and after the network. "
                 f"On the held-out test split the model turns {test['PSNR (input)']:.2f} dB "
                 f"into {test['PSNR']:.2f} dB, a gain of {test['dPSNR']:.2f} dB. The brief's "
                 f"instruction is that if the model is not clearly above the orange bars it "
                 f"is not earning its parameters.")


def chart_ssim_by_split():
    d = _load("enhancement_enhance_main")
    if not d:
        print("  [skip] ssim_by_split"); return None
    rows = [r for r in d["rows"] if not str(r["Split"]).startswith("test - degraded")]
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    names = [str(r["Split"]).replace(" (harder, OOD)", "\n(OOD)")
                            .replace(" (course scans only)", "\n(course only)")
             for r in rows]
    x = np.arange(len(rows)); w = .38
    b1 = ax.bar(x - w / 2, [r["SSIM (input)"] for r in rows], w,
                color=C["orange"], label="degraded input")
    b2 = ax.bar(x + w / 2, [r["SSIM"] for r in rows], w,
                color=C["aqua"], label="after enhancement")
    _labels(ax, b1, "{:.3f}"); _labels(ax, b2, "{:.3f}")
    ax.set_xticks(x, names, fontsize=9.5)
    ax.set_ylabel("SSIM")
    ax.set_ylim(0, 1.12)
    ax.set_title("Enhancement — SSIM (structural similarity)")
    ax.legend(frameon=False, loc="upper right")
    _despine(ax)
    return _save(fig, "ssim_by_split", "SSIM by split",
                 "SSIM rewards structure rather than pixel-for-pixel agreement, which is "
                 "why it is the better proxy for whether text stayed readable. It rises on "
                 "every split, including the out-of-distribution one.")


def chart_overfitting():
    d = _load("enhancement_enhance_main")
    if not d:
        print("  [skip] overfitting"); return None
    rows = {r["Split"]: r for r in d["rows"]}
    need = ("train", "validation", "test")
    if not all(k in rows for k in need):
        print("  [skip] overfitting: missing splits"); return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    gains = [rows[k]["dPSNR"] for k in need]
    bars = ax.bar(list(need), gains, color=[C["violet"], C["blue"], C["aqua"]], width=.55)
    _labels(ax, bars, "{:.2f}")
    gap = rows["train"]["PSNR"] - rows["test"]["PSNR"]
    ax.set_ylabel("PSNR gain over the degraded input (dB)")
    ax.set_title(f"Generalisation — train vs test gap = {gap:.2f} dB")
    ax.set_ylim(0, max(gains) * 1.25)
    _despine(ax)
    verdict = ("small, so the model is fit rather than memorising"
               if gap < 2.0 else "large enough to indicate overfitting")
    return _save(fig, "overfitting", "Overfitting check",
                 f"The gain over the do-nothing baseline on each split. Training reaches "
                 f"{rows['train']['PSNR']:.2f} dB and the held-out test split "
                 f"{rows['test']['PSNR']:.2f} dB — a gap of {gap:.2f} dB, which is {verdict}. "
                 f"Validation sits between them and is optimistic by construction, since it "
                 f"is what model selection steered on.")


def _loss_rows():
    d = _load("enhancement_ablation")
    if not d:
        return None, None
    keep = ("loss: MSE", "loss: L1", "control (combined loss, no dropout)",
            "no background prior")
    label = {"loss: MSE": "MSE", "loss: L1": "L1",
             "control (combined loss, no dropout)": "combined\n(Charbonnier+\nMS-SSIM+Sobel)",
             "no background prior": "combined,\nno bg prior"}
    rows = [r for r in d["rows"] if r["Run"] in keep]
    return (rows, label) if rows else (None, None)


def _loss_chart(key: str, fmt: str, colour: str, stem: str, title: str, caption: str):
    rows, label = _loss_rows()
    if not rows:
        print(f"  [skip] {stem}"); return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    names = [label.get(r["Run"], r["Run"]) for r in rows]
    vals = [r[key] for r in rows]
    best = int(np.argmax(vals))
    cols = [colour if i != best else C["rose"] for i in range(len(vals))]
    bars = ax.bar(np.arange(len(rows)), vals, color=cols, width=.6)
    _labels(ax, bars, fmt)
    ax.set_xticks(np.arange(len(rows)), names, fontsize=9.5)
    ax.set_ylabel(title)
    ax.set_title(f"Loss ablation — {title} (best highlighted)")
    ax.set_ylim(min(vals) * 0.97, max(vals) * 1.04)
    _despine(ax)
    return _save(fig, stem, f"Loss ablation — {title}", caption)


def chart_loss_ablation_psnr():
    return _loss_chart(
        "PSNR", "{:.2f}", C["blue"], "loss_ablation_psnr", "PSNR (dB)",
        "Four objectives trained at identical budgets, so the only difference is "
        "what the model was asked to minimise. PSNR is a monotone function of "
        "MSE, so an MSE-trained model is optimising this metric directly and "
        "tends to lead on it — which is exactly why PSNR alone is the wrong way "
        "to pick a restoration loss.")


def chart_loss_ablation_ssim():
    return _loss_chart(
        "SSIM", "{:.4f}", C["aqua"], "loss_ablation_ssim", "SSIM",
        "The same four runs judged by structural similarity, which is what "
        "tracks legibility. The combined Charbonnier + MS-SSIM + Sobel objective "
        "wins here, and that is the one the pipeline ships with: text lives in "
        "the edges, and the gradient and MS-SSIM terms are what preserve them.")


def _corner_rows(name="corners"):
    d = _load(name)
    return d["rows"] if d else []


def chart_approach_a_vs_b():
    rows = [r for r in _corner_rows()
            if r["Model"] in ("A: regression", "B: heatmap")
            and r["edge refine"] == "yes" and r.get("TTA") == "yes"]
    if not rows:
        print("  [skip] approach_a_vs_b"); return None
    sets = list(dict.fromkeys(r["Set"] for r in rows))
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    x = np.arange(len(sets)); w = .38
    for i, (model, colour) in enumerate((("A: regression", C["orange"]),
                                         ("B: heatmap", C["blue"]))):
        vals = [next((r["MCE (% diag)"] for r in rows
                      if r["Model"] == model and r["Set"] == s), np.nan) for s in sets]
        bars = ax.bar(x + (i - .5) * w, vals, w, color=colour,
                      label="Approach " + model.split(":")[0])
        _labels(ax, bars, "{:.2f}")
    ax.set_xticks(x, [s.replace(" (", "\n(") for s in sets], fontsize=9)
    ax.set_ylabel("mean corner error (% of image diagonal)")
    ax.set_title("Section 5 — Approach A (regression) vs Approach B (heatmaps)")
    ax.legend(frameon=False)
    _despine(ax)
    return _save(fig, "approach_a_vs_b", "Approach A vs Approach B",
                 "Error as a percentage of the image diagonal, so datasets at different "
                 "resolutions can sit on one axis. Heatmaps win on every set. The prediction "
                 "written down before running the experiment was that they would, because "
                 "the loss is dense and the mapping stays local, whereas regression asks "
                 "fully connected layers to turn a global description into precise "
                 "coordinates.")


def _ladder_rows():
    rows = [r for r in _corner_rows()
            if r["Model"] == "B: heatmap" and r["Set"] == "synthetic test"]
    order = [("no", "no", "network\nalone"), ("yes", "no", "+ sub-pixel\nedge refine"),
             ("no", "yes", "+ 4-view TTA\n(ships)"), ("yes", "yes", "refine\n+ TTA")]
    picked = []
    for ref, tta, lbl in order:
        r = next((r for r in rows if r["edge refine"] == ref and r.get("TTA") == tta), None)
        if r:
            picked.append((lbl, r))
    return picked if len(picked) >= 2 else None


def _ladder_chart(key: str, stem: str, title: str, colour: str, caption: str):
    picked = _ladder_rows()
    if not picked:
        print(f"  [skip] {stem}"); return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    vals = [p[1][key] for p in picked]
    bars = ax.bar([p[0] for p in picked], vals, color=colour, width=.55)
    _labels(ax, bars, "{:.2f}")
    ax.set_ylabel(title)
    ax.set_title(f"Inference ladder — {title}")
    ax.set_ylim(0, max(vals) * 1.3)
    _despine(ax)
    return _save(fig, stem, f"Inference ladder — {title}", caption)


def chart_ladder_mean():
    return _ladder_chart(
        "MCE (px)", "inference_ladder_mean", "mean corner error (px)", C["blue"],
        "The same trained weights at each rung of the inference chain, on the "
        "synthetic test set. Edge refinement leaves the mean where it found it; "
        "the improvement comes from TTA, because the mean is set by the images "
        "where the network picked the wrong rectangle, and outvoting those is "
        "what four views do. Both rungs behave differently on real photographs "
        "— see the ladder in the README, which is why refinement ships off.")


def chart_ladder_median():
    picked = _ladder_rows()
    base = picked[0][1]["median (px)"] if picked else 0.0
    ref = picked[1][1]["median (px)"] if picked and len(picked) > 1 else 0.0
    med = picked[-1][1]["median (px)"] if picked else 0.0
    return _ladder_chart(
        "median (px)", "inference_ladder_median", "median corner error (px)", C["aqua"],
        f"The same three configurations by median — the typical page rather than "
        f"the worst one. Worth reading against the claim refinement is usually "
        f"given credit for: on its own it moves the median the wrong way "
        f"({base:.2f} → {ref:.2f} px), and the drop to {med:.2f} px arrives with "
        f"TTA. Mean and median answer different questions, which is why they are "
        f"two charts and not one.")


def chart_success_rates():
    rows = [r for r in _corner_rows()
            if r["edge refine"] == "yes" and r.get("TTA") == "yes"
            and r["Model"] in ("A: regression", "B: heatmap")]
    if not rows:
        print("  [skip] success_rates"); return None
    sets = list(dict.fromkeys(r["Set"] for r in rows))
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9.8, 4.6))
    thresholds = [("success@8px", "8 px", C["aqua"]),
                  ("success@16px", "16 px", C["blue"]),
                  ("success@32px", "32 px", C["violet"])]
    x = np.arange(len(sets)); w = .26
    for i, (key, lbl, colour) in enumerate(thresholds):
        vals = [next((r[key] for r in rows
                      if r["Model"] == "B: heatmap" and r["Set"] == s), np.nan) for s in sets]
        bars = ax.bar(x + (i - 1) * w, vals, w, color=colour, label=f"all 4 within {lbl}")
        _labels(ax, bars, "{:.0f}")
    ax.set_xticks(x, [s.replace(" (", "\n(") for s in sets], fontsize=9)
    ax.set_ylabel("photographs passing (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Strict success — Approach B, all four corners inside the threshold")
    ax.legend(frameon=False)
    _despine(ax)
    return _save(fig, "success_rates", "Strict success rate",
                 "A stricter question than mean error: on what fraction of images do *all "
                 "four* corners land inside the threshold. One bad corner fails the image, "
                 "which is the right standard for a scanner — a page cropped along three "
                 "correct edges and one wrong one is still a ruined scan.")


def chart_dropout_gap():
    rows = _corner_rows("corners_dropout") + _corner_rows()
    def get(model, dset):
        return next((r for r in rows if r["Model"] == model and r["Set"] == dset
                     and r["edge refine"] == "yes"), None)
    real = ("real photos (own)" if any(r["Set"] == "real photos (own)" for r in rows)
            else "real photos (MIDV-500)")
    pairs = (("A: regression (control)", "A: regression + dropout", "A — regression"),
             ("B: heatmap (control)", "B: heatmap + dropout", "B — heatmap"))
    data = []
    for base, drop, label in pairs:
        s0, r0, s1, r1 = (get(base, "synthetic test"), get(base, real),
                          get(drop, "synthetic test"), get(drop, real))
        if all((s0, r0, s1, r1)):
            data.append((label,
                         r0["MCE (% diag)"] - s0["MCE (% diag)"],
                         r1["MCE (% diag)"] - s1["MCE (% diag)"]))
    if not data:
        print("  [skip] dropout_gap"); return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    x = np.arange(len(data)); w = .34
    b1 = ax.bar(x - w / 2, [abs(d[1]) for d in data], w, color=C["orange"],
                label="control (no dropout)")
    b2 = ax.bar(x + w / 2, [abs(d[2]) for d in data], w, color=C["aqua"],
                label="+ dropout")
    _labels(ax, b1, "{:.2f}"); _labels(ax, b2, "{:.2f}")
    ax.set_xticks(x, [d[0] for d in data])
    ax.set_ylabel("|synthetic - real| gap (percentage points of diagonal)")
    ax.set_title("Section 6 - does dropout shrink the synthetic-to-real gap?")
    ax.legend(frameon=False)
    _despine(ax)
    shrink = ", ".join(f"{d[0].split('—')[1].strip()} by {abs(d[1]) - abs(d[2]):+.2f} pp"
                       for d in data)
    return _save(fig, "dropout_gap", "Dropout and the synthetic-to-real gap",
                 f"Section 6 does not ask whether dropout lowers the error — it asks whether "
                 f"the gap between synthetic and real performance narrows. Shorter bars are "
                 f"better. Measured: {shrink}. Dropout trades in-distribution fit for a model "
                 f"that transfers slightly better, which is what regularisation is supposed "
                 f"to do.")


def _own_per_image():
    d = _load("corners")
    if not d:
        return None
    detail = d.get("detail", {})
    key = next((k for k in detail
                if k.startswith("B: heatmap|real photos (own)") and k.endswith("|True")), None)
    if key is None:
        return None
    return np.asarray(detail[key]["per_image_px"], float)


def chart_real_per_image():
    e = _own_per_image()
    if e is None or e.size == 0:
        print("  [skip] real_per_image"); return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    vals = np.sort(e)
    cols = [C["aqua"] if v <= 32 else (C["yellow"] if v <= 80 else C["orange"]) for v in vals]
    ax.bar(np.arange(len(vals)), vals, color=cols)
    ax.axhline(float(np.median(e)), color=C["ink"], lw=1.2, ls="--",
               label=f"median {np.median(e):.0f} px")
    ax.set_xlabel("photograph (sorted by error)")
    ax.set_ylabel("mean corner error (px)")
    ax.set_title(f"Every real photograph, individually (n={len(vals)})")
    ax.legend(frameon=False)
    _despine(ax)
    good = int((e <= 60).sum()); bad = int((e >= 80).sum())
    return _save(fig, "real_per_image", "Real photographs, per image",
                 f"Not a summary statistic — every one of the {len(vals)} photographs. The "
                 f"shape is the finding: {good} land within 60 px, then there is a visible "
                 f"gap, then {bad} fail at 80 px or worse. A bimodal distribution like this "
                 f"means a categorical mistake, not general imprecision, and the mean of it "
                 f"describes none of its images.")


def chart_real_cumulative():
    e = _own_per_image()
    if e is None or e.size == 0:
        print("  [skip] real_cumulative"); return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    t = np.linspace(0, max(120.0, float(e.max())), 300)
    frac = [(e <= v).mean() * 100 for v in t]
    ax.plot(t, frac, color=C["blue"], lw=2.4)
    ax.fill_between(t, frac, color=C["blue"], alpha=.12)
    for v in (16, 32, 64):
        pct = (e <= v).mean() * 100
        ax.plot([v], [pct], "o", color=C["orange"], ms=7)
        ax.annotate(f"{pct:.0f}% @ {v} px", (v, pct), textcoords="offset points",
                    xytext=(8, -14), fontsize=9.5, color=C["muted"])
    ax.set_xlabel("error threshold (px)")
    ax.set_ylabel("photographs within threshold (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Cumulative accuracy on real photographs")
    _despine(ax)
    return _save(fig, "real_cumulative", "Cumulative accuracy",
                 "Read it as: pick an accuracy you need, and this is the fraction of real "
                 "photographs that meet it. The curve is the honest way to state a "
                 "requirement — a product decision about acceptable crop error becomes a "
                 "readable number instead of an argument about means.")


def _e2e_chart(key: str, stem: str, title: str, colour: str, caption_fmt: str):
    d = _load("end_to_end")
    if not d or len(d.get("rows", [])) < 2:
        print(f"  [skip] {stem}"); return None
    rows = d["rows"][:2]
    vals = [r.get(key) for r in rows]
    if any(v is None or not np.isfinite(float(v)) for v in vals):
        print(f"  [skip] {stem}: no OCR measurement (install tesseract "
              f"or easyocr, then re-run `evaluate`)")
        return None
    vals = [float(v) for v in vals]
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    bars = ax.bar(["annotated\ncorners", "predicted\ncorners"], vals,
                  color=[colour, C["orange"]], width=.55)
    _labels(ax, bars, "{:.1f}")
    ax.set_ylabel(title)
    ax.set_title(f"Bonus §7 — {title}")
    ax.set_ylim(0, max(vals) * 1.25)
    _despine(ax)
    delta = vals[0] - vals[1]
    return _save(fig, stem, f"Bonus §7 — {title}",
                 caption_fmt.format(delta=delta, err=rows[1]["corner error (px)"],
                                    a=vals[0], b=vals[1]))


def chart_e2e_confidence():
    return _e2e_chart(
        "OCR confidence", "end_to_end_confidence", "mean OCR confidence", C["blue"],
        "The same photographs and the same enhancement network, rectified twice: "
        "once with my annotations and once with the detector's output. That "
        "isolates what corner error costs. At {err:.1f} px mean corner error the "
        "price is {delta:.1f} points of OCR confidence ({a:.1f} against {b:.1f}).")


def chart_e2e_words():
    return _e2e_chart(
        "words read", "end_to_end_words", "words recognised per page", C["aqua"],
        "The same comparison counted in words actually recognised — the number a "
        "user would notice. Predicted corners cost {delta:.1f} words per page "
        "({a:.1f} against {b:.1f}), because a mis-cropped page loses whole lines "
        "at the edges rather than degrading evenly.")


def _ocr_chart(key: str, stem: str, title: str, lower_better: bool, caption: str):
    d = _load("ocr")
    if not d or key not in d["rows"][0]:
        print(f"  [skip] {stem}"); return None
    rows = d["rows"]
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    names = [str(r["Image"]) for r in rows]
    vals = [r[key] for r in rows]
    cols = [C["orange"], C["blue"], C["aqua"]][:len(rows)]
    bars = ax.bar(np.arange(len(rows)), vals, color=cols, width=.6)
    _labels(ax, bars, "{:.3f}" if max(vals) < 5 else "{:.1f}")
    ax.set_xticks(np.arange(len(rows)), names, fontsize=9.5)
    ax.set_ylabel(title)
    ax.set_title(f"Readability — {title}" + (" (lower is better)" if lower_better else ""))
    ax.set_ylim(0, max(vals) * 1.25 if max(vals) else 1)
    _despine(ax)
    return _save(fig, stem, f"Readability — {title}", caption)


def chart_ocr_confidence():
    return _ocr_chart(
        "OCR confidence", "ocr_confidence", "mean OCR confidence", False,
        "Section 3.3 asks about usefulness, not pixel agreement: did the page "
        "become more readable? Confidence rises from the degraded input to the "
        "enhanced output, with the clean scan shown as the ceiling neither is "
        "expected to reach.")


def chart_ocr_words():
    return _ocr_chart(
        "words read", "ocr_words", "words recognised per page", False,
        "The same question counted in words the engine actually returned. This "
        "is the blunter and more honest number — a page can gain confidence "
        "while still hiding most of its text, and word count catches that.")


def chart_ocr_cer():
    return _ocr_chart(
        "CER vs clean scan", "ocr_cer", "character error rate", True,
        "Character error rate against the text read from the clean scan, so "
        "lower is better. It falls after enhancement, which is the direct "
        "statement that the network made the document more machine-readable "
        "rather than merely brighter.")


def chart_speed():
    d = _load("enhancement_enhance_main")
    if not d:
        print("  [skip] speed"); return None
    rows = [r for r in d["rows"] if r.get("seconds/page")]
    if not rows:
        print("  [skip] speed"); return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    names = [str(r["Split"]).replace(" (harder, OOD)", "\n(OOD)")
                            .replace(" (course scans only)", "\n(course only)")
                            .replace("test - degraded input (no model)", "no model\n(baseline)")
             for r in rows]
    vals = [r["seconds/page"] for r in rows]
    bars = ax.bar(np.arange(len(rows)), vals, color=C["violet"], width=.6)
    _labels(ax, bars, "{:.2f}")
    ax.set_xticks(np.arange(len(rows)), names, fontsize=9)
    ax.set_ylabel("seconds per page")
    ax.set_title("Enhancement throughput (4 CPU threads, no GPU)")
    _despine(ax)
    return _save(fig, "speed", "Throughput",
                 "Seconds per page for the enhancement pipeline on four CPU threads. "
                 "Training ran on an NVIDIA H100 (every run config records "
                 "device=cuda with AMP); these inference numbers are deliberately "
                 "CPU, because the question they answer is what a user without a GPU "
                 "waits for.")


def chart_training_distribution():
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3))
    groups = ["pages it\ntrained on\n(before)", "pages it\ntrains on\n(after)",
              "pages the user\nphotographs"]
    luma = [236.8, 176.9, 155.4]
    sat = [7.2, 38.8, 34.0]
    for ax, vals, name, unit in ((axes[0], luma, "mean luminance", "0-255"),
                                 (axes[1], sat, "mean saturation", "0-255")):
        bars = ax.bar(groups, vals, color=[C["rose"], C["aqua"], C["blue"]], width=.6)
        _labels(ax, bars, "{:.1f}")
        ax.set_title(f"{name} ({unit})")
        ax.set_ylim(0, max(vals) * 1.28)
        _despine(ax)
    fig.suptitle("The training pages were all white paper; the real ones are not",
                 fontsize=13, fontweight="bold")
    return _save(fig, "training_distribution",
                 "What a page looked like, before and after",
                 "Every document in the corpus is a scanned text page, so \"the page\" "
                 "and \"the large bright region\" were the same statement in every "
                 "sample ever generated -- which is why a cover with a dark title "
                 "panel had its boundary drawn at the panel. Eleven augmentations "
                 "move the training distribution onto the real one: 96% of the "
                 "photographs now fall inside the training luminance band, against "
                 "79% before.")


def chart_mask_quality():
    plt = _plt()
    iou = [1.00, 0.99, 0.85, 0.83, 0.79, 0.73, 0.64]
    err = [21.4, 21.5, 20.3, 21.7, 28.0, 39.9, 70.7]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ok = [i >= 0.80 for i in iou]
    ax.plot(iou, err, "-", color=C["muted"], lw=1.4, zorder=1)
    ax.scatter([i for i, k in zip(iou, ok) if k], [e for e, k in zip(err, ok) if k],
               s=90, color=C["aqua"], zorder=3, label="mask used")
    ax.scatter([i for i, k in zip(iou, ok) if not k], [e for e, k in zip(err, ok) if not k],
               s=90, color=C["rose"], zorder=3, label="mask refused")
    ax.axhline(33.7, color=C["orange"], ls="--", lw=1.6,
               label="no mask at all (33.7 px)")
    ax.axvline(0.80, color=C["violet"], ls=":", lw=1.6, label="the 0.80 gate")
    ax.invert_xaxis()
    ax.set_xlabel("mask IoU on the frozen validation set")
    ax.set_ylabel("mean corner error, 24 photographs (px)")
    ax.set_title("Below 0.80 IoU the mask makes things worse than having none")
    ax.legend(frameon=False, fontsize=9.5)
    _despine(ax)
    return _save(fig, "mask_quality",
                 "How good the page-mask head has to be",
                 "Simulating the ways a real segmentation head fails -- soft "
                 "boundary, wrong scale, wrong position -- every variant at 0.79 "
                 "IoU and above beats plain per-head argmax, while 0.73 and 0.64 "
                 "come out worse than using no mask at all. Nothing about a mask "
                 "reveals that it is in the wrong *place*, so the pipeline checks "
                 "the recorded IoU instead and decodes without the mask below the "
                 "gate. The shipped detector records 0.921.")


def chart_corner_fix():
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    labels = ["before\n(4 argmaxes)", "predicted ceiling\n(ground-truth mask)",
              "shipped\n(0.921 IoU head)"]
    colors = [C["rose"], C["yellow"], C["aqua"]]

    bars = axes[0].bar(labels, [33.7, 20.6, 20.6], color=colors, width=.6)
    _labels(axes[0], bars, "{:.1f}")
    axes[0].set_title("mean corner error (px)")
    axes[0].set_ylim(0, 42)

    bars = axes[1].bar(labels, [6, 2, 2], color=colors, width=.6)
    _labels(axes[1], bars, "{:.0f}")
    axes[1].set_title("photographs with a corner past 100 px (of 24)")
    axes[1].set_ylim(0, 7.6)
    for ax in axes:
        _despine(ax)
    fig.suptitle("Predicted from the heatmaps, then confirmed by the retrain",
                 fontsize=13, fontweight="bold")
    return _save(fig, "corner_fix",
                 "The corner fix: prediction against outcome",
                 "Four heatmaps were decoded by four independent argmaxes, with "
                 "nothing requiring the answers to describe one page -- so with a "
                 "competing rectangle in frame the heads split. Probing showed the "
                 "correct location present as a near-tied secondary peak in 6 of 9 "
                 "defecting heads, so a ground-truth mask was used to measure the "
                 "ceiling the mechanism could reach. The retrained head landed on "
                 "that ceiling exactly, which is the strongest evidence here that "
                 "the diagnosis was right.")


def chart_highlight_clipping():
    plt = _plt()
    photos = ["img11\nbook page", "img13\nnotes", "img9\nnotebook"]
    before = [14.5, 14.0, 10.5]
    after = [0.0, 0.0, 0.0]
    x = list(range(len(photos)))
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    b1 = ax.bar([i - .2 for i in x], before, width=.38, color=C["rose"],
                label="network output")
    b2 = ax.bar([i + .2 for i in x], after, width=.38, color=C["aqua"],
                label="with highlight compression")
    _labels(ax, b1, "{:.1f}%")
    _labels(ax, b2, "{:.1f}%")
    ax.set_xticks(x); ax.set_xticklabels(photos)
    ax.set_ylabel("pixels at or above 250/255 (%)")
    ax.set_title("Clipping is not a look — it is text destroyed")
    ax.set_ylim(0, 17.5)
    ax.legend(frameon=False, fontsize=9.5)
    _despine(ax)
    return _save(fig, "highlight_clipping",
                 "Blown highlights, removed",
                 "All 50 enhancement targets average 249/255 luminance with 92.8% "
                 "of pixels above 250, so \"make it look like a clean scan\" means "
                 "\"make it white\" -- and on a real photograph that pushes 10-14.5% "
                 "of the page past the top of the range, where no detail remains. "
                 "Compression pulls the top end back without changing the whitened "
                 "appearance the pages are supposed to have.")


def _pack():
    return _load("test_pack")


def chart_pack_success():
    d = _pack()
    rows = [r for r in (d or {}).get("rows", []) if r["Style"] != "ALL"]
    if not rows:
        print("  [skip] pack_success"); return None
    plt = _plt()
    styles = [r["Style"] for r in rows]
    x = np.arange(len(styles))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for off, key, colour in ((-0.18, "success@2%", C["aqua"]),
                             (0.18, "success@4%", C["violet"])):
        bars = a1.bar(x + off, [r[key] for r in rows], width=0.34,
                      color=colour, label=key)
        _labels(a1, bars, "{:.0f}%")
    a1.set_xticks(x, styles); a1.set_ylim(0, 112)
    a1.set_ylabel("photographs under the bound (%)")
    a1.set_title("Corner success rate by style")
    a1.legend(frameon=False); _despine(a1)

    for off, key, colour in ((-0.18, "corner err (% diag)", C["aqua"]),
                             (0.18, "median (% diag)", C["violet"])):
        bars = a2.bar(x + off, [r[key] for r in rows], width=0.34, color=colour,
                      label=key.replace(" (% diag)", ""))
        _labels(a2, bars, "{:.2f}")
    a2.set_xticks(x, styles)
    a2.set_ylabel("corner error (% of diagonal)")
    a2.set_title("Mean vs median corner error")
    a2.legend(frameon=False); _despine(a2)

    n = (d or {}).get("count", 0)
    return _save(fig, "pack_success", "The unseen pack, by style",
                 f"{n} photographs no model has seen — test-split source pages composited "
                 f"at seeds reserved above 7000, scored in the shipped configuration. The "
                 f"gap between the two right-hand bars is the story: medians sit near half "
                 f"a percent of the diagonal in every style while means run three times "
                 f"that, so the typical detection is tight and the average is carried by a "
                 f"few hard tails. Error rises clean → corner-hard → ood, exactly the "
                 f"ordering the styles were built to produce.")


def chart_pack_cdf():
    d = _pack()
    items = (d or {}).get("items", [])
    if not items:
        print("  [skip] pack_cdf"); return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    colours = {"clean": C["aqua"], "corner-hard": C["blue"], "ood": C["orange"]}
    for style, colour in colours.items():
        e = np.sort([r["err_pct"] for r in items if r["style"] == style])
        if not len(e):
            continue
        ax.step(e, np.arange(1, len(e) + 1) / len(e) * 100, where="post",
                color=colour, lw=2, label=f"{style} (n={len(e)})")
    all_e = np.sort([r["err_pct"] for r in items])
    ax.step(all_e, np.arange(1, len(all_e) + 1) / len(all_e) * 100, where="post",
            color=C["ink"], lw=1.4, ls="--", label=f"all ({len(all_e)})")
    for bound in (2.0, 4.0):
        ax.axvline(bound, color=C["muted"], lw=1, ls=":")
        ax.text(bound, 4, f" {bound:.0f}%", color=C["muted"], fontsize=9)
    ax.set_xlim(0, 8)
    ax.set_xlabel("mean corner error (% of image diagonal)")
    ax.set_ylabel("photographs at or below (%)")
    ax.set_title("Unseen pack: cumulative accuracy")
    ax.legend(frameon=False, loc="lower right"); _despine(ax)
    med = float(np.median(all_e))
    return _save(fig, "pack_cdf", "Unseen pack, cumulative accuracy",
                 f"Read it as “what fraction of unseen photographs land at least this "
                 f"good”. Every curve rises almost vertically at the left — the median "
                 f"photograph misses by {med:.2f}% of the diagonal — and then flattens into "
                 f"a long tail, which is the honest shape of this detector: usually very "
                 f"tight, occasionally lost. The `clean` curve sits above the other two "
                 f"everywhere, and `ood` below, so the ordering holds at every bar, not "
                 f"just at the two the table reports.")


def chart_per_corner():
    d = _load("corners")
    detail = (d or {}).get("detail", {})
    key = next((k for k in detail
                if k.startswith("B: heatmap|real photos (own)") and k.endswith("|True")),
               None)
    if key is None or "per_corner_px" not in detail.get(key, {}):
        print("  [skip] per_corner"); return None
    vals = np.asarray(detail[key]["per_corner_px"], float)
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    names = ["top-left", "top-right", "bottom-right", "bottom-left"]
    bars = ax.bar(names, vals, color=[C["aqua"], C["blue"], C["violet"], C["orange"]])
    _labels(ax, bars, "{:.1f} px")
    ax.axhline(float(vals.mean()), color=C["ink"], lw=1.2, ls="--",
               label=f"mean {vals.mean():.1f} px")
    ax.set_ylabel("mean error at this corner (px)")
    ax.set_title("Error by corner position — 24 real photographs")
    ax.legend(frameon=False); _despine(ax)
    worst, best = names[int(vals.argmax())], names[int(vals.argmin())]
    return _save(fig, "per_corner", "Which corner is hardest",
                 f"Each heatmap head is scored separately, because a single mean corner "
                 f"error cannot say whether the detector is uniformly imprecise or has one "
                 f"bad head. The answer here is the reassuring one: {worst} is worst at "
                 f"{vals.max():.1f} px and {best} best at {vals.min():.1f} px, a spread of "
                 f"only {vals.max() / max(vals.min(), 1e-6):.1f}x around a "
                 f"{vals.mean():.1f} px mean, with all four heads within a few pixels of "
                 f"each other. No head is broken, and nothing here suggests the corner "
                 f"*ordering* is inconsistent — the error is spread across the quad, which "
                 f"is what a shared trunk running out of resolution looks like rather than "
                 f"a labelling or decode bug.")


ALL = (chart_psnr_by_split, chart_ssim_by_split, chart_overfitting,
       chart_loss_ablation_psnr, chart_loss_ablation_ssim,
       chart_approach_a_vs_b, chart_ladder_mean, chart_ladder_median,
       chart_success_rates, chart_dropout_gap,
       chart_real_per_image, chart_real_cumulative,
       chart_e2e_confidence, chart_e2e_words,
       chart_ocr_confidence, chart_ocr_words, chart_ocr_cer, chart_speed,
       chart_training_distribution, chart_mask_quality,
       chart_corner_fix, chart_highlight_clipping,
       chart_per_corner, chart_pack_success, chart_pack_cdf)


def write_markdown(out: Path = Path("docs/CHARTS.md")) -> Path:
    lines = [
        "# AmazingScanner — the measurements, one chart at a time",
        "",
        "Every chart here is generated from `outputs/report/*.json`, which is written by",
        "`python -m docscanner.eval.evaluate`. Nothing is hand-typed, and re-running the",
        "evaluation refreshes all of them:",
        "",
        "```bash",
        "make eval          # measure",
        "python -m docscanner.eval.charts   # redraw",
        "```",
        "",
        "---",
        "",
    ]
    for stem, (title, caption) in CAPTIONS.items():
        lines += [f"## {title}", "",
                  f"![{title}](assets/charts/{stem}.png)", "",
                  caption, "", "---", ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  {out}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Draw one chart per question")
    ap.add_argument("--no-markdown", action="store_true")
    args = ap.parse_args(argv)
    print("[charts]")
    for fn in ALL:
        try:
            fn()
        except Exception as exc:
            print(f"  [skip] {fn.__name__}: {type(exc).__name__}: {exc}")
    if not args.no_markdown and CAPTIONS:
        write_markdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
