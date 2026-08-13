from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from ..data.datasets import FrozenCornerSet, FrozenPairSet, RealPhotoSet
from ..data.degrade import DegradationConfig
from ..utils.imageio import imread_rgb, imwrite_rgb
from ..utils.viz import (
    PALETTE,
    annotate,
    checkerboard_diff,
    difference_map,
    draw_corners,
    draw_quad,
    grid,
    hstack,
    label_bar,
    save_gif,
    side_by_side,
    vstack,
    wipe,
)

ASSETS = Path("docs/assets")
REPORT = Path("outputs/report")
RUNS = Path("runs")

C = {
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
    "yellow": "#eda100", "violet": "#4a3aa7",
    "ink": "#0b0b0b", "muted": "#52514e", "grid": "#d9d8d4", "surface": "#fcfcfb",
}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": C["surface"],
        "axes.facecolor": C["surface"],
        "axes.edgecolor": C["grid"],
        "axes.labelcolor": C["muted"],
        "axes.titlecolor": C["ink"],
        "axes.titleweight": "semibold",
        "axes.titlesize": 11,
        "axes.labelsize": 9.5,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": C["grid"],
        "grid.linewidth": 0.7,
        "xtick.color": C["muted"],
        "ytick.color": C["muted"],
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.size": 10,
        "figure.dpi": 150,
    })
    return plt


def _despine(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def _bar_labels(ax, bars, fmt="{:.2f}", dy=0.01):
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + span * dy, fmt.format(h),
                ha="center", va="bottom", fontsize=8.5, color=C["ink"])


def _letterbox(img: np.ndarray, w: int, h: int,
               fill: int = 22) -> np.ndarray:
    ih, iw = img.shape[:2]
    s = min(w / max(iw, 1), h / max(ih, 1))
    nw, nh = max(1, int(round(iw * s))), max(1, int(round(ih * s)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    if resized.ndim == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    canvas = np.full((h, w, 3), fill, np.uint8)
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized[..., :3]
    return canvas


def figure_dataset_stats(out: Path = ASSETS / "dataset_stats.png") -> Path | None:
    plt = _plt()
    splits_path = Path("data/splits.json")
    if not splits_path.exists():
        print("  [skip] dataset stats: no splits.json")
        return None
    splits = json.loads(splits_path.read_text())

    from ..data.prepare import make_generator
    stats = {}
    for task, cfg in (("corner detector", DegradationConfig.for_corners()),
                      ("enhancement", DegradationConfig.for_enhancement())):
        gen = make_generator("train", cfg=cfg, splits=splits, photo_long_side=420)
        ops, areas, rots = {}, [], []
        for i in range(90):
            s = gen.generate(np.random.default_rng((4242, i)), want_rectified=False)
            for name, _ in s.trace.ops:
                ops[name] = ops.get(name, 0) + 1
            q = s.corners
            h, w = s.photo.shape[:2]
            from ..utils.geometry import quad_area
            areas.append(quad_area(q) / (w * h))
            e = q[1] - q[0]
            rots.append(np.degrees(np.arctan2(e[1], e[0])))
        stats[task] = {"ops": ops, "areas": areas, "rots": rots}

    fig, axes = plt.subplots(1, 4, figsize=(17, 3.9))

    ax = axes[0]
    names = ["scans\ntrain", "scans\nval", "scans\ntest", "textures\ntrain", "textures\neval"]
    vals = [len(splits["scans"]["train"]), len(splits["scans"]["val"]),
            len(splits["scans"]["test"]), len(splits["backgrounds"]["train"]),
            len(splits["backgrounds"]["eval"])]
    colors = [C["blue"]] * 3 + [C["aqua"]] * 2
    bars = ax.bar(names, vals, color=colors, width=0.62)
    ax.set_title("Source corpora (split by document, and by texture family)")
    ax.set_ylabel("images")
    ax.set_ylim(0, max(vals) * 1.18)
    _bar_labels(ax, bars, "{:.0f}")
    _despine(ax)
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    all_ops = sorted({k for s in stats.values() for k in s["ops"]},
                     key=lambda k: -stats["enhancement"]["ops"].get(k, 0))
    all_ops = [o for o in all_ops if o not in ("warp_onto_background",)][:11]
    y = np.arange(len(all_ops))
    hgt = 0.38
    ax.barh(y + hgt / 2, [stats["enhancement"]["ops"].get(o, 0) / 90 * 100 for o in all_ops],
                 height=hgt, color=C["blue"], label="enhancement")
    ax.barh(y - hgt / 2, [stats["corner detector"]["ops"].get(o, 0) / 90 * 100 for o in all_ops],
                 height=hgt, color=C["orange"], label="corner detector")
    ax.set_yticks(y, [o.replace("_", " ") for o in all_ops], fontsize=8)
    ax.set_xlabel("% of samples")
    ax.set_title("Degradation frequency, per task policy")
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    _despine(ax, keep=("bottom",))
    ax.grid(axis="y", visible=False)

    ax = axes[2]
    for (task, s), col in zip(stats.items(), (C["orange"], C["blue"])):
        ax.hist(np.array(s["areas"]) * 100, bins=18, alpha=0.72, color=col, label=task)
    ax.set_xlabel("page area (% of frame)")
    ax.set_ylabel("samples")
    ax.set_title("How much of the frame the page fills")
    ax.legend()
    _despine(ax)
    ax.grid(axis="x", visible=False)

    ax = axes[3]
    for (task, s), col in zip(stats.items(), (C["orange"], C["blue"])):
        ax.hist(np.array(s["rots"]), bins=24, alpha=0.72, color=col, label=task)
    ax.set_xlabel("top-edge angle (deg)")
    ax.set_ylabel("samples")
    ax.set_title("In-plane rotation coverage")
    ax.legend()
    _despine(ax)
    ax.grid(axis="x", visible=False)

    fig.suptitle("Synthetic data: two tasks, two augmentation policies",
                 fontsize=13, fontweight="semibold", color=C["ink"], y=1.02)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")
    return out


def figure_training_curves(out: Path = ASSETS / "training_curves.png") -> Path | None:
    plt = _plt()
    runs = {p.parent.name: json.loads(p.read_text())
            for p in sorted(RUNS.glob("*/history.json"))}
    runs = {k: v for k, v in runs.items() if v}
    if not runs:
        print("  [skip] training curves: no history.json yet")
        return None

    enh = {k: v for k, v in runs.items() if "enhance" in k}
    cor = {k: v for k, v in runs.items() if "corner" in k}
    panels = [("Enhancement - loss", enh, "loss"),
              ("Enhancement - validation PSNR", enh, "val_psnr"),
              ("Corners - loss", cor, "loss"),
              ("Corners - validation corner error", cor, "val_mce_px")]
    panels = [p for p in panels if p[1]]
    if not panels:
        return None

    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.8))
    axes = np.atleast_1d(axes)
    cycle = [C["blue"], C["orange"], C["aqua"], C["yellow"], C["violet"]]

    for ax, (title, group, key) in zip(axes, panels):
        for i, (name, rows) in enumerate(sorted(group.items())):
            ep = [r["epoch"] for r in rows]
            col = cycle[i % len(cycle)]
            if key == "loss":
                if "train_loss" in rows[0]:
                    ax.plot(ep, [r["train_loss"] for r in rows], "-", lw=2,
                            color=col, label=f"{name} (train)")
                if "val_loss" in rows[0]:
                    ax.plot(ep, [r["val_loss"] for r in rows], "--", lw=1.8,
                            color=col, alpha=0.85, label=f"{name} (val)")
            elif key in rows[0]:
                ax.plot(ep, [r[key] for r in rows], "-o", lw=2, ms=3.5,
                        color=col, label=name)
        if key == "val_psnr" and enh:
            first = next(iter(enh.values()))
            if "val_psnr_input" in first[0]:
                base = float(np.mean([r["val_psnr_input"] for r in first]))
                ax.axhline(base, color=C["muted"], lw=1.2, ls=":",
                           label="degraded input (baseline)")
        ax.set_xlabel("epoch")
        ax.set_ylabel({"loss": "loss", "val_psnr": "PSNR (dB)",
                       "val_mce_px": "mean corner error (px)"}.get(key, key))
        ax.set_title(title)
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(fontsize=8)
        _despine(ax)
        ax.grid(axis="x", visible=False)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")
    return out


def figure_real_accuracy(out: Path = ASSETS / "real_accuracy.png") -> Path | None:
    plt = _plt()
    cor_path = REPORT / "corners.json"
    e2e_path = REPORT / "end_to_end.json"
    if not cor_path.exists():
        print("  [skip] real accuracy: run docscanner.eval.evaluate first")
        return None

    detail = json.loads(cor_path.read_text()).get("detail", {})
    key = next((k for k in detail
                if k.startswith("B: heatmap|real photos (own)") and k.endswith("|True")),
               None)
    if key is None:
        print("  [skip] real accuracy: no own-photo detail in corners.json")
        return None
    per_image = np.asarray(detail[key]["per_image_px"], float)
    if per_image.size == 0:
        return None

    has_e2e = e2e_path.exists()
    fig, axes = plt.subplots(1, 3 if has_e2e else 2,
                             figsize=(15.0 if has_e2e else 10.0, 4.2))
    axes = np.atleast_1d(axes)

    ax = axes[0]
    order = np.argsort(per_image)
    vals = per_image[order]
    colours = [C["aqua"] if v <= 32 else (C["yellow"] if v <= 80 else C["orange"])
               for v in vals]
    ax.bar(np.arange(len(vals)), vals, color=colours)
    ax.axhline(float(np.median(per_image)), color=C["ink"], lw=1.1, ls="--",
               label=f"median {np.median(per_image):.0f} px")
    ax.set_xlabel("photograph (sorted by error)")
    ax.set_ylabel("mean corner error (px)")
    ax.set_title(f"Real photographs, per image (n={len(vals)})")
    ax.legend(frameon=False)
    _despine(ax)

    ax = axes[1]
    thresholds = np.linspace(0, max(120.0, float(per_image.max())), 200)
    frac = [(per_image <= t).mean() * 100 for t in thresholds]
    ax.plot(thresholds, frac, color=C["blue"], lw=2.2)
    ax.fill_between(thresholds, frac, color=C["blue"], alpha=0.12)
    for t in (16, 32, 64):
        pct = (per_image <= t).mean() * 100
        ax.plot([t], [pct], "o", color=C["orange"], ms=6)
        ax.annotate(f"{pct:.0f}% @ {t}px", (t, pct), textcoords="offset points",
                    xytext=(6, -12), fontsize=9, color=C["muted"])
    ax.set_xlabel("error threshold (px)")
    ax.set_ylabel("photographs within threshold (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Cumulative accuracy")
    _despine(ax)

    if has_e2e:
        ax = axes[2]
        rows = json.loads(e2e_path.read_text())["rows"]
        names = ["annotated\ncorners", "predicted\ncorners"]
        conf = [r["OCR confidence"] for r in rows]
        words = [r["words read"] for r in rows]
        x = np.arange(2)
        w = 0.38
        b1 = ax.bar(x - w / 2, conf, w, color=C["blue"], label="OCR confidence")
        b2 = ax.bar(x + w / 2, words, w, color=C["aqua"], label="words read")
        _bar_labels(ax, b1, "{:.1f}")
        _bar_labels(ax, b2, "{:.1f}")
        ax.set_xticks(x, names)
        ax.set_title("What corner error costs (bonus §7)")
        ax.legend(frameon=False)
        _despine(ax)

    fig.suptitle("AmazingScanner on 24 real smartphone photographs",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=C["surface"])
    plt.close(fig)
    print(f"  {out}")
    return out


def figure_architecture(out: Path = ASSETS / "architecture.png") -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(15.5, 6.4))
    ax.set_xlim(0, 132); ax.set_ylim(-1.5, 46)
    ax.axis("off")

    def box(x, y, w, h, text, color, alpha=1.0, fs=8.2, tc="white"):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, alpha=alpha,
                                   edgecolor="none", zorder=2,
                                   linewidth=0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, zorder=3, fontweight="semibold")

    def arrow(x1, y1, x2, y2, color=C["muted"], style="-|>", lw=1.4, ls="-"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops={"arrowstyle": style, "color": color, "lw": lw,
                                "linestyle": ls, "shrinkA": 1, "shrinkB": 1}, zorder=1)

    ax.text(0, 44.4, "DocEnhanceNet  -  rectified page in, clean page out",
            fontsize=11.5, fontweight="700", color=C["ink"])

    ys = 30
    bw, gap = 8.5, 3.0
    enc = [("192²·32", 8), ("96²·64", 6.6), ("48²·128", 5.2), ("24²·192", 4.0)]
    dec = [("24²·192", 4.0), ("48²·128", 5.2), ("96²·64", 6.6), ("192²·32", 8)]

    box(1.0, ys - 2.0, 5.0, 4.0, "input\n+ bg\nprior", C["muted"], fs=6.6)
    x = 1.0 + 5.0 + gap
    enc_x = []
    prev = 6.0
    for lbl, h in enc:
        arrow(prev, ys, x, ys)
        box(x, ys - h / 2, bw, h, lbl, C["blue"])
        enc_x.append(x)
        prev = x + bw
        x += bw + gap

    arrow(prev, ys, x, ys)
    box(x, ys - 1.8, 12.0, 3.6, "12²·256\ndilated 1/2/4/8", C["violet"], fs=7.2)
    prev = x + 12.0
    x = prev + gap

    dec_x = []
    for lbl, h in dec:
        arrow(prev, ys, x, ys)
        box(x, ys - h / 2, bw, h, lbl, C["aqua"])
        dec_x.append(x)
        prev = x + bw
        x += bw + gap

    arrow(prev, ys, x, ys)
    box(x, ys - 2.2, 7.0, 4.4, "+ input\n(residual)", C["muted"], fs=6.6)

    for i in range(4):
        ex = enc_x[i] + bw / 2
        dx = dec_x[3 - i] + bw / 2
        top = ys + 6.0 + i * 1.7
        arrow(ex, ys + enc[i][1] / 2, ex, top, style="-", color=C["orange"], lw=1.1)
        arrow(ex, top, dx, top, style="-", color=C["orange"], lw=1.1)
        arrow(dx, top, dx, ys + dec[3 - i][1] / 2, color=C["orange"], lw=1.1)
    ax.text(enc_x[0] + bw / 2, ys + 6.0 + 3 * 1.7 + 1.4,
            "skip connections - thin strokes bypass the bottleneck",
            fontsize=8.5, color=C["orange"], fontweight="semibold")

    ax.text(0, 18.6, "Two corner detectors  -  raw photo in, four corners out",
            fontsize=11.5, fontweight="700", color=C["ink"])

    def chain(y, stages, head, head_label, second=None, second_label=None):
        box(1.0, y - 2.2, 8.0, 4.4, "photo\n256²", C["muted"], fs=7.6)
        prev = 9.0
        x = prev + 3.0
        for lbl, col in stages:
            arrow(prev, y, x, y)
            box(x, y - 2.0, 8.6, 4.0, lbl, col, fs=7.2)
            prev = x + 8.6
            x = prev + 3.0
        if second is None:
            arrow(prev, y, x, y)
            box(x, y - 2.6, 12.0, 5.2, head, C["yellow"], fs=7.4)
            arrow(x + 12.0, y, x + 14.5, y)
            ax.text(x + 15.0, y, head_label, fontsize=9, va="center",
                    color=C["ink"], fontweight="semibold")
            return

        hi, lo = y + 2.9, y - 2.9
        for yy, txt, col in ((hi, head, C["yellow"]), (lo, second, C["violet"])):
            arrow(prev, y, x, yy)
            box(x, yy - 2.3, 12.0, 4.6, txt, col, fs=7.0)
            arrow(x + 12.0, yy, x + 14.5, yy)
        ax.text(x + 15.0, hi, head_label, fontsize=9, va="center",
                color=C["ink"], fontweight="semibold")
        ax.text(x + 15.0, lo, second_label, fontsize=9, va="center",
                color=C["ink"], fontweight="semibold")

    chain(14.0,
          [(l, C["blue"]) for l in ["128²·24", "64²·48", "32²·96", "16²·192", "8²·256"]],
          "flatten\nFC 256-128-8", "A: 8 coordinates")
    chain(5.0,
          [(l, C["blue"]) for l in ["128²·24", "64²·48", "32²·96", "16²·192"]]
          + [(l, C["aqua"]) for l in ["32²·96", "64²·48"]],
          "4 heatmaps\nsoft-argmax", "B: 4 peaks",
          second="page mask\n64²·1  (sigmoid)", second_label="B: page region")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor=C["surface"])
    plt.close(fig)
    print(f"  {out}")
    return out


def _enhancer():
    from ..pipeline.enhance_pipeline import EnhancementPipeline
    for p in [Path("models/enhance.pt"), RUNS / "enhance_main" / "best.pt"]:
        if p.exists():
            return EnhancementPipeline(p, max_side=1400)
    return None


def _corner_pipe(name: str = "corner_heatmap"):
    from ..pipeline.corner_pipeline import CornerPipeline
    for p in [Path(f"models/{name}.pt"),
              RUNS / name / "best.pt"]:
        if p.exists():
            return CornerPipeline(p)
    return None


def figure_qualitative_enhance(n: int = 4,
                               out: Path = ASSETS / "qualitative_enhance.jpg") -> Path | None:
    pipe = _enhancer()
    if pipe is None:
        print("  [skip] qualitative enhance: no checkpoint")
        return None
    ds = FrozenPairSet(Path("data/frozen/enhance_test"))
    from .metrics import psnr, ssim
    rows = []
    for i in range(min(n, len(ds))):
        inp, tgt = ds.page(i)
        out_img = pipe(inp, mode="raw").network_output
        p0, p1 = psnr(inp, tgt), psnr(out_img, tgt)
        s0, s1 = ssim(inp, tgt), ssim(out_img, tgt)
        rows.append(side_by_side(
            [inp, out_img, tgt, difference_map(out_img, tgt)],
            [f"degraded input  ({p0:.1f} dB / {s0:.3f})",
             f"our output  ({p1:.1f} dB / {s1:.3f})",
             "clean target (ground truth)",
             "remaining error (x4)"], cell=330))
    img = vstack(rows)
    imwrite_rgb(out, img, quality=92)
    print(f"  {out}")
    return out


def figure_qualitative_corners(n: int = 6,
                               out: Path = ASSETS / "qualitative_corners.jpg") -> Path | None:
    heat, reg = _corner_pipe("corner_heatmap"), _corner_pipe("corner_regression")
    if heat is None and reg is None:
        print("  [skip] qualitative corners: no checkpoint")
        return None
    ds = FrozenCornerSet(Path("data/frozen/corners_test"), size=512)
    tiles = []
    for i in range(min(n, len(ds))):
        img, gt = ds.raw(i)
        p_heat = heat(img).corners if heat else None
        p_reg = reg(img).corners if reg else None
        tiles.append(draw_corners(img, gt=gt, pred=p_heat, pred2=p_reg,
                                  legend=("ground truth", "B: heatmap", "A: regression")))
    imwrite_rgb(out, grid(tiles, cols=3, cell=340), quality=92)
    print(f"  {out}")
    return out


def figure_end_to_end(n: int = 3, out: Path = ASSETS / "end_to_end.jpg") -> Path | None:
    from ..pipeline.scanner import DocumentScanner
    enh = Path("models/enhance.pt") if Path("models/enhance.pt").exists() \
        else RUNS / "enhance_main" / "best.pt"
    cor = Path("models/corner_heatmap.pt") if Path("models/corner_heatmap.pt").exists() \
        else RUNS / "corner_heatmap" / "best.pt"
    if not (enh.exists() and cor.exists()):
        print("  [skip] end-to-end: checkpoints missing")
        return None
    scanner = DocumentScanner(cor, enh)
    photos = sorted(ASSETS.glob("examples/*.jpg"))[:n]
    rows = []
    for p in photos:
        img = imread_rgb(p)
        res = scanner.scan(img)
        rows.append(side_by_side(
            [draw_quad(img, res.corners, PALETTE["pred"]), res.rectified, res.image],
            ["1. detected page", "2. rectified", "3. enhanced"], cell=360))
    if not rows:
        return None
    imwrite_rgb(out, vstack(rows), quality=92)
    print(f"  {out}")
    return out


def _own_photos() -> RealPhotoSet | None:
    manifest = Path("data/real/own/annotations.json")
    if not manifest.exists():
        print("  [skip] no own real photos")
        return None
    return RealPhotoSet(manifest)


def figure_real_end_to_end(n: int = 6, start: int = 0,
                           out: Path = ASSETS / "real_end_to_end.jpg") -> Path | None:
    from ..pipeline.scanner import DocumentScanner

    ds = _own_photos()
    if ds is None:
        return None
    enh = Path("models/enhance.pt")
    cor = Path("models/corner_heatmap.pt")
    if not (enh.exists() and cor.exists()):
        print("  [skip] real end-to-end: checkpoints missing")
        return None
    scanner = DocumentScanner(cor, enh, max_side=900)

    rows = []
    for i in range(start, min(start + n, len(ds))):
        img, gt, _ = ds.raw(i)
        res = scanner.scan(img)
        err = float(np.linalg.norm(res.corners - gt, axis=1).mean())
        overlay = draw_corners(img, gt=gt, pred=res.corners,
                               legend=("annotated", "predicted", ""))
        worst = float(np.linalg.norm(res.corners - gt, axis=1).max())
        src = res.meta.get("corner_source", "network")
        rows.append(side_by_side(
            [img, overlay, res.rectified, res.image],
            [f"raw photograph  ({img.shape[1]}x{img.shape[0]})",
             f"detected  mean {err:.0f} px / worst {worst:.0f} px  [{src}]",
             f"rectified  ({res.rectified.shape[1]}x{res.rectified.shape[0]})",
             f"enhanced  {res.seconds:.1f}s"
             + (f", turned {res.rotation_applied} deg" if res.rotation_applied else "")],
            cell=300))
    if not rows:
        return None
    imwrite_rgb(out, vstack(rows), quality=92)
    print(f"  {out}")
    return out


def figure_real_enhance_pairs(n: int = 8,
                              out: Path = ASSETS / "real_enhance_pairs.jpg") -> Path | None:
    from ..pipeline.scanner import rectify_with_corners

    ds = _own_photos()
    pipe = _enhancer()
    if ds is None or pipe is None:
        return None
    tiles = []
    step = max(1, len(ds) // n)
    for i in range(0, min(len(ds), n * step), step):
        img, gt, _ = ds.raw(i)
        rect, _ = rectify_with_corners(img, gt, max_side=760)
        res = pipe(rect, mode="color")
        outp = res.image

        def _tone(x):
            g = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
            return (float(g.mean()),
                    float(cv2.cvtColor(x, cv2.COLOR_RGB2HSV)[..., 1].mean()),
                    float((g > 250).mean() * 100))

        a, b = _tone(rect), _tone(outp)
        tiles.append(side_by_side(
            [rect, outp],
            [f"rectified photo   luma {a[0]:.0f}  sat {a[1]:.0f}  blown {a[2]:.1f}%",
             f"enhanced   luma {b[0]:.0f}  sat {b[1]:.0f}  blown {b[2]:.1f}%"],
            cell=300))
    if not tiles:
        return None
    imwrite_rgb(out, grid(tiles[:n], cols=2, cell=620), quality=92)
    print(f"  {out}")
    return out


def figure_real_gallery(cols: int = 6,
                        out: Path = ASSETS / "real_gallery.jpg") -> Path | None:
    ds = _own_photos()
    heat = _corner_pipe("corner_heatmap")
    if ds is None or heat is None:
        return None
    tiles = []
    for i in range(len(ds)):
        img, gt, _ = ds.raw(i)
        pred = heat(img).corners
        err = float(np.linalg.norm(pred - gt, axis=1).mean())
        diag = float(np.hypot(*img.shape[:2]))
        tiles.append(annotate(draw_corners(img, gt=gt, pred=pred),
                              f"{err:.0f}px ({100 * err / diag:.1f}% diag)", (10, 26)))
    imwrite_rgb(out, grid(tiles, cols=cols, cell=290), quality=90)
    print(f"  {out}")
    return out


def figure_network_choice(out: Path = ASSETS / "network_choice.jpg") -> Path | None:
    from ..pipeline.enhance_pipeline import EnhancementPipeline
    from ..utils.geometry import order_corners, rectify
    shipped_p = Path("models/enhance.pt")
    ds = _own_photos()
    if ds is None or not shipped_p.exists():
        print("  [skip] network_choice: need the shipped checkpoint and the photos")
        return None

    import subprocess
    import tempfile
    got = subprocess.run(["git", "show", "51d5b7d:models/enhance.pt"],
                         capture_output=True)
    if got.returncode != 0 or not got.stdout:
        print("  [skip] network_choice: retrain checkpoint not in git history "
              "(shallow clone?)")
        return None
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp.write(got.stdout)
    shipped = EnhancementPipeline(shipped_p, max_side=1000)
    retrain = EnhancementPipeline(Path(tmp.name), max_side=1000)

    def grey_pct(rgb: np.ndarray) -> float:
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return float(((g >= 120) & (g <= 220)).mean() * 100)

    rows = []

    darkness = []
    for idx in range(len(ds.items)):
        img, gt, _ = ds.raw(idx)
        page = rectify(img, order_corners(gt.astype(np.float32)), max_side=320)[0]
        darkness.append((float(cv2.cvtColor(page, cv2.COLOR_RGB2GRAY).mean()), idx))
    for _, idx in sorted(darkness)[:2]:
        img, gt, _ = ds.raw(idx)
        page = rectify(img, order_corners(gt.astype(np.float32)), max_side=1000)[0]
        a = retrain(page, mode="color").image
        b = shipped(page, mode="color").image
        tiles = [label_bar(page, "rectified input"),
                 label_bar(a, f"the retrain (higher PSNR)   {grey_pct(a):.0f}% grey"),
                 label_bar(b, f"shipped (pinned)   {grey_pct(b):.0f}% grey")]
        hmax = max(t.shape[0] for t in tiles)
        rows.append(np.concatenate(
            [cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 6,
                                cv2.BORDER_CONSTANT, value=(255, 255, 255))
             for t in tiles], axis=1))
    wmax = max(r.shape[1] for r in rows)
    sheet = np.concatenate(
        [cv2.copyMakeBorder(r, 0, 8, 0, wmax - r.shape[1],
                            cv2.BORDER_CONSTANT, value=(255, 255, 255))
         for r in rows], axis=0)
    if sheet.shape[1] > 1900:
        sheet = cv2.resize(sheet, (1900, int(sheet.shape[0] * 1900 / sheet.shape[1])))
    imwrite_rgb(out, sheet, quality=92)
    print(f"  {out}")
    return out


def figure_realism_check(n: int = 6,
                         out: Path = ASSETS / "realism_check.jpg") -> Path | None:

    ds = _own_photos()
    if ds is None:
        return None
    try:
        from ..data.degrade import DegradationConfig
        from ..data.prepare import load_splits, make_generator
        gen = make_generator("test", cfg=DegradationConfig.for_corners(),
                             splits=load_splits(), photo_long_side=760)
    except Exception as exc:
        print(f"  [skip] realism check: {type(exc).__name__}: {exc}")
        return None

    tiles: list[tuple[np.ndarray, str]] = []
    for i in range(n):
        s = gen.generate(np.random.default_rng(4000 + i), want_rectified=False)
        tiles.append((s.photo, "synthetic"))
    step = max(1, len(ds) // n)
    for k in range(0, min(len(ds), n * step), step):
        tiles.append((ds.raw(k)[0], "real"))

    order = np.random.default_rng(7).permutation(len(tiles))
    shuffled = [tiles[i] for i in order]
    grid_img = grid([t for t, _ in shuffled], cols=4, cell=300)
    key = "  ".join(f"{i + 1}:{lbl[0].upper()}" for i, (_, lbl) in enumerate(shuffled))
    imwrite_rgb(out, grid_img, quality=92)
    print(f"  {out}   (answer key, S=synthetic R=real: {key})")
    (out.with_suffix(".key.txt")).write_text(key + "\n", encoding="utf-8")
    return out


def figure_real_output_modes(index: int = 0,
                             out: Path = ASSETS / "real_output_modes.jpg") -> Path | None:
    from ..pipeline.enhance_pipeline import apply_output_mode
    from ..pipeline.scanner import rectify_with_corners

    ds = _own_photos()
    pipe = _enhancer()
    if ds is None or pipe is None:
        return None
    img, gt, _ = ds.raw(min(index, len(ds) - 1))
    rect, _ = rectify_with_corners(img, gt, max_side=820)
    base = pipe(rect, mode="raw").network_output
    panels = [rect] + [apply_output_mode(base, m)
                       for m in ("color", "gray", "bw", "whiteboard")]
    titles = ["rectified photo", "color", "grayscale", "black & white", "whiteboard"]
    imwrite_rgb(out, side_by_side(panels, titles, cell=300), quality=92)
    print(f"  {out}")
    return out


def figure_real_failures(n: int = 4,
                         out: Path = ASSETS / "real_failures.jpg") -> Path | None:
    ds = _own_photos()
    heat = _corner_pipe("corner_heatmap")
    if ds is None or heat is None:
        return None
    scored = []
    for i in range(len(ds)):
        img, gt, _ = ds.raw(i)
        pred = heat(img).corners
        err = float(np.linalg.norm(pred - gt, axis=1).mean())
        scored.append((err, i, img, gt, pred))
    scored.sort(key=lambda t: -t[0])
    tiles = []
    for rank, (e, i, im, g, p) in enumerate(scored[:n]):
        worst = float(np.linalg.norm(p - g, axis=1).max())
        panel = draw_corners(im, gt=g, pred=p,
                             legend=("annotated", "predicted", "")
                             if rank == 0 else None)
        tiles.append(label_bar(panel,
                               f"#{i}   mean {e:.0f} px   worst corner {worst:.0f} px"))
    imwrite_rgb(out, grid(tiles, cols=2, cell=470), quality=92)
    print(f"  {out}")
    return out


def _verify_generator(policy: str = "corners", source: str | None = None):
    from ..data.degrade import DegradationConfig
    from ..data.prepare import load_splits, make_generator
    try:
        splits = load_splits()
    except Exception:
        return None
    cfg = (DegradationConfig.for_corners() if policy == "corners"
           else DegradationConfig.for_enhancement())
    kw = {"source": source} if source else {}
    return make_generator("train", cfg=cfg, splits=splits, photo_long_side=760, **kw)


def figure_verify_pairs(n: int = 4, out: Path = ASSETS / "verify_pairs.jpg") -> Path | None:
    gen = _verify_generator("enhancement")
    if gen is None:
        return None
    rng = np.random.default_rng(3)
    rows = []
    for _ in range(n):
        s = gen.generate(rng, want_rectified=True)
        rows.append(side_by_side(
            [s.photo, s.rectified, s.target],
            ["composited photograph", "rectified input", "clean target"], cell=300))
    imwrite_rgb(out, vstack(rows, gap=10), quality=92)
    print(f"  {out}")
    return out


def figure_verify_corners(n: int = 8, out: Path = ASSETS / "verify_corners.jpg") -> Path | None:
    gen = _verify_generator("corners")
    if gen is None:
        return None
    rng = np.random.default_rng(11)
    tiles = []
    for _ in range(n):
        s = gen.generate(rng, want_rectified=False)
        from ..utils.geometry import order_corners
        tiles.append(draw_quad(s.photo, order_corners(s.corners_content),
                               PALETTE["gt"]))
    imwrite_rgb(out, grid(tiles, cols=4, cell=300), quality=92)
    print(f"  {out}")
    return out


def figure_verify_crops(n: int = 12, out: Path = ASSETS / "verify_crops.jpg") -> Path | None:
    gen = _verify_generator("enhancement")
    if gen is None:
        return None
    rng = np.random.default_rng(21)
    tiles = []
    for _ in range(n):
        s = gen.generate(rng, want_rectified=True, page_size=(256, 256))
        tiles.append(hstack([s.rectified, s.target], gap=4))
    imwrite_rgb(out, grid(tiles, cols=4, cell=280), quality=92)
    print(f"  {out}")
    return out


def figure_verify_course(n: int = 4, out: Path = ASSETS / "verify_course.jpg") -> Path | None:
    gen = _verify_generator("enhancement", source="course")
    if gen is None:
        return None
    rng = np.random.default_rng(31)
    rows = []
    for _ in range(n):
        s = gen.generate(rng, want_rectified=True)
        rows.append(side_by_side(
            [s.photo, s.rectified, s.target],
            ["composited photograph", "rectified input", "clean target (course scan)"],
            cell=300))
    imwrite_rgb(out, vstack(rows, gap=10), quality=92)
    print(f"  {out}")
    return out


def figure_verify_midv(n: int = 8, out: Path = ASSETS / "verify_midv.jpg") -> Path | None:
    manifest = Path("data/real/midv500/annotations.json")
    if not manifest.exists():
        print("  [skip] verify_midv: no MIDV-500 annotations")
        return None
    ds = RealPhotoSet(manifest)
    if len(ds) == 0:
        return None

    by_cond: dict[str, list[int]] = {}
    for i, it in enumerate(ds.items):
        code = Path(it["file"]).name.split("_")[1][:2]
        by_cond.setdefault(code, []).append(i)

    order = sorted(by_cond, key=lambda c: (-len(by_cond[c]), c))
    idx = [by_cond[c][len(by_cond[c]) // 2] for c in order][:n]
    tiles = []
    for i in idx:
        img, corners, _ = ds.raw(int(i))
        tiles.append(draw_quad(img, corners, PALETTE["gt"]))
    imwrite_rgb(out, grid(tiles, cols=4, cell=300), quality=92)
    print(f"  {out}")
    return out


def _stem_of(item) -> str:
    return Path(item["file"]).name.split("_")[0].split(".")[0]


def _photo_by_name(ds, stem: str) -> int | None:
    for i, it in enumerate(ds.items):
        if _stem_of(it) == stem:
            return i
    return None


def _pick_photos(ds, prefer: tuple[str, ...], n: int | None = None) -> list[int]:
    want = n if n is not None else len(prefer)
    picked = [i for i in (_photo_by_name(ds, s) for s in prefer) if i is not None]
    for i in range(len(ds.items)):
        if len(picked) >= want:
            break
        if i not in picked:
            picked.append(i)
    return picked[:want]


def gif_real_scan(out: Path = ASSETS / "real_scan_demo.gif") -> Path | None:
    from ..eval.metrics import match_cyclic
    from ..pipeline.scanner import DocumentScanner
    from ..utils.geometry import order_corners

    ds = _own_photos()
    enh, cor = Path("models/enhance.pt"), Path("models/corner_heatmap.pt")
    if ds is None or not (enh.exists() and cor.exists()):
        return None
    scanner = DocumentScanner(cor, enh, max_side=1600)
    frames = []
    for i in _pick_photos(ds, ("img13", "img10",), 2):
        stem = _stem_of(ds.items[i])
        img, gt, _ = ds.raw(i)
        res = scanner.scan(img)
        g = order_corners(gt.astype(np.float32))
        p = order_corners(np.asarray(res.corners, np.float32))
        err = float(np.linalg.norm(p - match_cyclic(p[None], g[None])[0],
                                   axis=1).mean())
        src = res.corner_result.source if res.corner_result else "network"
        stages = [
            (img, f"{stem} - the photograph", 4),
            (draw_quad(img, p, PALETTE["pred"]),
             f"corners by the {src}   ({err:.0f} px off the annotation)", 5),
            (res.rectified, "rectified by the homography", 4),
            (res.image, "enhanced - the finished scan", 7),
        ]
        for frame, title, hold in stages:
            frames += [label_bar(_letterbox(frame, 560, 748), title)] * hold
    if not frames:
        return None
    save_gif(frames, out, fps=4.0)
    print(f"  {out}")
    return out


def gif_pack_scan(out: Path = ASSETS / "pack_scan_demo.gif") -> Path | None:
    from ..pipeline.scanner import DocumentScanner

    manifest = Path("data/test_pack/ground_truth/corners.json")
    enh, cor = Path("models/enhance.pt"), Path("models/corner_heatmap.pt")
    if not (manifest.exists() and enh.exists() and cor.exists()):
        print("  [skip] pack scan gif: pack or checkpoints missing")
        return None
    items = json.loads(manifest.read_text())["items"]
    by_style: dict[str, list[dict]] = {}
    for it in items:
        by_style.setdefault(it["style"], []).append(it)
    picks = [it for style in sorted(by_style) for it in by_style[style][:2]]
    scanner = DocumentScanner(cor, enh, max_side=1400)
    frames = []
    for it in picks:
        img = imread_rgb(Path("data/test_pack") / it["file"])
        res = scanner.scan(img)
        frames += [label_bar(_letterbox(img, 520, 640),
                             f"unseen photo ({it['style']})")] * 5
        frames += [label_bar(_letterbox(res.image, 520, 640), "scan")] * 7
    save_gif(frames, out, fps=5.0)
    print(f"  {out}")
    return out


def gif_heatmaps(out: Path = ASSETS / "heatmaps_demo.gif") -> Path | None:
    ds = _own_photos()
    pipe = _corner_pipe("corner_heatmap")
    if ds is None or pipe is None:
        return None
    import torch
    i = (_pick_photos(ds, ("img13",), 1) or [None])[0]
    if i is None:
        return None
    img, _, _ = ds.raw(i)
    x = pipe._preprocess(img)
    with torch.no_grad():
        logits, seg = pipe.model.forward_with_seg(x)
    hm = torch.sigmoid(logits)[0].numpy()
    mask = torch.sigmoid(seg)[0, 0].numpy()
    base = cv2.resize(img, (480, 480), interpolation=cv2.INTER_AREA)

    def overlay(field: np.ndarray, title: str, colour=(244, 63, 94)) -> np.ndarray:
        f = cv2.resize(field, (480, 480), interpolation=cv2.INTER_CUBIC)
        f = np.clip(f, 0, 1)[..., None]
        tint = np.zeros_like(base); tint[..., :] = colour
        blend = (base * (1 - 0.75 * f) + tint * (0.75 * f)).astype(np.uint8)
        return label_bar(blend, title)

    frames = [label_bar(base, "the photograph")] * 3
    for c, name in enumerate(("top-left", "top-right", "bottom-right", "bottom-left")):
        frames += [overlay(hm[c], f"corner heatmap - {name}")] * 3
    frames += [overlay(mask, "page-interior mask (the decoder's referee)",
                       colour=(45, 212, 191))] * 4
    quad = pipe(img).corners
    done = cv2.resize(draw_quad(img, quad, PALETTE["pred"]), (480, 480),
                      interpolation=cv2.INTER_AREA)
    frames += [label_bar(done, "decoded quad - peaks scored against the mask")] * 6
    save_gif(frames, out, fps=2.2)
    print(f"  {out}")
    return out


def gif_modes(out: Path = ASSETS / "modes_demo.gif") -> Path | None:
    from ..pipeline.enhance_pipeline import apply_output_mode
    from ..utils.geometry import order_corners, rectify
    ds = _own_photos()
    pipe = _enhancer()
    if ds is None or pipe is None:
        return None
    i = (_pick_photos(ds, ("img10",), 1) or [None])[0]
    if i is None:
        return None
    img, gt, _ = ds.raw(i)
    page = rectify(img, order_corners(gt.astype(np.float32)), max_side=1100)[0]
    raw = pipe(page, mode="raw").image
    frames = []
    for mode in ("color", "gray", "bw", "whiteboard"):
        styled = apply_output_mode(raw, mode)
        if styled.ndim == 2:
            styled = cv2.cvtColor(styled, cv2.COLOR_GRAY2RGB)
        frames += [label_bar(_letterbox(styled, 460, 620), f"mode: {mode}")] * 5
    save_gif(frames, out, fps=2.5)
    print(f"  {out}")
    return out


def gif_corner_ab(out: Path = ASSETS / "corner_ab_demo.gif") -> Path | None:
    ds = _own_photos()
    heat = _corner_pipe("corner_heatmap")
    regr = _corner_pipe("corner_regression")
    if ds is None or heat is None or regr is None:
        return None
    from ..eval.metrics import match_cyclic
    from ..utils.geometry import order_corners
    frames = []
    for i in _pick_photos(ds, ("img3", "img13", "img21",), 3):
        stem = _stem_of(ds.items[i])
        img, gt, _ = ds.raw(i)
        g = order_corners(gt.astype(np.float32))
        for name, pipe in (("B: heatmaps + mask", heat), ("A: regression", regr)):
            p = order_corners(np.asarray(pipe(img).corners, np.float32))
            err = float(np.linalg.norm(p - match_cyclic(p[None], g[None])[0],
                                       axis=1).mean())
            tile = label_bar(_letterbox(draw_quad(img, p, PALETTE["pred"]), 500, 620),
                             f"{stem} - {name}   {err:.0f} px")
            frames += [tile] * 5
    save_gif(frames, out, fps=1.6)
    print(f"  {out}")
    return out


def gif_wipe_real(out: Path = ASSETS / "enhance_wipe_real.gif") -> Path | None:
    from ..utils.geometry import order_corners, rectify
    ds = _own_photos()
    pipe = _enhancer()
    if ds is None or pipe is None:
        return None
    i = (_pick_photos(ds, ("img16",), 1) or [None])[0]
    if i is None:
        return None
    img, gt, _ = ds.raw(i)
    page = rectify(img, order_corners(gt.astype(np.float32)), max_side=1100)[0]
    outp = pipe(page, mode="color").image
    frames = []
    for f in list(np.linspace(0, 1, 14)) + list(np.linspace(1, 0, 14)):
        frames.append(wipe(page, outp, float(f)))
    save_gif(frames, out, fps=12, max_width=720)
    print(f"  {out}")
    return out


def figure_rotation_invariance(out: Path = ASSETS / "rotation_invariance.jpg") -> Path | None:
    from ..pipeline.scanner import DocumentScanner

    ds = _own_photos()
    enh, cor = Path("models/enhance.pt"), Path("models/corner_heatmap.pt")
    if ds is None or not (enh.exists() and cor.exists()):
        return None
    i = (_pick_photos(ds, ("img10",), 1) or [None])[0]
    if i is None:
        return None
    img, _, _ = ds.raw(i)
    scanner = DocumentScanner(cor, enh, max_side=1400)
    rot = {0: None, 90: cv2.ROTATE_90_COUNTERCLOCKWISE,
           180: cv2.ROTATE_180, 270: cv2.ROTATE_90_CLOCKWISE}
    tops, bottoms = [], []
    for angle, k in rot.items():
        photo = img if k is None else cv2.rotate(img, k)
        res = scanner.scan(photo)
        tops.append(label_bar(_letterbox(photo, 360, 400),
                              f"input rotated {angle}\N{DEGREE SIGN}"))
        bottoms.append(label_bar(_letterbox(res.image, 360, 400),
                                 f"scan (auto-rotated {res.rotation_applied}\N{DEGREE SIGN})"))
    sheet = np.concatenate([np.concatenate(tops, axis=1),
                            np.concatenate(bottoms, axis=1)], axis=0)
    imwrite_rgb(out, sheet, quality=90)
    print(f"  {out}")
    return out


def figure_pack_demo(out: Path = ASSETS / "pack_demo.jpg") -> Path | None:
    from ..pipeline.scanner import DocumentScanner

    manifest = Path("data/test_pack/ground_truth/corners.json")
    enh, cor = Path("models/enhance.pt"), Path("models/corner_heatmap.pt")
    if not (manifest.exists() and enh.exists() and cor.exists()):
        print("  [skip] pack demo: pack or checkpoints missing")
        return None
    items = [it for it in json.loads(manifest.read_text())["items"]
             if it["style"] == "ood"][:3]
    scanner = DocumentScanner(cor, enh, max_side=1400)
    rows = []
    for it in items:
        img = imread_rgb(Path("data/test_pack") / it["file"])
        res = scanner.scan(img)
        from ..eval.metrics import match_cyclic
        from ..utils.geometry import order_corners
        p = order_corners(np.asarray(res.corners, np.float32))
        g = order_corners(np.asarray(it["corners"], np.float32))
        err = float(np.linalg.norm(p - match_cyclic(p[None], g[None])[0],
                                   axis=1).mean())
        rows.append(np.concatenate([
            label_bar(_letterbox(draw_quad(img, p, PALETTE["pred"]), 430, 330),
                      f"unseen OOD photo - detected ({err:.0f} px off truth)"),
            label_bar(_letterbox(res.image, 430, 330), "scan"),
        ], axis=1))
    imwrite_rgb(out, np.concatenate(rows, axis=0), quality=90)
    print(f"  {out}")
    return out


def gif_before_after(out: Path = ASSETS / "before_after_loop.gif") -> Path | None:
    from ..pipeline.scanner import DocumentScanner

    ds = _own_photos()
    enh, cor = Path("models/enhance.pt"), Path("models/corner_heatmap.pt")
    if ds is None or not (enh.exists() and cor.exists()):
        return None
    scanner = DocumentScanner(cor, enh, max_side=1400)
    frames = []
    for i in _pick_photos(ds, ("img13", "img3", "img10", "img6", "img21", "img14",), 6):
        stem = _stem_of(ds.items[i])
        img, _, _ = ds.raw(i)
        res = scanner.scan(img)
        frames += [label_bar(_letterbox(img, 520, 660), f"photo ({stem})")] * 5
        frames += [label_bar(_letterbox(res.image, 520, 660), "scan")] * 7
    if not frames:
        return None
    save_gif(frames, out, fps=5.0)
    print(f"  {out}")
    return out


def figure_output_modes(out: Path = ASSETS / "output_modes.jpg") -> Path | None:
    from ..pipeline.enhance_pipeline import apply_output_mode
    pipe = _enhancer()
    if pipe is None:
        print("  [skip] output modes: no checkpoint")
        return None
    ds = FrozenPairSet(Path("data/frozen/enhance_test"))
    inp, _ = ds.page(3)
    base = pipe(inp, mode="raw").network_output
    panels = [inp] + [apply_output_mode(base, m) for m in
                      ("color", "gray", "bw", "whiteboard")]
    titles = ["degraded input", "color", "grayscale", "black & white", "whiteboard"]
    imwrite_rgb(out, side_by_side(panels, titles, cell=300), quality=92)
    print(f"  {out}")
    return out


def figure_gallery_photos(n: int = 16, cols: int = 4,
                          out: Path = ASSETS / "gallery_photos.jpg") -> Path | None:
    from ..data.prepare import load_splits, make_generator
    try:
        gen = make_generator("test", cfg=DegradationConfig.for_corners(),
                             splits=load_splits(), photo_long_side=560)
    except Exception as exc:
        print(f"  [skip] gallery_photos: {exc}")
        return None
    tiles = [draw_quad(s.photo, s.corners, PALETTE["gt"], label_corners=False)
             for s in (gen.generate(np.random.default_rng((2024, i)),
                                    want_rectified=False) for i in range(n))]
    imwrite_rgb(out, grid(tiles, cols=cols, cell=300), quality=90)
    print(f"  {out}")
    return out


def figure_gallery_degradations(n: int = 12, cols: int = 4,
                                out: Path = ASSETS / "gallery_degradations.jpg") -> Path | None:
    from ..data.prepare import load_splits, make_generator
    try:
        splits = load_splits()
        gen = make_generator("test", cfg=DegradationConfig.for_enhancement(),
                             splits=splits, source="course", photo_long_side=760)
    except Exception as exc:
        print(f"  [skip] gallery_degradations: {exc}")
        return None
    tiles = []
    for i in range(n):
        s = gen.generate(np.random.default_rng((31, i)), want_rectified=True,
                         scan_index=0)
        ops = [o for o, _ in s.trace.ops
               if o in ("soft_shadow", "specular_glare", "motion_blur",
                        "defocus_blur", "vignette", "page_curl")]
        tiles.append(label_bar(s.rectified, ", ".join(ops[:3]) or "mild"))
    imwrite_rgb(out, grid(tiles, cols=cols, cell=320), quality=90)
    print(f"  {out}")
    return out


def figure_gallery_enhance(n: int = 8, out: Path = ASSETS / "gallery_enhance.jpg") -> Path | None:
    pipe = _enhancer()
    if pipe is None:
        print("  [skip] gallery_enhance: no checkpoint")
        return None
    from .metrics import psnr
    ds = FrozenPairSet(Path("data/frozen/enhance_test"))
    tiles = []
    step = max(1, len(ds) // n)
    for i in range(0, min(len(ds), n * step), step):
        inp, tgt = ds.page(i)
        out_img = pipe(inp, mode="raw").network_output
        tiles.append(label_bar(hstack([inp, out_img], gap=4),
                               f"input {psnr(inp, tgt):.1f} dB  ->  enhanced "
                               f"{psnr(out_img, tgt):.1f} dB"))
    if not tiles:
        return None
    imwrite_rgb(out, grid(tiles[:n], cols=2, cell=560), quality=90)
    print(f"  {out}")
    return out


def figure_gallery_course(n: int = 8, out: Path = ASSETS / "gallery_course.jpg") -> Path | None:
    from ..data.prepare import load_splits, make_generator
    pipe = _enhancer()
    try:
        gen = make_generator("test", cfg=DegradationConfig.for_enhancement(),
                             splits=load_splits(), source="course",
                             photo_long_side=900)
    except Exception as exc:
        print(f"  [skip] gallery_course: {exc}")
        return None
    rows = []
    for i in range(n):
        s = gen.generate(np.random.default_rng((808, i)))
        panels = [draw_quad(s.photo, s.corners, PALETTE["gt"], label_corners=False),
                  s.rectified]
        titles = ["photograph", "rectified input"]
        if pipe is not None:
            panels.append(pipe(s.rectified, mode="raw").network_output)
            titles.append("enhanced")
        panels.append(s.target)
        titles.append("clean scan (target)")
        rows.append(side_by_side(panels, titles, cell=270))
    imwrite_rgb(out, vstack(rows), quality=90)
    print(f"  {out}")
    return out


def figure_gallery_backgrounds(n: int = 12, out: Path = ASSETS / "gallery_backgrounds.jpg") -> Path | None:
    from ..data.prepare import corpora_for, load_splits
    try:
        _, bgs = corpora_for("test", load_splits())
    except Exception as exc:
        print(f"  [skip] gallery_backgrounds: {exc}")
        return None
    rng = np.random.default_rng(5)
    tiles = [bgs.canvas(300, 300, rng) for _ in range(n)]
    imwrite_rgb(out, grid(tiles, cols=6, cell=200), quality=88)
    print(f"  {out}")
    return out


def gif_degradation(out: Path = ASSETS / "degradation_pipeline.gif") -> Path | None:
    from ..data.prepare import load_splits, make_generator
    try:
        gen = make_generator("test", cfg=DegradationConfig.for_enhancement(),
                             splits=load_splits(), photo_long_side=980)
    except Exception as exc:
        print(f"  [skip] degradation gif: {exc}")
        return None
    s = gen.generate(np.random.default_rng(18), keep_stages=True, want_rectified=False)
    frames = []
    n = len(s.trace.stages)
    for k, (name, img) in enumerate(s.trace.stages, start=1):
        frames.append(label_bar(img, f"stage {k}/{n} - {name.replace('_', ' ')}"))
    frames += [frames[-1]] * 4
    save_gif(frames, out, fps=1.6, max_width=880)
    print(f"  {out}")
    return out


def gif_scan(out: Path = ASSETS / "scan_demo.gif") -> Path | None:
    from ..pipeline.scanner import DocumentScanner
    enh = RUNS / "enhance_main" / "best.pt"
    cor = RUNS / "corner_heatmap" / "best.pt"
    if Path("models/enhance.pt").exists():
        enh = Path("models/enhance.pt")
    if Path("models/corner_heatmap.pt").exists():
        cor = Path("models/corner_heatmap.pt")
    if not (enh.exists() and cor.exists()):
        print("  [skip] scan gif: checkpoints missing")
        return None
    photos = sorted(ASSETS.glob("examples/*.jpg"))
    if not photos:
        return None
    scanner = DocumentScanner(cor, enh)
    img = imread_rgb(photos[0])
    res = scanner.scan(img)

    def fit(a):
        return _letterbox(a, 560, 640)

    frames = [label_bar(fit(img), "1. photograph (synthetic, held-out scan)")] * 3
    frames += [label_bar(fit(draw_quad(img, res.corners, PALETTE["pred"])),
                         "2. corners detected")] * 4
    frames += [label_bar(fit(res.rectified), "3. rectified by homography")] * 4
    for f in np.linspace(0, 1, 8):
        frames.append(label_bar(fit(wipe(res.rectified, res.image, f)),
                                "4. enhanced by the network"))
    frames += [label_bar(fit(res.image), "done - clean scan")] * 6
    save_gif(frames, out, fps=3.5, max_width=720)
    print(f"  {out}")
    return out


def gif_wipe(out: Path = ASSETS / "enhance_wipe.gif") -> Path | None:
    pipe = _enhancer()
    if pipe is None:
        print("  [skip] wipe gif: no checkpoint")
        return None
    from ..eval.metrics import psnr
    ds = FrozenPairSet(Path("data/frozen/enhance_test"))
    worst_i, worst_score = 0, 1e9
    for i in range(min(12, len(ds))):
        inp, tgt = ds.page(i)
        score = psnr(inp, tgt)
        if score < worst_score:
            worst_i, worst_score = i, score
    inp, _ = ds.page(worst_i)
    outp = pipe(inp, mode="raw").network_output
    frames = []
    for f in list(np.linspace(0, 1, 14)) + list(np.linspace(1, 0, 14)):
        frames.append(wipe(inp, outp, float(f)))
    save_gif(frames, out, fps=12, max_width=720)
    print(f"  {out}")
    return out


def gif_stages(out: Path = ASSETS / "stages_demo.gif") -> Path | None:
    from ..pipeline.scanner import DocumentScanner
    ds = _own_photos()
    enh, cor = Path("models/enhance.pt"), Path("models/corner_heatmap.pt")
    if ds is None or not (enh.exists() and cor.exists()):
        print("  [skip] stages gif: photos or checkpoints missing")
        return None
    scanner = DocumentScanner(cor, enh)

    def fit(a):
        return _letterbox(a, 520, 700)

    frames = []
    for i in _pick_photos(ds, ("img1", "img13", "img10",), 3):
        img = ds.raw(i)[0]
        res = scanner.scan(img)
        turn = f", turned {res.rotation_applied} deg" if res.rotation_applied else ""
        steps = [
            (img, "1. the photograph as taken"),
            (draw_quad(img, res.corners, PALETTE["pred"]), "2. corners, by the CNN"),
            (res.rectified, "3. rectified -- homography from those 4 points"),
            (res.image, f"4. enhanced{turn}"),
        ]
        for frame, caption in steps:
            frames += [label_bar(fit(frame), caption)] * 5
    if not frames:
        return None
    save_gif(frames, out, fps=3.0, max_width=680)
    print(f"  {out}")
    return out


def gif_ocr(out: Path = ASSETS / "ocr_demo.gif") -> Path | None:
    from ..eval.ocr import ocr_available, run_ocr
    from ..utils.geometry import order_corners, rectify
    ds = _own_photos()
    pipe = _enhancer()
    if ds is None or pipe is None or not ocr_available():
        print("  [skip] ocr gif: photos, checkpoint or OCR engine missing")
        return None

    def boxed(img):
        res = run_ocr(img, with_boxes=True)
        shown = img.copy()
        scale = float(res.meta.get("scale", 1.0)) or 1.0
        for b in res.meta.get("boxes", []):
            x, y, w, h = (int(v / scale) for v in b["box"])
            cv2.rectangle(shown, (x, y), (x + w, y + h), (46, 204, 113), 2)
        return shown, res.words, res.mean_confidence

    frames = []
    for i in _pick_photos(ds, ("img1", "img3", "img13",), 3):
        img, gt = ds.raw(i)[:2]
        page = rectify(img, order_corners(gt.astype(np.float32)), max_side=1200)[0]
        out_img = pipe(page, mode="raw").network_output
        for src, tag in ((page, "rectified input"), (out_img, "after enhancement")):
            shown, words, conf = boxed(src)
            frames += [label_bar(_letterbox(shown, 520, 700),
                                 f"{tag} -- {words} words, confidence {conf:.0f}")] * 6
    if not frames:
        return None
    save_gif(frames, out, fps=2.2, max_width=660)
    print(f"  {out}")
    return out


def gif_tta(out: Path = ASSETS / "tta_demo.gif") -> Path | None:
    from ..utils.geometry import order_corners
    pipe = _corner_pipe()
    ds = _own_photos()
    if pipe is None or ds is None:
        print("  [skip] tta gif: photos or checkpoint missing")
        return None
    i = (_pick_photos(ds, ("img13",), 1) or [0])[0]
    img, gt = ds.raw(i)[:2]

    frames = []
    for k in range(4):
        turned = np.rot90(img, k).copy()
        quad = pipe(turned, tta=False).corners
        shown = draw_quad(turned, quad, PALETTE["pred"])
        frames += [label_bar(_letterbox(shown, 560, 700),
                             f"view {k + 1}/4 -- predicted at {k * 90} deg")] * 4
    voted = pipe(img, tta=True).corners
    err = float(np.linalg.norm(order_corners(voted)
                               - order_corners(gt.astype(np.float32)), axis=1).mean())
    shown = draw_quad(draw_quad(img, gt, PALETTE["gt"]), voted, PALETTE["pred"])
    frames += [label_bar(_letterbox(shown, 560, 700),
                         f"per-corner median of the four -- {err:.1f} px from truth")] * 8
    save_gif(frames, out, fps=2.5, max_width=640)
    print(f"  {out}")
    return out


def gif_dropout_ab(out: Path = ASSETS / "dropout_demo.gif") -> Path | None:
    from ..utils.geometry import order_corners
    ds = _own_photos()
    ctrl = Path("runs/abl_heat_ctrl/best.pt")
    drop = Path("runs/abl_heat_drop/best.pt")
    if ds is None or not (ctrl.exists() and drop.exists()):
        print("  [skip] dropout gif: ablation checkpoints missing")
        return None
    from ..pipeline.corner_pipeline import CornerPipeline
    pipes = {"control (no dropout)": CornerPipeline(ctrl, refine=False),
             "with dropout": CornerPipeline(drop, refine=False)}

    frames = []
    for i in _pick_photos(ds, ("img1", "img3", "img13",), 3):
        img, gt = ds.raw(i)[:2]
        gt4 = order_corners(gt.astype(np.float32))
        for name, pipe in pipes.items():
            quad = pipe(img).corners
            err = float(np.linalg.norm(order_corners(quad) - gt4, axis=1).mean())
            shown = draw_quad(draw_quad(img, gt, PALETTE["gt"]), quad, PALETTE["pred"])
            frames += [label_bar(_letterbox(shown, 540, 680),
                                 f"{name} -- {err:.1f} px")] * 6
    if not frames:
        return None
    save_gif(frames, out, fps=2.2, max_width=640)
    print(f"  {out}")
    return out


def gif_surfaces(out: Path = ASSETS / "surfaces_demo.gif") -> Path | None:
    from ..utils.geometry import order_corners
    gen = _verify_generator("corners")
    pipe = _corner_pipe()
    if gen is None or pipe is None:
        print("  [skip] surfaces gif: generator or checkpoint missing")
        return None
    rng = np.random.default_rng(404)
    frames = []
    for k in range(12):
        s = gen.generate(rng)
        quad = pipe(s.photo).corners
        err = float(np.linalg.norm(order_corners(quad)
                                   - order_corners(np.asarray(s.corners, np.float32)),
                                   axis=1).mean())
        diag = float(np.hypot(*s.photo.shape[:2]))
        shown = draw_quad(draw_quad(s.photo, s.corners, PALETTE["gt"]),
                          quad, PALETTE["pred"])
        frames += [label_bar(_letterbox(shown, 560, 620),
                             f"surface {k + 1}/12 -- {err / diag * 100:.2f}% of diagonal")] * 3
    save_gif(frames, out, fps=2.4, max_width=640)
    print(f"  {out}")
    return out


def gif_zoom(out: Path = ASSETS / "zoom_demo.gif") -> Path | None:
    from ..utils.geometry import order_corners, rectify
    ds = _own_photos()
    pipe = _enhancer()
    if ds is None or pipe is None:
        print("  [skip] zoom gif: photos or checkpoint missing")
        return None
    i = _photo_by_name(ds, "img1") or 0
    img, gt = ds.raw(i)[:2]
    page = rectify(img, order_corners(gt.astype(np.float32)), max_side=1400)[0]
    out_img = pipe(page, mode="raw").network_output
    h, w = page.shape[:2]
    cy, cx = int(h * 0.42), int(w * 0.5)

    frames = []
    for f in list(np.linspace(1.0, 0.28, 10)) + [0.28] * 4 + list(np.linspace(0.28, 1.0, 6)):
        hh, ww = max(64, int(h * f / 2)), max(64, int(w * f / 2))
        y0, y1 = max(0, cy - hh), min(h, cy + hh)
        x0, x1 = max(0, cx - ww), min(w, cx + ww)
        pair = side_by_side([page[y0:y1, x0:x1], out_img[y0:y1, x0:x1]],
                            ["degraded input", "network output"], cell=340)
        frames.append(pair)
    save_gif(frames, out, fps=6, max_width=720)
    print(f"  {out}")
    return out


def gif_severity(out: Path = ASSETS / "severity_demo.gif") -> Path | None:
    from ..data.degrade import DegradationConfig
    from ..data.prepare import load_splits, make_generator
    from ..eval.metrics import psnr
    pipe = _enhancer()
    if pipe is None:
        print("  [skip] severity gif: no checkpoint")
        return None
    try:
        splits = load_splits()
    except Exception:
        print("  [skip] severity gif: datasets not built")
        return None

    frames = []
    for level in (0.15, 0.35, 0.55, 0.75, 1.0):
        base = DegradationConfig.for_enhancement()
        cfg = replace(
            base,
            soft_shadow_strength=(0.05, max(0.08, level)),
            illumination_strength=(0.05, max(0.08, level)),
            gaussian_sigma=(0.1, max(0.2, level * 2.4)),
            noise_sigma=(0.001, max(0.002, level * 0.04)),
            jpeg_quality=(int(85 - 50 * level), int(90 - 45 * level)),
            p_soft_shadow=1.0, p_illumination=1.0, p_glare=level,
        )
        gen = make_generator("test", cfg=cfg, splits=splits, photo_long_side=1000)
        s = gen.generate(np.random.default_rng((77, int(level * 100))),
                         want_rectified=True)
        restored = pipe(s.rectified, mode="raw").network_output
        before, after = psnr(s.rectified, s.target), psnr(restored, s.target)
        pair = side_by_side([s.rectified, restored],
                            [f"degraded  {before:.1f} dB",
                             f"restored  {after:.1f} dB"], cell=340)
        frames += [label_bar(pair, f"severity {level:.0%}  "
                                   f"(+{after - before:.1f} dB recovered)")] * 5
    save_gif(frames, out, fps=2.4, max_width=720)
    print(f"  {out}")
    return out


def gif_labels(out: Path = ASSETS / "labels_demo.gif") -> Path | None:
    gen = _verify_generator("enhancement")
    if gen is None:
        print("  [skip] labels gif: datasets not built")
        return None
    rng = np.random.default_rng(11)
    frames = []
    for _ in range(4):
        s = gen.generate(rng, want_rectified=True)
        views = [
            (draw_quad(s.photo, s.corners, PALETTE["gt"]),
             "composited photo + the 4 chosen points (free labels)"),
            (side_by_side([s.rectified, s.target],
                          ["rectified input", "clean target"], cell=320),
             "warped back by the same homography"),
            (checkerboard_diff(s.rectified, s.target, tile=40),
             "checkerboard interleave -- edges must run straight through"),
        ]
        for frame, caption in views:
            frames += [label_bar(_letterbox(frame, 660, 470), caption)] * 5
    save_gif(frames, out, fps=2.6, max_width=700)
    print(f"  {out}")
    return out


def figure_ocr_text(out: Path = ASSETS / "ocr_text.jpg") -> Path | None:
    from ..eval.ocr import ocr_available, run_ocr
    from ..pipeline.scanner import DocumentScanner

    ds = _own_photos()
    enh, cor = Path("models/enhance.pt"), Path("models/corner_heatmap.pt")
    if ds is None or not ocr_available() or not (enh.exists() and cor.exists()):
        print("  [skip] ocr_text: photos, checkpoints or OCR engine missing")
        return None
    scanner = DocumentScanner(cor, enh)

    scored = []
    for i in range(len(ds.items)):
        img = ds.raw(i)[0]
        res = scanner.scan(img)
        got = run_ocr(res.image)
        scored.append((got.words, i, res.image, got))
    scored.sort(key=lambda t: t[0])
    picks = [("worst-reading", scored[0]),
             ("median", scored[len(scored) // 2]),
             ("best-reading", scored[-1])]

    def text_panel(text: str, w: int, h: int) -> np.ndarray:
        panel = np.full((h, w, 3), 252, np.uint8)
        words = " ".join(text.split())
        line, y, per = "", 34, max(18, w // 11)
        for word in words.split(" "):
            if len(line) + len(word) + 1 > per:
                cv2.putText(panel, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (36, 36, 44), 1, cv2.LINE_AA)
                line, y = word, y + 24
                if y > h - 18:
                    cv2.putText(panel, "...", (14, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, (120, 120, 130), 1, cv2.LINE_AA)
                    return panel
            else:
                line = f"{line} {word}".strip()
        if line:
            cv2.putText(panel, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (36, 36, 44), 1, cv2.LINE_AA)
        return panel

    rows = []
    for tag, (words, i, scan, got) in picks:
        stem = _stem_of(ds.items[i])
        h = 620
        page = cv2.resize(scan, (int(round(scan.shape[1] * h / scan.shape[0])), h))
        panel = text_panel(got.text or "(nothing recognised)", 560, h)
        left = label_bar(page, f"{stem} - {tag}")
        right = label_bar(panel,
                          f"recognised text: {words} words at "
                          f"{got.mean_confidence:.0f} confidence")
        hm = max(left.shape[0], right.shape[0])
        rows.append(np.concatenate(
            [cv2.copyMakeBorder(t, 0, hm - t.shape[0], 0, 8,
                                cv2.BORDER_CONSTANT, value=(255, 255, 255))
             for t in (left, right)], axis=1))
    wmax = max(r.shape[1] for r in rows)
    sheet = np.concatenate(
        [cv2.copyMakeBorder(r, 0, 10, 0, wmax - r.shape[1],
                            cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in rows],
        axis=0)
    imwrite_rgb(out, sheet, quality=92)
    print(f"  {out}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate documentation figures")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--charts", action="store_true")
    ap.add_argument("--images", action="store_true")
    ap.add_argument("--gifs", action="store_true")
    args = ap.parse_args(argv)
    if not any([args.all, args.charts, args.images, args.gifs]):
        args.all = True

    ASSETS.mkdir(parents=True, exist_ok=True)
    if args.all or args.charts:
        print("[charts]")
        figure_architecture()
        figure_dataset_stats()
        figure_training_curves()
        figure_real_accuracy()
    if args.all or args.images:
        print("[image figures]")
        figure_qualitative_enhance()
        figure_gallery_photos()
        figure_gallery_degradations()
        figure_gallery_enhance()
        figure_gallery_course()
        figure_gallery_backgrounds()
        figure_qualitative_corners()
        figure_end_to_end()
        figure_output_modes()
        print("[real photographs]")
        figure_real_gallery()
        figure_real_end_to_end(n=6, start=0)
        figure_real_end_to_end(n=6, start=12,
                               out=ASSETS / "real_end_to_end_2.jpg")
        figure_real_enhance_pairs()
        figure_real_output_modes()
        figure_real_failures()
        figure_realism_check()
        figure_network_choice()
        figure_rotation_invariance()
        figure_pack_demo()
        figure_verify_pairs()
        figure_verify_corners()
        figure_verify_crops()
        figure_verify_course()
        figure_verify_midv()
    if args.all or args.gifs:
        print("[animations]")
        gif_degradation()
        gif_scan()
        gif_wipe()
        gif_wipe_real()
        gif_real_scan()
        gif_before_after()
        gif_pack_scan()
        gif_heatmaps()
        gif_modes()
        gif_corner_ab()
        gif_stages()
        gif_ocr()
        figure_ocr_text()
        gif_tta()
        gif_dropout_ab()
        gif_surfaces()
        gif_zoom()
        gif_severity()
        gif_labels()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
