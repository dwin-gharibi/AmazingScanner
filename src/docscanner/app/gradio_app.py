from __future__ import annotations

import argparse
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from ..data.degrade import DegradationConfig
from ..data.generator import SyntheticSampleGenerator
from ..eval.ocr import (
    ENGINE_MODES,
    PAGE_MODES,
    extract_text,
    language_choices,
    ocr_available,
    ocr_backends,
    run_ocr,
    searchable_pdf,
)
from ..pipeline.corner_pipeline import CornerPipeline
from ..pipeline.enhance_pipeline import EnhancementPipeline, apply_output_mode
from ..pipeline.postprocess import add_margin, adjust, auto_crop, deskew, rotate
from ..pipeline.scanner import DocumentScanner, rectify_with_corners, save_pdf
from ..utils.geometry import order_corners
from ..utils.imageio import ensure_rgb, imread_rgb
from ..utils.viz import PALETTE, draw_corners, draw_quad, heatmap_overlay, side_by_side
from .theme import CSS, build_theme, hero, note, stat_card

RUNS = Path("runs")
MODELS = Path("models")
DOCS = Path("docs/assets")
OUTPUT_MODES = ["color", "gray", "bw", "whiteboard"]


def _first_existing(*paths: Path) -> Path | None:
    for p in paths:
        if p and Path(p).exists():
            return Path(p)
    return None


@dataclass
class Registry:
    enhance_ckpt: Path | None = None
    heatmap_ckpt: Path | None = None
    regression_ckpt: Path | None = None

    def __post_init__(self) -> None:
        self.enhance_ckpt = _first_existing(MODELS / "enhance.pt",
                                            RUNS / "enhance_main" / "best.pt",
                                            RUNS / "enhance_main" / "last.pt")
        self.heatmap_ckpt = _first_existing(MODELS / "corner_heatmap.pt",
                                            RUNS / "corner_heatmap" / "best.pt",
                                            RUNS / "corner_heatmap" / "last.pt")
        self.regression_ckpt = _first_existing(MODELS / "corner_regression.pt",
                                               RUNS / "corner_regression" / "best.pt",
                                               RUNS / "corner_regression" / "last.pt")
        self._enh: EnhancementPipeline | None = None
        self._heat: CornerPipeline | None = None
        self._reg: CornerPipeline | None = None
        self._scanner: DocumentScanner | None = None

    @property
    def enhancer(self) -> EnhancementPipeline:
        if self._enh is None:
            if not self.enhance_ckpt:
                raise gr.Error("No enhancement checkpoint found. Train one first:\n"
                               "  python -m docscanner.engine.train_enhance")
            self._enh = EnhancementPipeline(self.enhance_ckpt, max_side=1600)
        return self._enh

    def corners(self, approach: str = "heatmap") -> CornerPipeline:
        if approach.startswith("heat"):
            if self._heat is None:
                if not self.heatmap_ckpt:
                    raise gr.Error("No heatmap corner checkpoint found.")
                self._heat = CornerPipeline(self.heatmap_ckpt)
            return self._heat
        if self._reg is None:
            if not self.regression_ckpt:
                raise gr.Error("No regression corner checkpoint found.")
            self._reg = CornerPipeline(self.regression_ckpt)
        return self._reg

    @property
    def scanner(self) -> DocumentScanner:
        if self._scanner is None:
            ck = self.heatmap_ckpt or self.regression_ckpt
            if not ck or not self.enhance_ckpt:
                raise gr.Error("Both a corner and an enhancement checkpoint are needed.")
            self._scanner = DocumentScanner(ck, self.enhance_ckpt)
        return self._scanner

    def status_html(self) -> str:
        def card(label, path):
            if path:
                return stat_card(label, "ready", Path(path).parent.name, "good")
            return stat_card(label, "missing", "train it first", "bad")
        _engines = ocr_backends()
        ocr = stat_card("OCR", "ready" if _engines else "absent",
                        " + ".join(_engines) or "no engine",
                        "good" if _engines else "warn")
        return ('<div class="ds-stats">'
                + card("Enhancement", self.enhance_ckpt)
                + card("Corners (heatmap)", self.heatmap_ckpt)
                + card("Corners (regression)", self.regression_ckpt)
                + ocr + "</div>")


REG = Registry()

_GEN: SyntheticSampleGenerator | None = None


def _generator() -> SyntheticSampleGenerator | None:
    global _GEN
    if _GEN is None:
        try:
            from ..data.prepare import load_splits, make_generator
            _GEN = make_generator("test", splits=load_splits(), photo_long_side=1100)
        except Exception:
            return None
    return _GEN


def _tmp_path(suffix: str) -> str:
    return tempfile.NamedTemporaryFile(delete=False, suffix=suffix).name


def _save_png(img: np.ndarray, stem: str = "scan") -> str:
    path = _tmp_path(f"_{stem}.png")
    cv2.imwrite(path, cv2.cvtColor(ensure_rgb(img), cv2.COLOR_RGB2BGR))
    return path


def _stats(pairs: list[tuple[str, str, str, str]]) -> str:
    return '<div class="ds-stats">' + "".join(
        stat_card(k, v, s, t) for k, v, s, t in pairs) + "</div>"


def _require(img) -> np.ndarray:
    if img is None:
        raise gr.Error("Upload an image first.")
    return ensure_rgb(np.asarray(img))


def auto_scan(photo, mode: str, do_enhance: bool, do_refine: bool,
              do_orient: bool, max_side: int, turn: str = "0",
              do_deskew: bool = False, do_autocrop: bool = False,
              margin: float = 0.0, brightness: float = 0.0,
              contrast: float = 1.0, sharpness: float = 0.0,
              do_tta: bool = True):
    img = _require(photo)
    scanner = REG.scanner
    scanner.max_side = int(max_side)
    scanner.enhancer.max_side = int(max_side)

    res = scanner.scan(img, mode=mode, enhance=do_enhance,
                       auto_orientation=do_orient, refine=do_refine,
                       tta=do_tta)
    res.image = _finish(res.image, turn, do_deskew, do_autocrop, margin,
                        brightness, contrast, sharpness)

    overlay = draw_quad(img, res.corners, PALETTE["pred"])
    if res.corner_result is not None and res.corner_result.refined:
        before = res.corner_result.corners_prerefine
        was_fallback = res.corner_result.source != "network"
        overlay = draw_corners(
            img, gt=res.corners,
            pred=before if before is not None else res.corner_result.corners_raw,
            legend=("refined (final)",
                    "classical detection" if was_fallback else "network output", ""))

    conf = res.meta.get("confidence", 1.0)
    tone = "good" if conf > 0.5 else ("warn" if conf > 0.25 else "bad")
    source = res.meta.get("corner_source", "network")
    source_label = {"network": "CNN detection",
                    "classical": "classical fallback",
                    "full_frame": "page not found",
                    "supplied": "your corners"}.get(source, source)
    t = res.timings
    cards = [
        ("Total", f"{res.seconds:.2f}s", "photo to page", ""),
        ("Detect", f"{t.get('detect', 0) * 1000:.0f} ms",
         "refined" if res.meta.get("refined") else "network only", ""),
        ("Enhance", f"{t.get('enhance', 0):.2f}s",
         f"{res.image.shape[1]}x{res.image.shape[0]}", ""),
        ("Confidence", f"{conf:.2f}", source_label, tone),
        ("Rotation", f"{res.rotation_applied}°", "auto-orient", ""),
    ]
    pdf = searchable_pdf([res.image], _tmp_path("_scan.pdf"))
    text = extract_text(res.image) if ocr_available() else ""
    txt_path = _tmp_path("_text.txt")
    Path(txt_path).write_text(text or "(OCR unavailable)", encoding="utf-8")
    return (overlay, res.rectified, res.image, (res.rectified, res.image),
            _stats(cards), _save_png(res.image), str(pdf), text, txt_path)


def _finish(page: np.ndarray, turn: str = "0", do_deskew: bool = False,
            do_autocrop: bool = False, margin: float = 0.0,
            brightness: float = 0.0, contrast: float = 1.0,
            sharpness: float = 0.0) -> np.ndarray:
    out = page
    if str(turn) not in ("0", "", "None"):
        out = rotate(out, int(turn))
    if do_deskew:
        out, _ = deskew(out)
    if do_autocrop:
        out = auto_crop(out)
    if margin and margin > 0:
        out = add_margin(out, float(margin))
    if brightness or contrast != 1.0 or sharpness:
        out = adjust(out, brightness=brightness, contrast=contrast,
                     sharpness=sharpness)
    return out


def batch_scan(files, mode: str, do_orient: bool, max_side: int,
               make_searchable: bool):
    if not files:
        raise gr.Error("Add some photographs first.")
    scanner = REG.scanner
    scanner.max_side = int(max_side)
    scanner.enhancer.max_side = int(max_side)

    pages, gallery, rows = [], [], []
    t0 = time.time()
    for i, f in enumerate(files):
        path = f if isinstance(f, str) else getattr(f, "name", None)
        if not path:
            continue
        img = imread_rgb(path)
        res = scanner.scan(img, mode=mode, auto_orientation=do_orient)
        pages.append(res.image)
        gallery.append((res.image, f"page {i + 1}"))
        rows.append(f"| {i + 1} | {Path(path).name} | {res.seconds:.1f}s | "
                    f"{res.meta.get('confidence', 1.0):.2f} | "
                    f"{res.image.shape[1]}x{res.image.shape[0]} |")

    if not pages:
        raise gr.Error("No readable images were supplied.")
    pdf = (searchable_pdf(pages, _tmp_path("_batch.pdf")) if make_searchable
           else save_pdf(pages, _tmp_path("_batch.pdf")))
    table = ("| # | file | time | confidence | size |\n|---|---|---|---|---|\n"
             + "\n".join(rows))
    cards = [("Pages", str(len(pages)), "scanned", "good"),
             ("Total", f"{time.time() - t0:.1f}s",
              f"{(time.time() - t0) / len(pages):.1f}s per page", ""),
             ("PDF", "searchable" if make_searchable else "image-only",
              "text layer" if make_searchable else "no OCR", "")]
    return gallery, str(pdf), _stats(cards), table


def click_corner(points: list, img, evt: gr.SelectData):
    if img is None:
        return points, None, "Upload a photo first."
    pts = list(points or [])
    idx = evt.index
    if not isinstance(idx, (list, tuple)) or len(idx) < 2:
        return pts, _draw_points(_require(img), pts), "Could not read the click position."
    x, y = float(idx[0]), float(idx[1])

    h, w = np.asarray(img).shape[:2]
    if x > w and y <= w and x <= h:
        x, y = y, x
    x, y = float(np.clip(x, 0, w - 1)), float(np.clip(y, 0, h - 1))
    if len(pts) >= 4:
        pts = []
    pts.append([x, y])
    preview = _draw_points(_require(img), pts)
    msg = (f"{len(pts)}/4 corners set - click "
           f"{'the next corner' if len(pts) < 4 else 'again to start over'}.")
    return pts, preview, msg


def _draw_points(img: np.ndarray, pts: list) -> np.ndarray:
    out = img.copy()
    r = max(4, int(max(img.shape[:2]) / 160))
    for i, (x, y) in enumerate(pts):
        cv2.circle(out, (int(x), int(y)), r, PALETTE["accent"], -1, cv2.LINE_AA)
        cv2.putText(out, str(i + 1), (int(x) + r + 2, int(y) - r),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, PALETTE["accent"], 2, cv2.LINE_AA)
    if len(pts) == 4:
        out = draw_quad(out, order_corners(np.array(pts, np.float32)), PALETTE["gt"])
    elif len(pts) > 1:
        cv2.polylines(out, [np.array(pts, np.int32)], False, PALETTE["accent"],
                      max(2, r // 2), cv2.LINE_AA)
    return out


def manual_scan(points: list, img, mode: str, do_enhance: bool):
    image = _require(img)
    if not points or len(points) != 4:
        raise gr.Error("Click exactly four corners on the photo first.")
    quad = order_corners(np.array(points, np.float32))
    rect, _ = rectify_with_corners(image, quad, max_side=1600)
    out = rect
    t0 = time.time()
    if do_enhance:
        out = REG.enhancer(rect, mode="raw").network_output
    styled = apply_output_mode(out, mode)
    cards = [("Corners", "manual", "clicked by you", ""),
             ("Page", f"{styled.shape[1]}x{styled.shape[0]}", "rectified", ""),
             ("Enhance", f"{time.time() - t0:.2f}s", "network", "")]
    return rect, styled, (rect, styled), _stats(cards), _save_png(styled)


def auto_fill_corners(img):
    image = _require(img)
    res = REG.corners("heatmap")(image)
    pts = [[float(x), float(y)] for x, y in res.corners]
    return pts, _draw_points(image, pts), "Detector prediction loaded - drag by clicking to redo."


def enhance_only(img, mode: str, max_side: int, run_ocr_flag: bool):
    image = _require(img)
    pipe = REG.enhancer
    pipe.max_side = int(max_side)
    res = pipe(image, mode=mode)

    cards = [("Time", f"{res.seconds:.2f}s", "tiled" if res.meta.get("tiled") else "single pass", ""),
             ("Size", f"{res.image.shape[1]}x{res.image.shape[0]}", "output", ""),
             ("Mode", mode, "post-process", "")]
    ocr_md = ""
    if run_ocr_flag and ocr_available():
        before, after = run_ocr(image), run_ocr(res.image)
        delta = after.mean_confidence - before.mean_confidence
        tone = "good" if delta > 0 else "warn"
        cards.append(("OCR conf", f"{after.mean_confidence:.1f}",
                      f"{delta:+.1f} vs input", tone))
        ocr_md = (f"**Words read** input {before.words} -> enhanced {after.words}  \n"
                  f"**Mean confidence** {before.mean_confidence:.1f} -> "
                  f"{after.mean_confidence:.1f}")
    return res.image, (image, res.image), _stats(cards), ocr_md, _save_png(res.image)


def _draw_boxes(image: np.ndarray, result) -> np.ndarray:
    out = ensure_rgb(np.asarray(image)).copy()
    boxes = result.meta.get("boxes") or []
    scale = result.meta.get("scale", 1.0) or 1.0
    for b in boxes:
        x, y, w, h = [int(v / scale) for v in b["box"]]
        conf = float(b["conf"])
        colour = ((46, 204, 113) if conf >= 75 else
                  (241, 196, 15) if conf >= 45 else (231, 76, 60))
        cv2.rectangle(out, (x, y), (x + w, y + h), colour, 2, cv2.LINE_AA)
    return out


def run_ocr_tab(img, langs, mode_label: str, engine_label: str,
                enhance_first: bool, show_boxes: bool):
    if not ocr_available():
        raise gr.Error("Tesseract is not installed. Install it with:\n"
                       "  sudo apt-get install tesseract-ocr")
    image = _require(img)
    if enhance_first:
        image = REG.enhancer(image, mode="raw").network_output

    codes = langs if isinstance(langs, list) else [langs]
    lang = "+".join([c for c in codes if c]) or "eng"
    psm = PAGE_MODES.get(mode_label, 3)
    oem = ENGINE_MODES.get(engine_label, 3)

    t0 = time.time()
    res = run_ocr(image, psm=psm, lang=lang, oem=oem, with_boxes=show_boxes)
    dt = time.time() - t0

    overlay = _draw_boxes(image, res) if show_boxes else image
    tone = "good" if res.mean_confidence >= 70 else (
        "warn" if res.mean_confidence >= 45 else "bad")
    cards = [("Confidence", f"{res.mean_confidence:.1f}", "mean, accepted words", tone),
             ("Words", str(res.words), f"{res.meta.get('all_words', 0)} detected", ""),
             ("Characters", str(res.chars), "excluding spaces", ""),
             ("Language", lang, f"psm {psm} · oem {oem}", ""),
             ("Time", f"{dt:.2f}s", "recognition", "")]

    txt = _tmp_path("_ocr.txt")
    Path(txt).write_text(res.text, encoding="utf-8")
    pdf = searchable_pdf([image], _tmp_path("_ocr.pdf"), lang=lang)
    return overlay, res.text, _stats(cards), txt, str(pdf)


def corner_lab(img, refine: bool):
    image = _require(img)
    panels, rows = [], []
    heat_vis = None
    for label, approach in (("A: regression", "regression"), ("B: heatmap", "heatmap")):
        try:
            pipe = REG.corners(approach)
        except gr.Error:
            rows.append(f"| {label} | not trained | - | - |")
            continue
        res = pipe(image, refine=refine)
        panels.append((label, res))
        rows.append(f"| {label} | {res.seconds * 1000:.0f} ms | "
                    f"{'yes' if res.refined else 'no'} | {res.confidence:.2f} |")
        if res.heatmaps is not None:
            heat_vis = heatmap_overlay(image, res.heatmaps.max(axis=0))

    if not panels:
        raise gr.Error("No corner checkpoints found - train one first.")

    overlay = image
    if len(panels) == 2:
        overlay = draw_corners(image, gt=panels[1][1].corners, pred=panels[0][1].corners,
                               legend=("B: heatmap", "A: regression", ""))
    else:
        overlay = draw_quad(image, panels[0][1].corners, PALETTE["pred"])

    crops = [side_by_side([rectify_with_corners(image, r.corners)[0]], [lbl], cell=340)
             for lbl, r in panels]
    table = ("| Approach | time | edge-refined | confidence |\n|---|---|---|---|\n"
             + "\n".join(rows))
    return overlay, heat_vis, crops, table


def degradation_lab(seed: int, perspective: float, rotation: float, shadow: float,
                    illumination: float, blur: float, noise: float, jpeg: int,
                    downscale: float, glare: bool, curl: bool):
    gen = _generator()
    if gen is None:
        raise gr.Error("Synthetic corpora not prepared. Run:\n"
                       "  python -m docscanner.data.prepare --all")
    cfg = DegradationConfig(
        quad_jitter=float(perspective),
        quad_rotation_deg=(-float(rotation), float(rotation)),
        soft_shadow_strength=(max(0.02, shadow * 0.4), max(0.05, shadow)),
        p_soft_shadow=1.0 if shadow > 0.02 else 0.0,
        illumination_strength=(max(0.02, illumination * 0.4), max(0.05, illumination)),
        p_illumination=1.0 if illumination > 0.02 else 0.0,
        gaussian_sigma=(max(0.05, blur * 0.4), max(0.1, blur)),
        noise_sigma=(max(0.001, noise * 0.4), max(0.002, noise)),
        jpeg_quality=(int(jpeg), int(jpeg)),
        downscale_factor=(max(1.02, downscale * 0.8), max(1.05, downscale)),
        p_glare=1.0 if glare else 0.0,
        p_page_curl=0.9 if curl else 0.0,
        page_curl_strength=(0.02, 0.05),
    )
    gen.cfg = cfg
    gen.pipeline.cfg = cfg
    sample = gen.generate(np.random.default_rng(int(seed)), keep_stages=True)

    stages = [(name, img) for name, img in (sample.trace.stages or [])]
    gallery = [(img, name.replace("_", " ")) for name, img in stages]
    overlay = draw_quad(sample.photo, sample.corners)
    return (overlay, sample.rectified, sample.target, gallery,
            f"```\n{sample.trace.summary()}\n```")


def _read_md(path: Path) -> str:
    return path.read_text() if path.exists() else ""


def benchmarks_md() -> str:
    root = Path("outputs/report")
    parts = []
    sections = [
        ("Enhancement - PSNR / SSIM by split", "enhancement_enhance_main.md"),
        ("Enhancement - loss and architecture ablation", "enhancement_ablation.md"),
        ("Corner detection - Approach A vs B", "corners.md"),
        ("Readability - OCR", "ocr.md"),
        ("End-to-end - annotated vs predicted corners", "end_to_end.md"),
    ]
    for title, fname in sections:
        body = _read_md(root / fname)
        if body:
            parts.append(f"### {title}\n\n{body}\n")
    if not parts:
        return ("_No benchmark results yet._\n\nRun:\n\n```bash\n"
                "python -m docscanner.eval.evaluate --all\n```")
    return "\n".join(parts)


def curve_images() -> list:
    out = []
    for run in sorted(RUNS.glob("*/curves.png")):
        out.append((str(run), run.parent.name))
    return out


def figure_images() -> list:
    names = [("verify_pairs.jpg", "Synthetic pairs: photo / rectified input / clean target"),
             ("verify_corners.jpg", "Corner labels on composited photos"),
             ("verify_midv.jpg", "Real photos (MIDV-500) with derived corner ground truth"),
             ("qualitative_enhance.jpg", "Enhancement: input / output / target"),
             ("qualitative_corners.jpg", "Corner predictions vs ground truth"),
             ("end_to_end.jpg", "End-to-end scans")]
    return [(str(DOCS / n), c) for n, c in names if (DOCS / n).exists()]


ABOUT = """
### What AmazingScanner is

A document scanner built from scratch: two convolutional networks, a synthetic
data engine, and the classical geometry that glues them together.

**1 - Synthetic data.** No photographs were annotated to train this. A clean scan
is warped onto a random surface with a random homography; the four points chosen
for that warp *are* the corner labels, and inverting the same homography yields a
pixel-perfectly aligned (degraded, clean) pair. Label generator and data
generator are one function.

**2 - Degradations.** Perspective, resolution loss, brightness/contrast/colour
cast, illumination gradients, soft shadows, glare, vignetting, camera
auto-exposure, motion and defocus blur, sensor noise and JPEG - all implemented
with OpenCV and NumPy only, every parameter randomised per sample.

**3 - The enhancement network.** A U-shaped encoder-decoder with skip
connections (thin strokes do not survive a bottleneck without them), a dilated
context block at 1/16 resolution (illumination is a global decision), squeeze-
excitation gating, and a classical background-estimate prior fed in as extra
input channels. It predicts a *residual*, so the identity is the starting point.
Trained on 192x192 crops, run fully convolutionally on whole pages.

**4 - The corner detectors.** Approach A regresses eight numbers from a
flattened feature map. Approach B predicts four Gaussian heatmaps over a grid
that extends slightly beyond the image, so a corner just outside the frame is
still representable, and reads them back with a windowed sub-pixel soft-argmax.
Both are trained with a cyclic-permutation-invariant loss, which removes the
labelling discontinuity that appears when a page is rotated near 45 degrees.

**5 - Classical refinement.** The network localises to a few pixels; the page
border is a straight, high-contrast edge. Fitting lines to the gradient ridge
near each predicted edge and re-intersecting them recovers the corners far more
precisely. CNN proposes, geometry refines.

**6 - The chain.** Detect -> rectify (with a projective aspect-ratio estimate, so
a page shot at a steep angle comes back correctly proportioned) -> enhance ->
auto-orient -> style -> export.

### Honest limits

Curled or folded pages are not planar, so no homography can flatten them.
Extreme shadows that clip to black destroy information the network cannot invent.
And the synthetic-to-real gap is the central risk of this design: the model can
only remove degradations the generator knows how to produce.
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="AmazingScanner") as demo:
        gr.HTML(hero(
            "AmazingScanner",
            "Detect the page, flatten the perspective, and restore a clean scan - "
            "two convolutional networks trained entirely on synthetic data.",
            ["from-scratch CNNs", "synthetic supervision", "OpenCV degradations",
             "mask-guided corner decoding", "OCR-verified"],
        ))
        gr.HTML(REG.status_html())

        with gr.Tab("Auto Scan"):
            gr.HTML(note("The full chain, no human input: <b>corner detection -> "
                         "rectification -> enhancement -> orientation</b>."))
            with gr.Row():
                with gr.Column(scale=3):
                    a_in = gr.Image(label="Photograph of a document", type="numpy",
                                    height=330,
                                    sources=["upload", "webcam", "clipboard"])
                    with gr.Row():
                        a_mode = gr.Dropdown(OUTPUT_MODES, value="color", label="Output style")
                        a_max = gr.Slider(640, 2400, value=1600, step=64,
                                          label="Max page side (px)")
                    with gr.Row():
                        a_enh = gr.Checkbox(True, label="Enhance")
                        a_ref = gr.Checkbox(False, label="Edge refinement")
                        a_ori = gr.Checkbox(True, label="Auto-orient")
                    a_tta = gr.Checkbox(
                        True, label="Accurate detection (4-view TTA)",
                        info="Four quarter turns, per-corner median. Cuts badly "
                             "missed pages from 17% to 12% on validation, for "
                             "~3x the detection time. Uncheck for speed.")
                    with gr.Accordion("Adjust the result", open=False):
                        a_turn = gr.Radio(["0", "90", "180", "270"], value="0",
                                          label="Rotate (degrees)")
                        with gr.Row():
                            a_deskew = gr.Checkbox(False, label="Deskew")
                            a_crop = gr.Checkbox(False, label="Auto-crop")
                        a_margin = gr.Slider(0.0, 0.08, value=0.0, step=0.005,
                                             label="Add margin")
                        a_bright = gr.Slider(-0.5, 0.5, value=0.0, step=0.02,
                                             label="Brightness")
                        a_contrast = gr.Slider(0.5, 2.0, value=1.0, step=0.05,
                                               label="Contrast")
                        a_sharp = gr.Slider(0.0, 2.0, value=0.0, step=0.1,
                                            label="Sharpen")
                    a_btn = gr.Button("Scan document", variant="primary", size="lg")
                with gr.Column(scale=4):
                    a_stats = gr.HTML()
                    with gr.Row():
                        a_overlay = gr.Image(label="Detected page", height=290)
                        a_rect = gr.Image(label="Rectified", height=290)
                    a_slider = gr.ImageSlider(label="Rectified vs enhanced", height=420)
                    a_out = gr.Image(label="Final scan", height=300, visible=False)
                    with gr.Row():
                        a_png = gr.DownloadButton("Download PNG")
                        a_pdf = gr.DownloadButton("Searchable PDF")
                        a_txt = gr.DownloadButton("Extracted text")
                    with gr.Accordion("Recognised text", open=False):
                        a_text = gr.Textbox(lines=8, show_label=False,
                                            placeholder="OCR output appears here")
            ex = sorted(DOCS.glob("examples/*.jpg"))
            if ex:
                gr.Examples([[str(p)] for p in ex[:8]], inputs=[a_in],
                            label="Example photographs (synthetic, held-out scans)")
            a_btn.click(auto_scan,
                        [a_in, a_mode, a_enh, a_ref, a_ori, a_max, a_turn,
                         a_deskew, a_crop, a_margin, a_bright, a_contrast, a_sharp,
                         a_tta],
                        [a_overlay, a_rect, a_out, a_slider, a_stats, a_png, a_pdf,
                         a_text, a_txt])

        with gr.Tab("Manual Corners"):
            gr.HTML(note("Click the four page corners in order - <b>top-left, "
                         "top-right, bottom-right, bottom-left</b> - or load the "
                         "detector's guess and compare."))
            with gr.Row():
                with gr.Column(scale=3):
                    m_in = gr.Image(label="Photograph (click on it)", type="numpy",
                                    height=420,
                                    sources=["upload", "webcam", "clipboard"])
                    m_msg = gr.Markdown("0/4 corners set.")
                    with gr.Row():
                        m_auto = gr.Button("Load detector prediction")
                        m_clear = gr.Button("Clear")
                    with gr.Row():
                        m_mode = gr.Dropdown(OUTPUT_MODES, value="color", label="Output style")
                        m_enh = gr.Checkbox(True, label="Enhance")
                    m_btn = gr.Button("Rectify and enhance", variant="primary")
                with gr.Column(scale=4):
                    m_stats = gr.HTML()
                    m_preview = gr.Image(label="Your corners", height=300)
                    m_slider = gr.ImageSlider(label="Rectified vs enhanced", height=420)
                    m_rect = gr.Image(visible=False)
                    m_out = gr.Image(visible=False)
                    m_png = gr.DownloadButton("Download PNG")
            m_pts = gr.State([])
            m_in.select(click_corner, [m_pts, m_in], [m_pts, m_preview, m_msg])
            m_auto.click(auto_fill_corners, [m_in], [m_pts, m_preview, m_msg])
            m_clear.click(lambda img: ([], img, "0/4 corners set."),
                          [m_in], [m_pts, m_preview, m_msg])
            m_btn.click(manual_scan, [m_pts, m_in, m_mode, m_enh],
                        [m_rect, m_out, m_slider, m_stats, m_png])

        with gr.Tab("Enhance Only"):
            gr.HTML(note("Section 3.4 pipeline: give it an <b>already rectified</b> "
                         "page and it returns a scan-quality one."))
            with gr.Row():
                with gr.Column(scale=3):
                    e_in = gr.Image(label="Rectified document", type="numpy", height=340,
                                    sources=["upload", "webcam", "clipboard"])
                    with gr.Row():
                        e_mode = gr.Dropdown(OUTPUT_MODES, value="color", label="Output style")
                        e_max = gr.Slider(640, 2400, value=1600, step=64, label="Max side")
                    e_ocr = gr.Checkbox(ocr_available(), label="Measure OCR gain",
                                        interactive=ocr_available())
                    e_btn = gr.Button("Enhance", variant="primary", size="lg")
                with gr.Column(scale=4):
                    e_stats = gr.HTML()
                    e_slider = gr.ImageSlider(label="Before vs after", height=460)
                    e_out = gr.Image(visible=False)
                    e_md = gr.Markdown()
                    e_png = gr.DownloadButton("Download PNG")
            e_btn.click(enhance_only, [e_in, e_mode, e_max, e_ocr],
                        [e_out, e_slider, e_stats, e_md, e_png])

        with gr.Tab("Batch / Multi-page"):
            gr.HTML(note("Drop in a stack of photographs and get <b>one searchable "
                         "PDF</b> - the multi-page workflow a scanner app is for."))
            with gr.Row():
                with gr.Column(scale=3):
                    bt_files = gr.File(label="Photographs", file_count="multiple",
                                       file_types=["image"], height=220)
                    with gr.Row():
                        bt_mode = gr.Dropdown(OUTPUT_MODES, value="color",
                                              label="Output style")
                        bt_max = gr.Slider(640, 2400, value=1400, step=64,
                                           label="Max page side")
                    with gr.Row():
                        bt_orient = gr.Checkbox(True, label="Auto-orient")
                        bt_search = gr.Checkbox(ocr_available(),
                                                label="Searchable PDF (OCR layer)",
                                                interactive=ocr_available())
                    bt_btn = gr.Button("Scan all pages", variant="primary", size="lg")
                    bt_pdf = gr.DownloadButton("Download PDF")
                with gr.Column(scale=4):
                    bt_stats = gr.HTML()
                    bt_gallery = gr.Gallery(label="Scanned pages", columns=3, height=430)
                    bt_table = gr.Markdown()
            bt_btn.click(batch_scan, [bt_files, bt_mode, bt_orient, bt_max, bt_search],
                         [bt_gallery, bt_pdf, bt_stats, bt_table])

        with gr.Tab("Corner Lab"):
            gr.HTML(note("<b>Approach A</b> (coordinate regression) against "
                         "<b>Approach B</b> (heatmaps) on the same photo."))
            with gr.Row():
                with gr.Column(scale=3):
                    c_in = gr.Image(label="Photograph", type="numpy", height=340,
                                    sources=["upload", "webcam", "clipboard"])
                    c_ref = gr.Checkbox(False, label="Sub-pixel edge refinement")
                    c_btn = gr.Button("Compare detectors", variant="primary")
                    c_table = gr.Markdown()
                with gr.Column(scale=4):
                    with gr.Row():
                        c_overlay = gr.Image(label="Both predictions", height=330)
                        c_heat = gr.Image(label="Heatmaps (Approach B)", height=330)
                    c_crops = gr.Gallery(label="Rectified with each prediction",
                                         columns=2, height=330)
            c_btn.click(corner_lab, [c_in, c_ref], [c_overlay, c_heat, c_crops, c_table])

        with gr.Tab("OCR / Text"):
            _langs = language_choices()
            gr.HTML(note("Recognise the text on a page. <b>Language, page-layout "
                         "mode and engine</b> are all selectable, and languages "
                         "combine with <code>+</code> for mixed-script documents."))
            with gr.Row():
                with gr.Column(scale=3):
                    o_in = gr.Image(label="Page (rectified or raw)", type="numpy",
                                    height=320,
                                    sources=["upload", "webcam", "clipboard"])
                    o_lang = gr.Dropdown(
                        choices=_langs or [("English (eng)", "eng")],
                        value=["eng"] if any(c == "eng" for _, c in _langs) else None,
                        multiselect=True, label="Language(s)",
                        info="Select several for a mixed-script page")
                    o_mode = gr.Dropdown(list(PAGE_MODES), value=next(iter(PAGE_MODES)),
                                         label="Page layout")
                    o_engine = gr.Dropdown(list(ENGINE_MODES), value=next(iter(ENGINE_MODES)),
                                           label="Engine")
                    with gr.Row():
                        o_enh = gr.Checkbox(True, label="Enhance first")
                        o_box = gr.Checkbox(True, label="Show word boxes")
                    o_btn = gr.Button("Recognise text", variant="primary", size="lg")
                    with gr.Row():
                        o_txt = gr.DownloadButton("Download .txt")
                        o_pdf = gr.DownloadButton("Searchable PDF")
                with gr.Column(scale=4):
                    o_stats = gr.HTML()
                    o_img = gr.Image(label="Recognised words (green = confident)",
                                     height=380)
                    o_text = gr.Textbox(label="Text", lines=12)
            o_btn.click(run_ocr_tab, [o_in, o_lang, o_mode, o_engine, o_enh, o_box],
                        [o_img, o_text, o_stats, o_txt, o_pdf])

        with gr.Tab("Degradation Lab"):
            gr.HTML(note("The synthetic engine that trains everything, live. "
                         "Every knob is a term in the generator; the trace below "
                         "lists the exact parameters drawn."))
            with gr.Row():
                with gr.Column(scale=2):
                    d_seed = gr.Slider(0, 9999, value=7, step=1, label="Seed")
                    d_persp = gr.Slider(0.0, 0.30, value=0.16, step=0.01,
                                        label="Perspective jitter")
                    d_rot = gr.Slider(0, 90, value=40, step=1, label="Rotation (deg)")
                    d_shadow = gr.Slider(0.0, 0.8, value=0.45, step=0.02, label="Shadow")
                    d_illum = gr.Slider(0.0, 0.8, value=0.35, step=0.02,
                                        label="Illumination gradient")
                    d_blur = gr.Slider(0.0, 3.0, value=1.2, step=0.1, label="Blur sigma")
                    d_noise = gr.Slider(0.0, 0.08, value=0.02, step=0.002, label="Noise")
                    d_jpeg = gr.Slider(15, 95, value=55, step=1, label="JPEG quality")
                    d_down = gr.Slider(1.0, 5.0, value=2.0, step=0.1,
                                       label="Resolution loss")
                    with gr.Row():
                        d_glare = gr.Checkbox(False, label="Glare")
                        d_curl = gr.Checkbox(False, label="Page curl (not planar!)")
                    d_btn = gr.Button("Generate sample", variant="primary")
                with gr.Column(scale=5):
                    with gr.Row():
                        d_photo = gr.Image(label="Degraded photo + corner labels", height=300)
                        d_rect = gr.Image(label="Rectified (network input)", height=300)
                        d_tgt = gr.Image(label="Clean target", height=300)
                    d_stages = gr.Gallery(label="Pipeline, stage by stage", columns=5,
                                          height=230)
                    d_trace = gr.Markdown()
            d_btn.click(degradation_lab,
                        [d_seed, d_persp, d_rot, d_shadow, d_illum, d_blur, d_noise,
                         d_jpeg, d_down, d_glare, d_curl],
                        [d_photo, d_rect, d_tgt, d_stages, d_trace])

        with gr.Tab("Benchmarks"):
            gr.Markdown("## Measured results")
            b_md = gr.Markdown(benchmarks_md())
            with gr.Row():
                gr.Gallery(value=curve_images(), label="Training curves",
                           columns=3, height=320)
            gr.Gallery(value=figure_images(), label="Qualitative figures",
                       columns=2, height=420)
            b_btn = gr.Button("Reload results")
            b_btn.click(lambda: benchmarks_md(), None, b_md)

        with gr.Tab("How it works"):
            gr.Markdown(ABOUT)

    return demo


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Launch the document scanner UI")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)

    import torch
    torch.set_num_threads(args.threads)

    demo = build_app()
    demo.queue(max_size=16).launch(server_name=args.host, server_port=args.port,
                                   share=args.share, show_error=True,
                                   theme=build_theme(), css=CSS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
