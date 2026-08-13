from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..eval.ocr import detect_orientation, tesseract_available
from ..utils.geometry import estimate_page_size, homography, order_corners
from ..utils.imageio import ensure_rgb
from .corner_pipeline import CornerPipeline, CornerResult
from .enhance_pipeline import EnhancementPipeline, OutputMode, apply_output_mode

__all__ = ["DocumentScanner", "ScanResult", "rectify_with_corners",
           "auto_orient", "save_pdf"]


@dataclass
class ScanResult:
    image: np.ndarray
    rectified: np.ndarray
    photo: np.ndarray
    corners: np.ndarray
    corner_result: CornerResult | None = None
    rotation_applied: int = 0
    mode: str = "color"
    seconds: float = 0.0
    timings: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


def rectify_with_corners(
    photo: np.ndarray,
    corners: np.ndarray,
    max_side: int = 1600,
    min_side: int = 320,
) -> tuple[np.ndarray, np.ndarray]:
    img = ensure_rgb(np.asarray(photo))
    quad = order_corners(corners)
    w, h = estimate_page_size(quad, img.shape[:2])

    scale = 1.0
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
    if min(w, h) * scale < min_side:
        scale = min_side / float(min(w, h))
    w = max(16, int(round(w * scale)))
    h = max(16, int(round(h * scale)))

    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    h_mat = homography(quad, dst)
    page = cv2.warpPerspective(img, h_mat, (w, h), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)
    return page, h_mat


def text_axis_score(image: np.ndarray) -> float:
    g = cv2.cvtColor(ensure_rgb(image), cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, (512, 512), interpolation=cv2.INTER_AREA).astype(np.float32)
    ink = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    ink = np.maximum(ink - ink.mean(), 0)
    row_var = float(np.var(ink.sum(axis=1)))
    col_var = float(np.var(ink.sum(axis=0)))
    return (row_var - col_var) / (row_var + col_var + 1e-6)


def _projection_orientation(image: np.ndarray) -> int:
    return 0 if text_axis_score(image) >= 0 else 90


def _rotate(img: np.ndarray, angle: int) -> np.ndarray:
    k = {90: cv2.ROTATE_90_COUNTERCLOCKWISE,
         180: cv2.ROTATE_180,
         270: cv2.ROTATE_90_CLOCKWISE}.get(angle % 360)
    return img if k is None else cv2.rotate(img, k)


_ORIENT_MARGIN = 2.0
_ORIENT_MIN_WORDS = 20
_ORIENT_WORD_KEEP = 0.85
_ORIENT_AXIS_MARGIN = 0.05
_ORIENT_AXIS_CONFIDENT = 0.30
_ORIENT_READS_WELL = 70.0


def auto_orient(image: np.ndarray, use_osd: bool = True,
                verify: bool = True) -> tuple[np.ndarray, int]:
    from ..eval.ocr import ocr_backends, run_ocr

    img = ensure_rgb(np.asarray(image))

    proposals: set[int] = set()
    axis_sideways = text_axis_score(img) < -_ORIENT_AXIS_MARGIN
    if axis_sideways:
        proposals.update({90, 270})
    if use_osd and tesseract_available():
        osd = detect_orientation(img) % 360
        if osd == 180 or (osd in (90, 270) and axis_sideways):
            proposals.add(osd)

    proposals.discard(0)
    if not proposals:
        return img, 0

    engines = ocr_backends()
    if not (verify and engines):
        angle = sorted(proposals)[0]
        return _rotate(img, angle), angle % 360

    engine = "tesseract" if "tesseract" in engines else engines[0]

    def _score(candidate: np.ndarray) -> tuple[float, int]:
        h, w = candidate.shape[:2]
        if max(h, w) > 900:
            f = 900.0 / max(h, w)
            candidate = cv2.resize(candidate, (max(1, int(w * f)), max(1, int(h * f))),
                                   interpolation=cv2.INTER_AREA)
        r = run_ocr(candidate, backend=engine)
        return r.mean_confidence, r.words

    base_conf, base_words = _score(img)
    if (text_axis_score(img) < -_ORIENT_AXIS_CONFIDENT
            and base_words >= _ORIENT_MIN_WORDS
            and base_conf < _ORIENT_READS_WELL):
        scored = {a: _score(_rotate(img, a)) for a in (90, 270)}
        readable = {a: (c, w) for a, (c, w) in scored.items()
                    if w >= _ORIENT_MIN_WORDS}
        if readable:
            angle = max(readable, key=lambda a: (readable[a][1], readable[a][0]))
            return _rotate(img, angle), angle % 360

    best_angle, best_conf = 0, base_conf
    for angle in sorted(proposals):
        if angle in (90, 270) and base_words < _ORIENT_MIN_WORDS:
            continue
        conf, words = _score(_rotate(img, angle))
        if words < _ORIENT_MIN_WORDS:
            continue
        if words < _ORIENT_WORD_KEEP * base_words:
            continue
        if conf > best_conf + _ORIENT_MARGIN:
            best_angle, best_conf = angle, conf
    if best_angle == 0:
        return img, 0
    return _rotate(img, best_angle), best_angle % 360


def save_pdf(pages: Sequence[np.ndarray], path: str | Path, dpi: int = 200) -> Path:
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imgs = [Image.fromarray(ensure_rgb(np.asarray(p))).convert("RGB") for p in pages]
    if not imgs:
        raise ValueError("save_pdf: no pages")
    imgs[0].save(path, "PDF", resolution=float(dpi), save_all=True,
                 append_images=imgs[1:])
    return path


class DocumentScanner:
    def __init__(
        self,
        corner_checkpoint: str | Path,
        enhance_checkpoint: str | Path,
        device: str = "cpu",
        threads: int | None = None,
        max_side: int = 1600,
        refine_corners: bool = False,
        tta: bool = True,
        preserve_tone: float = 0.0,
    ):
        self.corners = CornerPipeline(corner_checkpoint, device=device,
                                      threads=threads, refine=refine_corners,
                                      tta=tta)
        self.enhancer = EnhancementPipeline(enhance_checkpoint, device=device,
                                            max_side=max_side,
                                            preserve_tone=preserve_tone)
        self.max_side = max_side

    def scan(
        self,
        photo: np.ndarray,
        mode: OutputMode = "color",
        corners: np.ndarray | None = None,
        auto_orientation: bool = True,
        enhance: bool = True,
        refine: bool | None = None,
        tta: bool | None = None,
    ) -> ScanResult:
        t_start = time.time()
        timings: dict[str, float] = {}
        img = ensure_rgb(np.asarray(photo))

        corner_result = None
        if corners is None:
            t0 = time.time()
            corner_result = self.corners(img, refine=refine, tta=tta)
            corners = corner_result.corners
            timings["detect"] = time.time() - t0
        corners = order_corners(np.asarray(corners, np.float32))

        t0 = time.time()
        rectified, _ = rectify_with_corners(img, corners, max_side=self.max_side)
        timings["rectify"] = time.time() - t0

        page = rectified
        if enhance:
            t0 = time.time()
            page = self.enhancer(rectified, mode="raw").image
            timings["enhance"] = time.time() - t0

        rotation = 0
        if auto_orientation:
            t0 = time.time()
            page, rotation = auto_orient(page)
            if rotation:
                k = {90: cv2.ROTATE_90_COUNTERCLOCKWISE, 180: cv2.ROTATE_180,
                     270: cv2.ROTATE_90_CLOCKWISE}[rotation]
                rectified = cv2.rotate(rectified, k)
            timings["orient"] = time.time() - t0

        styled = apply_output_mode(page, mode)

        return ScanResult(
            image=styled,
            rectified=rectified,
            photo=img,
            corners=corners,
            corner_result=corner_result,
            rotation_applied=rotation,
            mode=mode,
            seconds=time.time() - t_start,
            timings=timings,
            meta={"page_size": (styled.shape[1], styled.shape[0]),
                  "confidence": getattr(corner_result, "confidence", 1.0),
                  "refined": getattr(corner_result, "refined", False),
                  "corner_source": getattr(corner_result, "source", "supplied")},
        )

    def scan_many(self, photos: Iterable[np.ndarray], **kw) -> list[ScanResult]:
        return [self.scan(p, **kw) for p in photos]

    def scan_to_pdf(self, photos: Iterable[np.ndarray], path: str | Path,
                    **kw) -> Path:
        results = self.scan_many(photos, **kw)
        return save_pdf([r.image for r in results], path)
