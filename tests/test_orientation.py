from __future__ import annotations

import cv2
import numpy as np
import pytest

from docscanner.eval.ocr import ocr_available
from docscanner.pipeline.scanner import auto_orient, text_axis_score

pytestmark = pytest.mark.skipif(not ocr_available(),
                                reason="orientation verification needs an OCR engine")

_LINE = "the quick brown fox jumps over the lazy dog while reading pages"


def _text_page(w: int = 640, h: int = 880) -> np.ndarray:
    page = np.full((h, w, 3), 246, np.uint8)
    y = 54
    while y < h - 30:
        cv2.putText(page, _LINE, (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (30, 30, 36), 1, cv2.LINE_AA)
        y += 30
    return page


def test_upright_page_is_left_alone():
    _, rotation = auto_orient(_text_page())
    assert rotation == 0


def test_readable_sideways_page_is_left_alone():
    sideways = cv2.rotate(_text_page(), cv2.ROTATE_90_CLOCKWISE)
    assert text_axis_score(sideways) < -0.3, "fixture must measure sideways"
    _, rotation = auto_orient(sideways)
    assert rotation == 0


def test_near_textless_page_is_never_turned():
    cover = np.full((880, 640, 3), 235, np.uint8)
    cv2.rectangle(cover, (80, 120), (560, 300), (40, 42, 60), -1)
    _, rotation = auto_orient(cover)
    assert rotation == 0


@pytest.mark.skipif(not __import__("pathlib").Path(
    "data/real/own/annotations.json").exists(),
    reason="real photographs not present")
def test_photographed_page_fed_sideways_is_turned_back():
    import json
    from pathlib import Path

    from docscanner.pipeline.enhance_pipeline import EnhancementPipeline
    from docscanner.utils.geometry import order_corners, rectify
    from docscanner.utils.imageio import imread_rgb

    manifest = json.loads(Path("data/real/own/annotations.json").read_text())
    item = next(it for it in manifest["items"]
                if Path(it["file"]).name.startswith("img10"))
    img = imread_rgb(Path("data/real/own") / item["file"])
    corners = order_corners(np.asarray(item["corners"], np.float32))
    page = rectify(img, corners, max_side=900)[0]
    page = EnhancementPipeline("models/enhance.pt", max_side=900)(
        page, mode="raw").image

    upright, base_rot = auto_orient(page)
    sideways = cv2.rotate(upright, cv2.ROTATE_90_CLOCKWISE)
    out, rotation = auto_orient(sideways)
    assert rotation in (90, 270)
    small = cv2.resize(cv2.cvtColor(out, cv2.COLOR_RGB2GRAY), (128, 128))
    ref = cv2.resize(cv2.cvtColor(upright, cv2.COLOR_RGB2GRAY), (128, 128))
    corr = float(np.corrcoef(small.ravel(), ref.ravel())[0, 1])
    assert corr > 0.8, f"turned the wrong way (corr {corr:.2f})"
