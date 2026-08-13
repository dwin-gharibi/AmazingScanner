from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

PACK = Path("data/test_pack")

pytestmark = pytest.mark.skipif(not PACK.exists(),
                                reason="test pack not generated")


def _manifest() -> dict:
    return json.loads((PACK / "ground_truth" / "corners.json").read_text())


def test_every_manifest_entry_resolves():
    m = _manifest()
    assert m["count"] == len(m["items"]) > 0
    for it in m["items"]:
        assert (PACK / it["file"]).exists(), it["file"]
        assert (PACK / it["target"]).exists(), it["target"]


def test_corners_are_a_plausible_page():
    import cv2
    m = _manifest()
    for it in m["items"]:
        img = cv2.imread(str(PACK / it["file"]))
        h, w = img.shape[:2]
        q = np.asarray(it["corners"], np.float32)
        assert q.shape == (4, 2)
        assert q[:, 0].min() > -w * 0.25 and q[:, 0].max() < w * 1.25
        assert q[:, 1].min() > -h * 0.25 and q[:, 1].max() < h * 1.25
        area = cv2.contourArea(q.reshape(-1, 1, 2))
        assert area > 0.02 * w * h, f"{it['file']}: page covers {area/(w*h):.1%}"


def test_styles_are_all_represented():
    per = json.loads((PACK / "styles.json").read_text())
    assert set(per) == {"clean", "corner-hard", "ood"}
    assert all(v > 0 for v in per.values())
