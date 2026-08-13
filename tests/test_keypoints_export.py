from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path("data/real/own")
KEYPOINTS = ROOT / "coco_keypoints.json"

pytestmark = pytest.mark.skipif(
    not KEYPOINTS.exists(), reason="keypoint export not built")


def test_keypoints_match_the_manifest():
    from docscanner.utils.geometry import order_corners

    manifest = json.loads((ROOT / "annotations.json").read_text())
    items = manifest["items"] if isinstance(manifest, dict) else manifest
    coco = json.loads(KEYPOINTS.read_text())

    assert len(coco["annotations"]) == len(items)
    assert coco["categories"][0]["keypoints"] == [
        "top_left", "top_right", "bottom_right", "bottom_left"]

    by_name = {im["id"]: im["file_name"] for im in coco["images"]}
    wanted = {Path(it["file"]).name:
              order_corners(np.asarray(it["corners"], np.float32))
              for it in items}

    for ann in coco["annotations"]:
        pts = np.asarray(ann["keypoints"], np.float32).reshape(4, 3)
        assert (pts[:, 2] == 2).all(), "every corner is annotated and visible"
        got = pts[:, :2]
        expect = wanted[by_name[ann["image_id"]]]
        assert np.allclose(got, expect, atol=0.01), by_name[ann["image_id"]]


def test_the_export_is_readable_by_the_keypoint_ingester(tmp_path):
    from docscanner.data.real import from_coco_keypoints

    out = from_coco_keypoints(KEYPOINTS, ROOT / "photos", tmp_path, name="kp")
    written = json.loads(Path(out).read_text())
    items = written["items"] if isinstance(written, dict) else written
    assert len(items) == len(json.loads(KEYPOINTS.read_text())["annotations"])
