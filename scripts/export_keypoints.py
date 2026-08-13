#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

KEYPOINT_NAMES = ["top_left", "top_right", "bottom_right", "bottom_left"]


def build(manifest: Path, root: Path) -> dict:
    import numpy as np

    from docscanner.utils.geometry import order_corners
    from docscanner.utils.imageio import imread_rgb

    data = json.loads(manifest.read_text())
    items = data["items"] if isinstance(data, dict) else data

    images, annotations = [], []
    for i, it in enumerate(items, start=1):
        rel = it["file"]
        img = imread_rgb(root / rel)
        h, w = img.shape[:2]
        pts = order_corners(np.asarray(it["corners"], np.float32))

        images.append({"id": i, "file_name": Path(rel).name,
                       "width": int(w), "height": int(h)})
        flat: list[float] = []
        for x, y in pts:
            flat += [round(float(x), 2), round(float(y), 2), 2]
        xs, ys = pts[:, 0], pts[:, 1]
        annotations.append({
            "id": i, "image_id": i, "category_id": 1,
            "keypoints": flat, "num_keypoints": 4,
            "bbox": [round(float(xs.min()), 2), round(float(ys.min()), 2),
                     round(float(xs.max() - xs.min()), 2),
                     round(float(ys.max() - ys.min()), 2)],
            "area": round(float((xs.max() - xs.min()) * (ys.max() - ys.min())), 2),
            "iscrowd": 0,
        })

    return {
        "info": {"description": "AmazingScanner — page corners as COCO keypoints",
                 "note": "Same labels as annotations.json, re-expressed in the "
                         "format Section 1.2 names. Order is TL, TR, BR, BL."},
        "images": images,
        "annotations": annotations,
        "categories": [{
            "id": 1, "name": "page", "supercategory": "document",
            "keypoints": KEYPOINT_NAMES,
            "skeleton": [[1, 2], [2, 3], [3, 4], [4, 1]],
        }],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="data/real/own")
    ap.add_argument("--out", default=None,
                    help="default: <root>/coco_keypoints.json")
    args = ap.parse_args(argv)

    root = Path(args.root)
    out = Path(args.out) if args.out else root / "coco_keypoints.json"
    payload = build(root / "annotations.json", root)
    out.write_text(json.dumps(payload, indent=1))
    print(f"wrote {len(payload['annotations'])} keypoint annotations -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
