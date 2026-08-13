from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..utils.geometry import (estimate_page_size, order_corners, quad_area,
                              quad_is_convex)
from ..utils.imageio import imread_rgb, imwrite_rgb

__all__ = ["LabelSession", "CORNER_NAMES", "add_or_move_point", "quad_feedback"]

CORNER_NAMES = ("top-left", "top-right", "bottom-right", "bottom-left")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def add_or_move_point(points: list[list[float]], x: float, y: float,
                      snap_px: float = 24.0) -> tuple[list[list[float]], int, bool]:
    pts = [list(map(float, p)) for p in (points or [])]
    if pts:
        d = [float(np.hypot(px - x, py - y)) for px, py in pts]
        nearest = int(np.argmin(d))
        if len(pts) >= 4 or d[nearest] <= snap_px:
            pts[nearest] = [float(x), float(y)]
            return pts, nearest, True
    pts.append([float(x), float(y)])
    return pts, len(pts) - 1, False


def quad_feedback(points: Iterable[Iterable[float]] | None,
                  shape: tuple[int, int]) -> tuple[bool, str]:
    pts = np.asarray(list(points or []), np.float64)
    if pts.shape[0] < 4:
        n = pts.shape[0]
        return False, f"{n}/4 - click the {CORNER_NAMES[n]} corner"
    pts = pts[:4]
    h, w = int(shape[0]), int(shape[1])
    notes: list[str] = []
    ok = True

    if not quad_is_convex(pts):
        return False, "not convex - the corners are out of order or one is misplaced"

    area = quad_area(pts) / float(max(w * h, 1))
    if area < 0.02:
        ok = False
        notes.append(f"page covers only {area * 100:.1f}% of the frame")
    elif area > 1.4:
        notes.append(f"page covers {area * 100:.0f}% of the frame")

    margin = 0.08
    if (pts[:, 0] < -margin * w).any() or (pts[:, 0] > (1 + margin) * w).any() \
       or (pts[:, 1] < -margin * h).any() or (pts[:, 1] > (1 + margin) * h).any():
        notes.append("a corner lies well outside the image")

    canonical = order_corners(pts)
    if not np.allclose(canonical, pts.astype(np.float32), atol=1.5):
        notes.append("will be re-ordered to canonical TL,TR,BR,BL on save")

    pw, ph = estimate_page_size(canonical, (h, w))
    aspect = max(pw, ph) / max(min(pw, ph), 1)
    if aspect > 6.0:
        notes.append(f"implied page aspect {aspect:.1f}:1 is unusual")

    if ok and not notes:
        return True, f"looks good - {aspect:.2f}:1 page, {area * 100:.0f}% of frame"
    return ok, ("; ".join(notes) if notes else "ok")


@dataclass
class LabelSession:
    out_dir: Path
    name: str = "own-photos"
    files: list[str] = field(default_factory=list)
    corners: dict[str, list[list[float]]] = field(default_factory=dict)
    index: int = 0

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        (self.out_dir / "photos").mkdir(parents=True, exist_ok=True)

    @property
    def manifest_path(self) -> Path:
        return self.out_dir / "annotations.json"

    @classmethod
    def load(cls, out_dir: str | Path, name: str = "own-photos") -> "LabelSession":
        out_dir = Path(out_dir)
        s = cls(out_dir=out_dir, name=name)
        if s.manifest_path.exists():
            data = json.loads(s.manifest_path.read_text())
            s.name = data.get("name", name)
            for item in data.get("items", []):
                rel = item["file"]
                if not (out_dir / rel).exists():
                    continue
                s.files.append(rel)
                pts = item.get("corners")
                if pts:
                    s.corners[rel] = [list(map(float, p)) for p in pts]
        for p in sorted((out_dir / "photos").glob("*")):
            rel = f"photos/{p.name}"
            if p.suffix.lower() in IMAGE_SUFFIXES and rel not in s.files:
                s.files.append(rel)
        return s

    def add_images(self, paths: Iterable[str | Path]) -> list[str]:
        added: list[str] = []
        for src in paths:
            src = Path(src)
            if not src.exists() or src.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            stem = _unique_stem(self.out_dir / "photos", src.stem)
            dst = self.out_dir / "photos" / f"{stem}{src.suffix.lower()}"
            try:
                shutil.copyfile(src, dst)
            except OSError:
                imwrite_rgb(dst.with_suffix(".jpg"), imread_rgb(src), quality=96)
                dst = dst.with_suffix(".jpg")
            rel = f"photos/{dst.name}"
            self.files.append(rel)
            added.append(rel)
        return added

    def __len__(self) -> int:
        return len(self.files)

    @property
    def current(self) -> str | None:
        if not self.files:
            return None
        self.index = int(np.clip(self.index, 0, len(self.files) - 1))
        return self.files[self.index]

    def goto(self, i: int) -> str | None:
        if not self.files:
            return None
        self.index = int(np.clip(i, 0, len(self.files) - 1))
        return self.current

    def step(self, delta: int) -> str | None:
        return self.goto(self.index + delta)

    def next_unlabelled(self) -> str | None:
        n = len(self.files)
        for k in range(1, n + 1):
            i = (self.index + k) % n if n else 0
            if self.files and self.files[i] not in self.corners:
                return self.goto(i)
        return self.current

    def image(self, key: str | None = None) -> np.ndarray | None:
        key = key or self.current
        if key is None:
            return None
        path = self.out_dir / key
        return imread_rgb(path) if path.exists() else None

    def get(self, key: str | None = None) -> list[list[float]] | None:
        key = key or self.current
        return None if key is None else self.corners.get(key)

    def set(self, points: Iterable[Iterable[float]] | None,
            key: str | None = None, save: bool = True) -> None:
        key = key or self.current
        if key is None:
            return
        pts = list(points or [])
        if len(pts) >= 4:
            self.corners[key] = order_corners(np.asarray(pts[:4], np.float32)).tolist()
        else:
            self.corners.pop(key, None)
        if save:
            self.save()

    def clear(self, key: str | None = None, save: bool = True) -> None:
        self.set(None, key, save=save)

    def remove_current(self, delete_file: bool = False) -> None:
        key = self.current
        if key is None:
            return
        self.files.pop(self.index)
        self.corners.pop(key, None)
        if delete_file:
            (self.out_dir / key).unlink(missing_ok=True)
        self.index = int(np.clip(self.index, 0, max(0, len(self.files) - 1)))
        self.save()

    def items(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for rel in self.files:
            entry: dict[str, Any] = {"file": rel, "meta": {"source": "labeler"}}
            pts = self.corners.get(rel)
            if pts:
                entry["corners"] = np.round(np.asarray(pts, np.float64), 2).tolist()
            out.append(entry)
        return out

    def save(self) -> Path:
        items = self.items()
        done = [it for it in items if "corners" in it]
        pending = [it["file"] for it in items if "corners" not in it]
        payload = {"name": self.name, "count": len(done), "items": done}
        if pending:
            payload["pending"] = pending
        self.manifest_path.write_text(json.dumps(payload, indent=1))
        return self.manifest_path

    def stats(self) -> dict[str, int]:
        done = sum(1 for f in self.files if f in self.corners)
        return {"total": len(self.files), "labelled": done,
                "remaining": len(self.files) - done}

    def export_coco(self, path: str | Path) -> Path:
        images, annotations = [], []
        for i, rel in enumerate(self.files, start=1):
            pts = self.corners.get(rel)
            if not pts:
                continue
            img = self.image(rel)
            if img is None:
                continue
            h, w = img.shape[:2]
            images.append({"id": i, "file_name": Path(rel).name,
                           "width": int(w), "height": int(h)})
            flat: list[float] = []
            for x, y in pts:
                flat += [round(float(x), 2), round(float(y), 2), 2]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
            annotations.append({
                "id": i, "image_id": i, "category_id": 1, "iscrowd": 0,
                "keypoints": flat, "num_keypoints": 4,
                "bbox": [round(float(v), 2) for v in bbox],
                "area": round(float(quad_area(np.asarray(pts, np.float64))), 2),
            })
        coco = {
            "info": {"description": f"{self.name} - page corners"},
            "images": images,
            "annotations": annotations,
            "categories": [{
                "id": 1, "name": "page", "supercategory": "document",
                "keypoints": list(CORNER_NAMES),
                "skeleton": [[1, 2], [2, 3], [3, 4], [4, 1]],
            }],
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(coco, indent=1))
        return path


def _unique_stem(folder: Path, stem: str) -> str:
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in stem) or "photo"
    if not any(folder.glob(f"{safe}.*")):
        return safe
    k = 2
    while any(folder.glob(f"{safe}_{k}.*")):
        k += 1
    return f"{safe}_{k}"
