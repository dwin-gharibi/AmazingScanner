from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from ..utils.geometry import order_corners, quad_area, quad_is_convex
from ..utils.imageio import imwrite_rgb

__all__ = [
    "quad_from_mask",
    "quad_from_polygon",
    "photo_key",
    "build_manifest",
    "from_coco_keypoints",
    "from_coco_segmentation",
    "from_coco",
    "from_midv_parquet",
]


def quad_from_mask(mask: np.ndarray, min_area_frac: float = 0.02) -> np.ndarray | None:
    if mask.ndim == 3:
        mask = mask[..., 0]
    binary = (mask > 127).astype(np.uint8)
    h, w = binary.shape
    if binary.sum() < min_area_frac * h * w:
        return None

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < min_area_frac * h * w:
        return None
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)

    lo, hi = 0.001, 0.15
    best = None
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        approx = cv2.approxPolyDP(hull, mid * peri, True)
        n = len(approx)
        if n == 4:
            best = approx.reshape(4, 2).astype(np.float32)
            break
        if n > 4:
            lo = mid
        else:
            hi = mid
    if best is None:
        rect = cv2.minAreaRect(hull)
        best = cv2.boxPoints(rect).astype(np.float32)

    quad = order_corners(best)
    if not quad_is_convex(quad) or quad_area(quad) < min_area_frac * h * w:
        return None
    return quad


def quad_from_polygon(flat: Iterable[float]) -> np.ndarray | None:
    pts = np.asarray(list(flat), np.float32)
    if pts.size < 8 or pts.size % 2:
        return None
    pts = pts.reshape(-1, 2)
    if not np.isfinite(pts).all():
        return None

    if len(pts) == 4:
        quad = order_corners(pts)
        return quad if quad_is_convex(quad) else None

    hull = cv2.convexHull(pts.astype(np.float32))
    peri = cv2.arcLength(hull, True)
    best = None
    lo, hi = 0.001, 0.15
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        approx = cv2.approxPolyDP(hull, mid * peri, True)
        n = len(approx)
        if n == 4:
            best = approx.reshape(4, 2).astype(np.float32)
            break
        if n > 4:
            lo = mid
        else:
            hi = mid
    if best is None:
        best = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)
    quad = order_corners(best)
    return quad if quad_is_convex(quad) else None


def quad_touches_border(quad: np.ndarray, shape: tuple[int, int], tol: float = 2.0) -> bool:
    h, w = shape[:2]
    q = np.asarray(quad, np.float32)
    return bool(
        (q[:, 0] <= tol).any() or (q[:, 1] <= tol).any()
        or (q[:, 0] >= w - 1 - tol).any() or (q[:, 1] >= h - 1 - tol).any()
    )


def build_manifest(out_dir: str | Path, name: str, items: list[dict[str, Any]]) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "annotations.json"
    path.write_text(json.dumps({"name": name, "count": len(items), "items": items}, indent=1))
    return path


def from_coco_keypoints(
    coco_json: str | Path,
    images_dir: str | Path,
    out_dir: str | Path,
    name: str = "real-photos",
    reference_dir: str | Path | None = None,
    keypoint_order: Iterable[str] = ("top-left", "top-right", "bottom-right", "bottom-left"),
) -> Path:
    coco = json.loads(Path(coco_json).read_text())
    images_dir, out_dir = Path(images_dir), Path(out_dir)
    (out_dir / "photos").mkdir(parents=True, exist_ok=True)

    id2img = {im["id"]: im for im in coco["images"]}
    cats = {c["id"]: c for c in coco.get("categories", [])}
    items: list[dict[str, Any]] = []

    for ann in coco.get("annotations", []):
        kps = ann.get("keypoints")
        if not kps:
            continue
        pts = np.asarray(kps, np.float32).reshape(-1, 3)
        names = cats.get(ann.get("category_id"), {}).get("keypoints") or list(keypoint_order)
        keep = [(n, p) for n, p in zip(names, pts) if p[2] > 0]
        if len(keep) < 4:
            continue
        quad = np.stack([p[:2] for _, p in keep[:4]])
        quad = order_corners(quad)

        im = id2img[ann["image_id"]]
        src = images_dir / im["file_name"]
        if not src.exists():
            continue
        dst_rel = f"photos/{Path(im['file_name']).stem}.jpg"
        dst = out_dir / dst_rel
        if not dst.exists():
            from ..utils.imageio import imread_rgb
            imwrite_rgb(dst, imread_rgb(src), quality=96)

        entry: dict[str, Any] = {
            "file": dst_rel,
            "corners": np.round(quad, 2).tolist(),
            "meta": {"source": "coco-keypoints", "original": im["file_name"]},
        }
        _attach_reference(entry, Path(im["file_name"]).stem, out_dir, reference_dir)
        items.append(entry)

    return build_manifest(out_dir, name, items)


def _stash_photo(src: Path, out_dir: Path, stem: str) -> str:
    from ..utils.imageio import imread_rgb

    dst_rel = f"photos/{stem}.jpg"
    dst = out_dir / dst_rel
    if not dst.exists():
        imwrite_rgb(dst, imread_rgb(src), quality=96)
    return dst_rel


_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp",
                             ".tif", ".tiff"})

_PAIR_SUFFIXES = ("_main", "_photo", "_raw", "_orig", "_original",
                  "_scanned", "_scan", "_ref", "_reference", "_camscanner")


def photo_key(name: str) -> str:
    text = str(name)
    path = Path(text)
    stem = path.stem if path.suffix.lower() in _IMAGE_SUFFIXES else path.name
    stem = re.sub(r"_jpe?g\.rf\.[0-9a-f]+$", "", stem, flags=re.I)
    stem = re.sub(r"\.rf\.[0-9a-f]+$", "", stem, flags=re.I)
    low = stem.lower()
    for suf in sorted(_PAIR_SUFFIXES, key=len, reverse=True):
        if low.endswith(suf):
            stem = stem[: -len(suf)]
            break
    return stem.strip("_-. ") or Path(str(name)).stem


def _attach_reference(entry: dict[str, Any], stem: str, out_dir: Path,
                      reference_dir: str | Path | None) -> None:
    if not reference_dir:
        return
    from ..utils.imageio import imread_rgb

    ref_dir = Path(reference_dir)
    if not ref_dir.is_dir():
        return
    want = photo_key(stem)
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    candidates = sorted(p for p in ref_dir.iterdir()
                        if p.is_file() and p.suffix.lower() in exts)
    match = next((p for p in candidates if p.stem == stem), None)
    if match is None:
        match = next((p for p in candidates if photo_key(p.name) == want), None)
    if match is None:
        return
    ref_rel = f"reference/{want}.jpg"
    (out_dir / "reference").mkdir(parents=True, exist_ok=True)
    imwrite_rgb(out_dir / ref_rel, imread_rgb(match), quality=96)
    entry["reference"] = ref_rel


def from_coco_segmentation(
    coco_json: str | Path,
    images_dir: str | Path,
    out_dir: str | Path,
    name: str = "real-photos",
    reference_dir: str | Path | None = None,
) -> Path:
    coco = json.loads(Path(coco_json).read_text())
    images_dir, out_dir = Path(images_dir), Path(out_dir)
    (out_dir / "photos").mkdir(parents=True, exist_ok=True)
    if reference_dir:
        (out_dir / "reference").mkdir(parents=True, exist_ok=True)

    id2img = {im["id"]: im for im in coco.get("images", [])}
    items: list[dict[str, Any]] = []
    skipped: list[str] = []

    for ann in coco.get("annotations", []):
        seg = ann.get("segmentation")
        if not seg:
            continue
        if isinstance(seg, dict):
            skipped.append(f"image_id={ann.get('image_id')}: RLE segmentation")
            continue
        ring = max(seg, key=len) if isinstance(seg[0], (list, tuple)) else seg
        quad = quad_from_polygon(ring)
        im = id2img.get(ann.get("image_id"))
        if im is None:
            continue
        if quad is None:
            skipped.append(f"{im['file_name']}: polygon is not a usable quad")
            continue

        src = images_dir / im["file_name"]
        if not src.exists():
            skipped.append(f"{im['file_name']}: image file missing")
            continue
        stem = Path(im["file_name"]).stem
        entry: dict[str, Any] = {
            "file": _stash_photo(src, out_dir, stem),
            "corners": np.round(quad, 2).tolist(),
            "meta": {"source": "coco-segmentation", "original": im["file_name"],
                     "size": [im.get("width"), im.get("height")]},
        }
        _attach_reference(entry, stem, out_dir, reference_dir)
        items.append(entry)

    if skipped:
        print(f"  [warn] {len(skipped)} annotation(s) skipped:")
        for s in skipped[:10]:
            print(f"    - {s}")
    return build_manifest(out_dir, name, items)


def from_coco(coco_json: str | Path, images_dir: str | Path, out_dir: str | Path,
              name: str = "real-photos",
              reference_dir: str | Path | None = None) -> Path:
    coco = json.loads(Path(coco_json).read_text())
    anns = coco.get("annotations", [])
    has_kps = any(a.get("keypoints") for a in anns)
    has_seg = any(a.get("segmentation") for a in anns)
    if has_kps:
        return from_coco_keypoints(coco_json, images_dir, out_dir, name, reference_dir)
    if has_seg:
        return from_coco_segmentation(coco_json, images_dir, out_dir, name, reference_dir)
    raise ValueError(f"{coco_json}: no keypoint or segmentation annotations found")


def from_midv_parquet(
    parquets: Iterable[str | Path],
    out_dir: str | Path,
    name: str = "midv500-real",
    limit: int = 200,
    long_side: int = 1280,
    min_area_frac: float = 0.03,
    skip_border: bool = True,
    stride: int = 1,
) -> Path:
    import pyarrow.parquet as pq

    out_dir = Path(out_dir)
    (out_dir / "photos").mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    seen = 0

    for shard in parquets:
        if len(items) >= limit:
            break
        pf = pq.ParquetFile(str(shard))
        for batch in pf.iter_batches(batch_size=32):
            for row in batch.to_pylist():
                seen += 1
                if len(items) >= limit:
                    break
                if (seen - 1) % stride != 0:
                    continue
                try:
                    img = cv2.imdecode(np.frombuffer(row["pixel_values"]["bytes"], np.uint8),
                                       cv2.IMREAD_COLOR)
                    msk = cv2.imdecode(np.frombuffer(row["label"]["bytes"], np.uint8),
                                       cv2.IMREAD_UNCHANGED)
                except Exception:
                    continue
                if img is None or msk is None:
                    continue
                if msk.shape[:2] != img.shape[:2]:
                    msk = cv2.resize(msk, (img.shape[1], img.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
                quad = quad_from_mask(msk, min_area_frac=min_area_frac)
                if quad is None:
                    continue
                if skip_border and quad_touches_border(quad, img.shape[:2], tol=3.0):
                    continue

                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                scale = long_side / float(max(h, w))
                if scale < 1.0:
                    rgb = cv2.resize(rgb, (int(round(w * scale)), int(round(h * scale))),
                                     interpolation=cv2.INTER_AREA)
                    quad = quad * scale

                stem = Path(str(row["pixel_values"].get("path") or f"midv_{len(items):04d}")).stem
                rel = f"photos/{len(items):04d}_{stem}.jpg"
                imwrite_rgb(out_dir / rel, rgb, quality=94)
                items.append({
                    "file": rel,
                    "corners": np.round(order_corners(quad), 2).tolist(),
                    "meta": {"source": "midv500", "frame": stem,
                             "size": [rgb.shape[1], rgb.shape[0]]},
                })
            if len(items) >= limit:
                break

    return build_manifest(out_dir, name, items)
