from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

__all__ = ["imread_rgb", "imwrite_rgb", "resize_to", "fit_within", "pad_to_multiple",
           "ensure_rgb", "load_gray"]


def _apply_exif_orientation(path: str | os.PathLike, img: np.ndarray) -> np.ndarray:
    try:
        from PIL import Image
    except Exception:
        return img
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            orient = exif.get(274, 1) if exif else 1
    except Exception:
        return img
    if orient == 3:
        return cv2.rotate(img, cv2.ROTATE_180)
    if orient == 6:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if orient == 8:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def ensure_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return img


def imread_rgb(path: str | os.PathLike, exif: bool = True) -> np.ndarray:
    path = str(path)
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"could not decode image: {path}")
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if exif:
        img = _apply_exif_orientation(path, img)
    return np.ascontiguousarray(img)


def load_gray(path: str | os.PathLike) -> np.ndarray:
    img = imread_rgb(path)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def imwrite_rgb(path: str | os.PathLike, img: np.ndarray, quality: int = 95) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = ensure_rgb(np.asarray(img))
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ext = path.suffix.lower()
    params = [int(cv2.IMWRITE_JPEG_QUALITY), quality] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext if ext else ".png", bgr, params)
    if not ok:
        raise OSError(f"could not encode image for {path}")
    buf.tofile(str(path))


def resize_to(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    if img.shape[1] == w and img.shape[0] == h:
        return img
    shrinking = w * h < img.shape[0] * img.shape[1]
    interp = cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC
    return cv2.resize(img, (w, h), interpolation=interp)


def fit_within(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    s = max_side / float(m)
    return cv2.resize(img, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                      interpolation=cv2.INTER_AREA)


def pad_to_multiple(img: np.ndarray, multiple: int = 16,
                    mode: int = cv2.BORDER_REFLECT) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = img.shape[:2]
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph == 0 and pw == 0:
        return img, (0, 0)
    out = cv2.copyMakeBorder(img, 0, ph, 0, pw, mode)
    return out, (ph, pw)
