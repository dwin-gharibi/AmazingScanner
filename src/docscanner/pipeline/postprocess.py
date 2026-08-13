from __future__ import annotations

import cv2
import numpy as np

from ..utils.imageio import ensure_rgb

__all__ = ["adjust", "rotate", "deskew", "estimate_skew", "auto_crop", "add_margin"]


def adjust(image: np.ndarray, brightness: float = 0.0, contrast: float = 1.0,
           sharpness: float = 0.0, saturation: float = 1.0) -> np.ndarray:
    img = ensure_rgb(np.asarray(image)).astype(np.float32) / 255.0

    if contrast != 1.0 or brightness != 0.0:
        img = (img - 0.5) * float(contrast) + 0.5 + float(brightness)
        np.clip(img, 0.0, 1.0, out=img)

    if saturation != 1.0:
        luma = np.array([0.299, 0.587, 0.114], np.float32)
        m = saturation * np.eye(3, np.float32) + (1 - saturation) * np.tile(luma, (3, 1))
        img = np.clip(cv2.transform(img, m), 0.0, 1.0)

    if sharpness > 0.01:
        sigma = max(0.8, min(img.shape[:2]) / 600.0)
        blur = cv2.GaussianBlur(img, (0, 0), sigma)
        img = np.clip(img + float(sharpness) * (img - blur), 0.0, 1.0)

    return (img * 255.0 + 0.5).astype(np.uint8)


def rotate(image: np.ndarray, degrees: int) -> np.ndarray:
    k = int(round(degrees / 90.0)) % 4
    if k == 0:
        return np.asarray(image)
    code = {1: cv2.ROTATE_90_CLOCKWISE, 2: cv2.ROTATE_180,
            3: cv2.ROTATE_90_COUNTERCLOCKWISE}[k]
    return cv2.rotate(np.asarray(image), code)


def estimate_skew(image: np.ndarray, max_angle: float = 8.0) -> float:
    g = cv2.cvtColor(ensure_rgb(np.asarray(image)), cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, (480, int(480 * g.shape[0] / max(g.shape[1], 1))),
                   interpolation=cv2.INTER_AREA)
    ink = 255.0 - cv2.GaussianBlur(g, (0, 0), 1.0).astype(np.float32)
    ink -= ink.mean()
    if float(np.abs(ink).mean()) < 1e-3:
        return 0.0

    h, w = ink.shape
    centre = (w / 2.0, h / 2.0)
    best_angle, best_score = 0.0, -np.inf
    for angle in np.arange(-max_angle, max_angle + 0.25, 0.25):
        m = cv2.getRotationMatrix2D(centre, float(angle), 1.0)
        rot = cv2.warpAffine(ink, m, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        score = float(np.var(rot.sum(axis=1)))
        if score > best_score:
            best_angle, best_score = float(angle), score
    return best_angle


def deskew(image: np.ndarray, max_angle: float = 8.0,
           border: int = 255) -> tuple[np.ndarray, float]:
    angle = estimate_skew(image, max_angle)
    if abs(angle) < 0.15:
        return np.asarray(image), 0.0
    img = ensure_rgb(np.asarray(image))
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    out = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_CONSTANT,
                         borderValue=(border, border, border))
    return out, angle


def auto_crop(image: np.ndarray, threshold: int = 238, pad: int = 6) -> np.ndarray:
    img = ensure_rgb(np.asarray(image))
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = (g < threshold).astype(np.uint8)
    if mask.sum() < 50:
        return img
    ys, xs = np.where(mask)
    y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, img.shape[0])
    x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, img.shape[1])
    if (y1 - y0) * (x1 - x0) < 0.5 * img.shape[0] * img.shape[1]:
        return img
    return np.ascontiguousarray(img[y0:y1, x0:x1])


def add_margin(image: np.ndarray, fraction: float = 0.02,
               colour: int = 255) -> np.ndarray:
    img = ensure_rgb(np.asarray(image))
    m = int(round(fraction * max(img.shape[:2])))
    if m <= 0:
        return img
    return cv2.copyMakeBorder(img, m, m, m, m, cv2.BORDER_CONSTANT,
                              value=(colour, colour, colour))
