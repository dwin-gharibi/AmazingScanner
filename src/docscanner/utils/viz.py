from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import cv2
import numpy as np

__all__ = [
    "draw_quad", "draw_corners", "label_bar", "hstack", "vstack", "grid",
    "side_by_side", "wipe", "checkerboard_diff", "heatmap_overlay", "save_gif",
    "difference_map", "annotate",
]

PALETTE = {
    "gt": (46, 204, 113),
    "pred": (231, 76, 60),
    "alt": (52, 152, 219),
    "accent": (241, 196, 15),
    "ink": (33, 37, 41),
    "paper": (248, 249, 250),
}
CORNER_NAMES = ("TL", "TR", "BR", "BL")


def _as_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0 if img.max() <= 1.5 else img, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img[..., :3])


def draw_quad(
    img: np.ndarray,
    quad: np.ndarray,
    color: tuple[int, int, int] = PALETTE["gt"],
    thickness: int | None = None,
    label_corners: bool = True,
    radius: int | None = None,
) -> np.ndarray:
    out = _as_rgb(img).copy()
    h, w = out.shape[:2]
    scale = max(h, w) / 900.0
    thickness = thickness or max(2, int(round(3 * scale)))
    radius = radius or max(3, int(round(6 * scale)))
    pts = np.asarray(quad, np.float32).reshape(-1, 2)

    ipts = np.round(pts).astype(np.int32)
    cv2.polylines(out, [ipts], isClosed=True, color=color, thickness=thickness,
                  lineType=cv2.LINE_AA)
    for i, p in enumerate(ipts):
        cv2.circle(out, tuple(p), radius, color, -1, cv2.LINE_AA)
        cv2.circle(out, tuple(p), radius, (255, 255, 255), max(1, thickness // 2), cv2.LINE_AA)
        if label_corners:
            cv2.putText(out, CORNER_NAMES[i % 4], (p[0] + radius + 2, p[1] - radius - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55 * max(scale, 0.6), (255, 255, 255),
                        max(2, thickness), cv2.LINE_AA)
            cv2.putText(out, CORNER_NAMES[i % 4], (p[0] + radius + 2, p[1] - radius - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55 * max(scale, 0.6), color,
                        max(1, thickness // 2), cv2.LINE_AA)
    return out


def draw_corners(
    img: np.ndarray,
    gt: np.ndarray | None = None,
    pred: np.ndarray | None = None,
    pred2: np.ndarray | None = None,
    legend: Sequence[str] | None = ("ground truth", "prediction", "alternative"),
) -> np.ndarray:
    out = _as_rgb(img).copy()
    names = legend or ("", "", "")
    entries = []
    if gt is not None:
        out = draw_quad(out, gt, PALETTE["gt"])
        entries.append((names[0], PALETTE["gt"]))
    if pred is not None:
        out = draw_quad(out, pred, PALETTE["pred"], label_corners=False)
        entries.append((names[1], PALETTE["pred"]))
    if pred2 is not None:
        out = draw_quad(out, pred2, PALETTE["alt"], label_corners=False)
        entries.append((names[2], PALETTE["alt"]))
    if legend is not None and len(entries) > 1:
        out = _legend(out, entries)
    return out


def _legend(img: np.ndarray, entries: Sequence[tuple[str, tuple[int, int, int]]]) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    scale = max(0.45, min(w / 900.0, 1.2))
    pad = int(10 * scale)
    line = int(26 * scale)
    box_h = pad * 2 + line * len(entries)
    box_w = int(max(len(t) for t, _ in entries) * 12 * scale) + int(50 * scale)
    overlay = out.copy()
    cv2.rectangle(overlay, (pad, pad), (pad + box_w, pad + box_h), (20, 20, 20), -1)
    out = cv2.addWeighted(overlay, 0.55, out, 0.45, 0)
    for i, (text, color) in enumerate(entries):
        y = pad * 2 + line * i + int(line * 0.4)
        cv2.rectangle(out, (pad * 2, y - int(7 * scale)),
                      (pad * 2 + int(20 * scale), y + int(7 * scale)), color, -1)
        cv2.putText(out, text, (pad * 3 + int(20 * scale), y + int(6 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (255, 255, 255),
                    max(1, int(1.4 * scale)), cv2.LINE_AA)
    return out


def label_bar(img: np.ndarray, text: str, height: int | None = None,
              bg: tuple[int, int, int] = (28, 30, 34),
              fg: tuple[int, int, int] = (245, 246, 248)) -> np.ndarray:
    out = _as_rgb(img)
    w = out.shape[1]
    scale = max(0.45, min(w / 640.0, 1.1))
    height = height or int(34 * scale)
    bar = np.full((height, w, 3), bg, np.uint8)
    cv2.putText(bar, text, (int(10 * scale), int(height * 0.68)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, fg,
                max(1, int(1.4 * scale)), cv2.LINE_AA)
    return np.vstack([bar, out])


def annotate(img: np.ndarray, text: str, org: tuple[int, int] = (12, 28),
             color: tuple[int, int, int] = (255, 255, 255), scale: float = 0.6) -> np.ndarray:
    out = _as_rgb(img).copy()
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    return out


def _pad_to(img: np.ndarray, h: int, w: int, value: int = 245) -> np.ndarray:
    out = np.full((h, w, 3), value, np.uint8)
    ih, iw = img.shape[:2]
    y, x = (h - ih) // 2, (w - iw) // 2
    out[y:y + ih, x:x + iw] = img
    return out


def hstack(images: Sequence[np.ndarray], gap: int = 8, bg: int = 245,
           match: str = "height") -> np.ndarray:
    imgs = [_as_rgb(i) for i in images]
    if match == "height":
        h = max(i.shape[0] for i in imgs)
        imgs = [cv2.resize(i, (max(1, int(i.shape[1] * h / i.shape[0])), h),
                           interpolation=cv2.INTER_AREA) if i.shape[0] != h else i
                for i in imgs]
    h = max(i.shape[0] for i in imgs)
    total = sum(i.shape[1] for i in imgs) + gap * (len(imgs) - 1)
    out = np.full((h, total, 3), bg, np.uint8)
    x = 0
    for i in imgs:
        out[:i.shape[0], x:x + i.shape[1]] = i
        x += i.shape[1] + gap
    return out


def vstack(images: Sequence[np.ndarray], gap: int = 8, bg: int = 245) -> np.ndarray:
    imgs = [_as_rgb(i) for i in images]
    w = max(i.shape[1] for i in imgs)
    total = sum(i.shape[0] for i in imgs) + gap * (len(imgs) - 1)
    out = np.full((total, w, 3), bg, np.uint8)
    y = 0
    for i in imgs:
        out[y:y + i.shape[0], :i.shape[1]] = i
        y += i.shape[0] + gap
    return out


def grid(images: Sequence[np.ndarray], cols: int = 4, cell: int = 256,
         gap: int = 6, bg: int = 245) -> np.ndarray:
    imgs = []
    for im in images:
        im = _as_rgb(im)
        h, w = im.shape[:2]
        s = cell / max(h, w)
        im = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))),
                        interpolation=cv2.INTER_AREA)
        imgs.append(_pad_to(im, cell, cell, bg))
    rows = []
    for i in range(0, len(imgs), cols):
        chunk = imgs[i:i + cols]
        while len(chunk) < cols:
            chunk.append(np.full((cell, cell, 3), bg, np.uint8))
        rows.append(hstack(chunk, gap=gap, bg=bg))
    return vstack(rows, gap=gap, bg=bg)


def side_by_side(images: Sequence[np.ndarray], titles: Sequence[str],
                 gap: int = 10, cell: int | None = None) -> np.ndarray:
    panels = []
    for img, title in zip(images, titles):
        im = _as_rgb(img)
        if cell:
            h, w = im.shape[:2]
            s = cell / max(h, w)
            im = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))),
                            interpolation=cv2.INTER_AREA)
        panels.append(label_bar(im, title))
    return hstack(panels, gap=gap)


def wipe(before: np.ndarray, after: np.ndarray, fraction: float = 0.5,
         line_color: tuple[int, int, int] = (241, 196, 15)) -> np.ndarray:
    a, b = _as_rgb(before), _as_rgb(after)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    out = a.copy()
    x = int(np.clip(fraction, 0.0, 1.0) * a.shape[1])
    out[:, x:] = b[:, x:]
    if 0 < x < a.shape[1]:
        cv2.line(out, (x, 0), (x, a.shape[0]), line_color, max(1, a.shape[1] // 400 + 1))
    return out


def difference_map(a: np.ndarray, b: np.ndarray, gain: float = 4.0) -> np.ndarray:
    x, y = _as_rgb(a).astype(np.float32), _as_rgb(b).astype(np.float32)
    if x.shape != y.shape:
        y = cv2.resize(y, (x.shape[1], x.shape[0]), interpolation=cv2.INTER_AREA)
    d = np.abs(x - y).mean(axis=2) * gain
    d = np.clip(d, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(d, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


def checkerboard_diff(a: np.ndarray, b: np.ndarray, tile: int = 32) -> np.ndarray:
    x, y = _as_rgb(a), _as_rgb(b)
    if x.shape != y.shape:
        y = cv2.resize(y, (x.shape[1], x.shape[0]), interpolation=cv2.INTER_AREA)
    h, w = x.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    mask = (((yy // tile) + (xx // tile)) % 2).astype(bool)
    out = x.copy()
    out[mask] = y[mask]
    return out


def heatmap_overlay(img: np.ndarray, heat: np.ndarray, alpha: float = 0.55,
                    colormap: int = cv2.COLORMAP_TURBO) -> np.ndarray:
    base = _as_rgb(img)
    h = np.asarray(heat, np.float32)
    if h.ndim == 3:
        h = h.max(axis=0)
    h = h - h.min()
    h = h / max(float(h.max()), 1e-6)
    h = cv2.resize(h, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_LINEAR)
    cm = cv2.cvtColor(cv2.applyColorMap((h * 255).astype(np.uint8), colormap),
                      cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(base, 1.0 - alpha, cm, alpha, 0)


def save_gif(frames: Iterable[np.ndarray], path: str | Path, fps: float = 4.0,
             loop: int = 0, max_width: int | None = 900, colors: int = 256) -> Path:
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: list[np.ndarray] = []
    for f in frames:
        arr = _as_rgb(f)
        if max_width and arr.shape[1] > max_width:
            s = max_width / arr.shape[1]
            arr = cv2.resize(arr, (max_width, max(1, int(arr.shape[0] * s))),
                             interpolation=cv2.INTER_AREA)
        arrays.append(arr)
    if not arrays:
        raise ValueError("save_gif: no frames")

    ch = max(a.shape[0] for a in arrays)
    cw = max(a.shape[1] for a in arrays)
    rgb: list[Image.Image] = []
    for arr in arrays:
        if arr.shape[:2] != (ch, cw):
            canvas = np.zeros((ch, cw, 3), np.uint8)
            y0, x0 = (ch - arr.shape[0]) // 2, (cw - arr.shape[1]) // 2
            canvas[y0:y0 + arr.shape[0], x0:x0 + arr.shape[1]] = arr
            arr = canvas
        rgb.append(Image.fromarray(arr))

    n_sample = min(8, len(rgb))
    idx = [round(i * (len(rgb) - 1) / max(1, n_sample - 1)) for i in range(n_sample)]
    thumbs = [np.asarray(rgb[i].resize((160, max(1, 160 * ch // max(cw, 1)))))
              for i in dict.fromkeys(idx)]
    mosaic = Image.fromarray(np.concatenate(thumbs, axis=0))
    palette_source = mosaic.quantize(colors=max(2, min(256, colors)),
                                     method=Image.MEDIANCUT)
    quantised = [im.quantize(palette=palette_source, dither=Image.FLOYDSTEINBERG)
                 for im in rgb]
    quantised[0].save(path, save_all=True, append_images=quantised[1:],
                      duration=int(1000 / max(fps, 0.1)), loop=loop, optimize=True)
    return path
