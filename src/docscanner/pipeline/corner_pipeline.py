from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import torch

from ..data.datasets import DOC_MEAN, DOC_STD
from ..data.real import quad_from_mask
from ..models.corner_nets import CornerHeatmapNet, CornerRegressor, soft_argmax_2d
from ..utils.geometry import order_corners, quad_area, quad_is_convex, quad_is_plausible
from ..utils.imageio import ensure_rgb

Approach = Literal["regression", "heatmap", "auto"]

__all__ = ["CornerPipeline", "CornerResult", "refine_quad_with_edges",
           "detect_quad_classical", "full_frame_quad", "rot90_points_inverse",
           "heatmap_peaks", "decode_quad_masked", "DEFAULT_MIN_MASK_IOU"]

DEFAULT_MIN_MASK_IOU = 0.80

def heatmap_peaks(hm: np.ndarray, k: int = 4, nms: int = 5,
                  thresh: float = 0.05) -> list[tuple[float, float, float]]:
    pad_ = nms
    mx = cv2.dilate(hm, np.ones((pad_ * 2 + 1, pad_ * 2 + 1), np.uint8))
    local = (hm >= mx - 1e-9) & (hm > thresh)
    ys, xs = np.nonzero(local)
    if ys.size == 0:
        iy, ix = np.unravel_index(int(np.argmax(hm)), hm.shape)
        ys, xs = np.array([iy]), np.array([ix])
    order = np.argsort(hm[ys, xs])[::-1][:k]
    return [(float(hm[int(ys[i]), int(xs[i])]),
             *_subpixel(hm, int(ys[i]), int(xs[i])))
            for i in order]


def _subpixel(hm: np.ndarray, py: int, px: int, window: int = 7,
              temperature: float = 0.02) -> tuple[float, float]:
    h, w = hm.shape
    r = window // 2
    y0, y1 = max(0, py - r), min(h, py + r + 1)
    x0, x1 = max(0, px - r), min(w, px + r + 1)
    patch = hm[y0:y1, x0:x1].astype(np.float64)
    e = np.exp((patch - patch.max()) / max(temperature, 1e-6))
    e /= e.sum() + 1e-12
    gy, gx = np.mgrid[y0:y1, x0:x1]
    return float((e * gx).sum()), float((e * gy).sum())


def quad_from_seg(seg: np.ndarray, pad: float = 0.12) -> np.ndarray | None:
    from ..data.real import quad_from_mask

    gh, gw = seg.shape[:2]
    grid = quad_from_mask((np.clip(seg, 0, 1) * 255).astype(np.uint8))
    if grid is None:
        return None
    span = 1.0 + 2 * pad
    return np.stack([grid[:, 0] / max(gw - 1, 1) * span - pad,
                     grid[:, 1] / max(gh - 1, 1) * span - pad], axis=1).astype(np.float32)


def decode_quad_masked(heat: np.ndarray, seg: np.ndarray, pad: float = 0.12,
                       topk: int = 4, w_mask: float = 3.0,
                       min_iou: float = 0.72) -> np.ndarray:
    cand = [heatmap_peaks(heat[c], k=max(1, topk)) for c in range(4)]
    gh, gw = seg.shape[:2]
    span = 1.0 + 2 * pad
    mask_bin = (seg > 0.5).astype(np.uint8)
    mask_area = int(mask_bin.sum())

    best, best_score, best_iou = None, -np.inf, 0.0
    for combo in np.ndindex(*[len(c) for c in cand]):
        pts_grid = np.array([[cand[c][combo[c]][1], cand[c][combo[c]][2]]
                             for c in range(4)], np.float32)
        pts = np.stack([pts_grid[:, 0] / max(gw - 1, 1) * span - pad,
                        pts_grid[:, 1] / max(gh - 1, 1) * span - pad], axis=1)
        if not quad_is_convex(pts):
            continue
        poly = np.round(pts_grid).astype(np.int32)
        buf = np.zeros((gh, gw), np.uint8)
        cv2.fillConvexPoly(buf, poly, 1)
        inter = int((buf & mask_bin).sum())
        union = int(buf.sum()) + mask_area - inter
        iou = inter / union if union else 0.0
        score = float(np.mean([np.log(max(cand[c][combo[c]][0], 1e-4))
                               for c in range(4)])) + w_mask * iou
        if score > best_score:
            best_score, best, best_iou = score, pts.astype(np.float32), iou

    if best is None or best_iou < min_iou:
        from_seg = quad_from_seg(seg, pad=pad)
        if from_seg is not None:
            return _snap_to_peaks(from_seg, cand, gh, gw, pad)
    if best is None:
        t = torch.from_numpy(heat)[None]
        return soft_argmax_2d(t, pad=pad)[0].numpy()
    return best


def _snap_to_peaks(quad: np.ndarray, cand: list, gh: int, gw: int, pad: float,
                   radius: float = 0.10, min_score: float = 0.35) -> np.ndarray:
    span = 1.0 + 2 * pad
    out = quad.copy()
    for c in range(4):
        best_d, best_p = radius, None
        for score, gx, gy in cand[c]:
            if score < min_score:
                continue
            px = gx / max(gw - 1, 1) * span - pad
            py = gy / max(gh - 1, 1) * span - pad
            d = float(np.hypot(px - quad[c, 0], py - quad[c, 1]))
            if d < best_d:
                best_d, best_p = d, (px, py)
        if best_p is not None:
            out[c] = best_p
    return out.astype(np.float32)


@dataclass
class CornerResult:
    corners: np.ndarray
    corners_raw: np.ndarray
    corners_prerefine: np.ndarray | None = None
    confidence: float = 0.0
    refined: bool = False
    heatmaps: np.ndarray | None = None
    seconds: float = 0.0
    approach: str = ""
    source: str = "network"
    meta: dict[str, Any] = field(default_factory=dict)


def _fit_edge_line(gray: np.ndarray, p0: np.ndarray, p1: np.ndarray,
                   band: float, samples: int = 48) -> tuple[float, float, float] | None:
    h, w = gray.shape[:2]
    d = p1 - p0
    length = float(np.linalg.norm(d))
    if length < 8:
        return None
    t_hat = d / length
    n_hat = np.array([-t_hat[1], t_hat[0]], np.float32)

    band = max(3.0, float(band))
    offsets = np.arange(-band, band + 1, 1.0, dtype=np.float32)
    pts = []
    for s in np.linspace(0.12, 0.88, samples, dtype=np.float32):
        base = p0 + d * s
        coords = base[None, :] + offsets[:, None] * n_hat[None, :]
        xs = np.clip(coords[:, 0], 0, w - 1).astype(np.float32)
        ys = np.clip(coords[:, 1], 0, h - 1).astype(np.float32)
        vals = cv2.remap(gray, xs.reshape(-1, 1), ys.reshape(-1, 1),
                         cv2.INTER_LINEAR).ravel()
        if vals.size < 3:
            continue
        k = int(np.argmax(vals))
        if vals[k] < 1e-3:
            continue
        if 0 < k < len(vals) - 1:
            a0, b0, c0 = float(vals[k - 1]), float(vals[k]), float(vals[k + 1])
            denom = (a0 - 2 * b0 + c0)
            delta = 0.5 * (a0 - c0) / denom if abs(denom) > 1e-6 else 0.0
            delta = float(np.clip(delta, -1.0, 1.0))
        else:
            delta = 0.0
        pts.append(base + (offsets[k] + delta) * n_hat)

    if len(pts) < 12:
        return None
    arr = np.asarray(pts, np.float32)

    for _ in range(2):
        vx, vy, x0, y0 = cv2.fitLine(arr, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        a, b = -vy, vx
        c = -(a * x0 + b * y0)
        dist = np.abs(a * arr[:, 0] + b * arr[:, 1] + c)
        keep = dist <= max(1.5, float(np.median(dist)) * 2.5)
        if keep.sum() < 10:
            break
        arr = arr[keep]
    vx, vy, x0, y0 = cv2.fitLine(arr, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    a, b = -vy, vx
    return float(a), float(b), float(-(a * x0 + b * y0))


def _intersect(l1, l2) -> np.ndarray | None:
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-8:
        return None
    x = (b1 * c2 - b2 * c1) / det
    y = (c1 * a2 - c2 * a1) / det
    return np.array([x, y], np.float32)


def refine_quad_with_edges(image: np.ndarray, quad: np.ndarray,
                           band_frac: float = 0.02,
                           max_shift_frac: float = 0.04) -> tuple[np.ndarray, bool]:
    img = ensure_rgb(np.asarray(image))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), 1.2)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = mag / max(float(mag.max()), 1e-6)

    q = np.asarray(quad, np.float32).reshape(4, 2)
    diag = float(np.hypot(*img.shape[:2]))
    band = band_frac * diag

    lines = []
    for i in range(4):
        line = _fit_edge_line(mag, q[i], q[(i + 1) % 4], band)
        if line is None:
            return q, False
        lines.append(line)

    out = np.zeros_like(q)
    for i in range(4):
        p = _intersect(lines[(i - 1) % 4], lines[i])
        if p is None:
            return q, False
        out[i] = p

    if not quad_is_convex(out):
        return q, False
    if float(np.abs(out - q).max()) > max_shift_frac * diag:
        return q, False
    if quad_area(out) < 0.4 * quad_area(q):
        return q, False
    return out.astype(np.float32), True


def rot90_points_inverse(pts: np.ndarray, k: int,
                         rot_shape: tuple[int, int]) -> np.ndarray:
    p = np.asarray(pts, np.float32).reshape(-1, 2).copy()
    h, w = int(rot_shape[0]), int(rot_shape[1])
    for _ in range(int(k) % 4):
        p = np.stack([(h - 1) - p[:, 1], p[:, 0]], axis=1)
        h, w = w, h
    return p.astype(np.float32)


def full_frame_quad(shape: tuple[int, int]) -> np.ndarray:
    h, w = int(shape[0]), int(shape[1])
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)


def detect_quad_classical(image: np.ndarray, work: int = 384) -> np.ndarray | None:
    img = ensure_rgb(np.asarray(image))
    h, w = img.shape[:2]
    if min(h, w) < 16:
        return None
    scale = min(1.0, work / float(max(h, w)))
    small = (cv2.resize(img, (max(8, int(round(w * scale))), max(8, int(round(h * scale)))),
                        interpolation=cv2.INTER_AREA) if scale < 1.0 else img)
    sh, sw = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

    candidates: list[np.ndarray] = []

    smooth = cv2.bilateralFilter(gray, 7, 45, 45)
    med = float(np.median(smooth))
    lo = int(max(0, 0.66 * med))
    hi = int(min(255, 1.33 * med))
    edges = cv2.Canny(smooth, lo, max(lo + 1, hi))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, k, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=2)
    filled = edges.copy()
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        solid = np.zeros((sh, sw), np.uint8)
        cv2.drawContours(solid, [max(contours, key=cv2.contourArea)], -1, 255, -1)
        q = quad_from_mask(solid)
        if q is not None:
            candidates.append(q)

    blur = cv2.GaussianBlur(gray, (0, 0), 1.5)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, k, iterations=2)
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, k, iterations=1)
    q = quad_from_mask(otsu)
    if q is not None:
        candidates.append(q)

    frame_area = float(sw * sh)
    candidates = [c for c in candidates
                  if quad_is_plausible(c, (sh, sw)) and quad_area(c) < 0.98 * frame_area]
    if not candidates:
        return None
    best = max(candidates, key=quad_area)
    if scale < 1.0:
        best = best / np.array([sw / float(w), sh / float(h)], np.float32)
    return order_corners(best.astype(np.float32))


class CornerPipeline:
    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "cpu",
        size: int | None = None,
        threads: int | None = None,
        prefer_ema: bool = True,
        refine: bool = False,
        tta: bool = True,
        luma_normalise: float = 1.0,
        mask_topk: int = 4,
        mask_weight: float = 3.0,
        min_mask_iou: float | None = DEFAULT_MIN_MASK_IOU,
    ):
        if threads:
            torch.set_num_threads(threads)
        self.device = torch.device(device)
        self.refine = refine
        self.tta = tta
        self.luma_normalise = float(luma_normalise)
        self.mask_topk = int(mask_topk)
        self.mask_weight = float(mask_weight)
        self.model, self.approach, self.size, self.meta = self._load(
            checkpoint, prefer_ema, size)
        iou = self.meta.get("val", {}).get("val_mask_iou")
        self.mask_iou = float(iou) if iou is not None else None
        self.use_mask = bool(getattr(self.model, "has_seg", False))
        if self.use_mask and min_mask_iou is not None:
            if self.mask_iou is None:
                warnings.warn(
                    f"{Path(checkpoint).name} has a page-mask head but no recorded "
                    "val_mask_iou, so its quality cannot be checked; decoding "
                    "without it. Retrain with the current trainer, or pass "
                    "min_mask_iou=None to use it unchecked.", RuntimeWarning,
                    stacklevel=2)
                self.use_mask = False
            elif self.mask_iou < float(min_mask_iou):
                warnings.warn(
                    f"{Path(checkpoint).name} page-mask IoU {self.mask_iou:.3f} is "
                    f"below {float(min_mask_iou):.2f}; decoding without it. A mask "
                    "this weak measured worse than using none at all.",
                    RuntimeWarning, stacklevel=2)
                self.use_mask = False
        self.model = self.model.to(self.device).eval()
        self.model = self.model.to(memory_format=torch.channels_last)

    @staticmethod
    def _load(path, prefer_ema: bool, size_override: int | None):
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        extra = ckpt.get("config", {}).get("extra", {})
        approach = extra.get("approach", "heatmap")
        size = size_override or extra.get("size", 256)
        state = ckpt.get("ema") if (prefer_ema and ckpt.get("ema")) else ckpt["model"]
        if approach == "regression":
            model = CornerRegressor(base=extra.get("base", 24),
                                    depth=extra.get("depth", 5),
                                    head_dim=extra.get("head_dim", 256),
                                    dropout=0.0, input_size=size)
        else:
            has_seg = any(k.startswith("seg.") for k in state)
            model = CornerHeatmapNet(base=extra.get("base", 24),
                                     depth=extra.get("depth", 4),
                                     out_stride=extra.get("out_stride", 4),
                                     heatmap_pad=extra.get("pad", 0.12),
                                     dropout=0.0,
                                     seg_head=has_seg)
        model.load_state_dict(state)
        meta = {"path": str(path), "config": ckpt.get("config", {}),
                "val": ckpt.get("extra", {})}
        return model, approach, size, meta

    def _luma_normalise(self, img: np.ndarray) -> np.ndarray:
        s = self.luma_normalise
        if s <= 0:
            return img
        a = img.astype(np.float32) / 255.0
        g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        m, sd = float(g.mean()), float(g.std())
        if sd < 1e-4:
            return img
        target_mean = float(np.mean(DOC_MEAN))
        target_std = float(np.mean(DOC_STD))
        gain = 1.0 + s * ((target_std / sd) - 1.0)
        bias = (target_mean - m * gain) * s
        return (np.clip(a * gain + bias, 0.0, 1.0) * 255.0).astype(np.uint8)

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        img = cv2.resize(ensure_rgb(image), (self.size, self.size),
                         interpolation=cv2.INTER_AREA)
        img = self._luma_normalise(img)
        arr = img.astype(np.float32) / 255.0
        t = torch.from_numpy(arr.transpose(2, 0, 1))[None]
        mean = torch.tensor(DOC_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(DOC_STD).view(1, 3, 1, 1)
        return ((t - mean) / std).contiguous(memory_format=torch.channels_last)

    def _forward(self, img: np.ndarray) -> tuple[np.ndarray, float, np.ndarray | None]:
        h, w = img.shape[:2]
        x = self._preprocess(img).to(self.device)
        heatmaps = None
        if self.approach == "heatmap":
            seg = None
            if self.use_mask:
                logits, seg_logits = self.model.forward_with_seg(x)
                hm = torch.sigmoid(logits)
                seg = torch.sigmoid(seg_logits)[0, 0].cpu().numpy()
            else:
                hm = torch.sigmoid(self.model(x))
            heat = hm[0].cpu().numpy()
            if seg is not None:
                coords = decode_quad_masked(heat, seg, pad=self.model.heatmap_pad,
                                            topk=self.mask_topk,
                                            w_mask=self.mask_weight)
            else:
                coords = soft_argmax_2d(hm, pad=self.model.heatmap_pad)[0].cpu().numpy()
            heatmaps = heat
            confidence = float(hm.amax(dim=(2, 3)).mean())
        else:
            coords = self.model(x)[0].cpu().numpy()
            confidence = 1.0
        pts = np.stack([coords[:, 0] * w, coords[:, 1] * h], axis=1).astype(np.float32)
        return pts, confidence, heatmaps

    def _forward_tta(self, img: np.ndarray) -> tuple[np.ndarray, float, np.ndarray | None]:
        quads, confs, heatmaps = [], [], None
        for k in range(4):
            rot = np.ascontiguousarray(np.rot90(img, k))
            pts, conf, hm = self._forward(rot)
            if k == 0:
                heatmaps = hm
            quads.append(order_corners(rot90_points_inverse(pts, k, rot.shape[:2])))
            confs.append(conf)
        median = np.median(np.stack(quads), axis=0).astype(np.float32)
        return median, float(np.mean(confs)), heatmaps

    @torch.no_grad()
    def __call__(self, image: np.ndarray, refine: bool | None = None,
                 tta: bool | None = None) -> CornerResult:
        t0 = time.time()
        img = ensure_rgb(np.asarray(image))
        h, w = img.shape[:2]

        use_tta = self.tta if tta is None else tta
        if use_tta:
            pts, confidence, heatmaps = self._forward_tta(img)
        else:
            pts, confidence, heatmaps = self._forward(img)
        pts_raw = pts.copy()

        source = "network"
        if not quad_is_plausible(pts, (h, w)):
            classical = detect_quad_classical(img)
            if classical is not None:
                pts, source, confidence = classical, "classical", min(confidence, 0.35)
            else:
                pts, source, confidence = full_frame_quad((h, w)), "full_frame", 0.0
        else:
            confidence = min(1.0, confidence)

        did_refine = False
        pre_refine = pts.copy()
        use_refine = self.refine if refine is None else refine
        if use_refine and source != "full_frame":
            pts, did_refine = refine_quad_with_edges(img, pts)

        return CornerResult(
            corners=order_corners(pts),
            corners_raw=order_corners(pts_raw),
            corners_prerefine=order_corners(pre_refine),
            confidence=float(confidence),
            refined=did_refine,
            heatmaps=heatmaps,
            seconds=time.time() - t0,
            approach=self.approach,
            source=source,
            meta={"input_size": (w, h), "network_size": self.size,
                  "fallback": source != "network", "tta": bool(use_tta)},
        )

    @torch.no_grad()
    def predict_normalized(self, images: torch.Tensor) -> torch.Tensor:
        if self.approach == "heatmap":
            return soft_argmax_2d(torch.sigmoid(self.model(images)),
                                  pad=self.model.heatmap_pad)
        return self.model(images)
