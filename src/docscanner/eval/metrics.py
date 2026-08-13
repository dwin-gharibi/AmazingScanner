from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..models.losses import SSIM
from ..utils.geometry import corner_distance, quad_iou

__all__ = [
    "psnr", "ssim", "psnr_torch", "ssim_torch", "MetricAccumulator",
    "corner_metrics", "CornerScores", "summarize_corner_errors",
]

_SSIM_CACHE: dict[tuple, SSIM] = {}


def _ssim_module(channels: int) -> SSIM:
    key = (channels,)
    if key not in _SSIM_CACHE:
        _SSIM_CACHE[key] = SSIM(channels=channels)
    return _SSIM_CACHE[key]


def _to_tensor(img: np.ndarray) -> torch.Tensor:
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))[None]


def psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    a, b = _to_tensor(pred), _to_tensor(target)
    return psnr_torch(a, b, data_range)


def psnr_torch(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    mse = torch.mean((pred.clamp(0, 1) - target.clamp(0, 1)) ** 2).item()
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10((data_range ** 2) / mse))


def ssim(pred: np.ndarray, target: np.ndarray) -> float:
    a, b = _to_tensor(pred), _to_tensor(target)
    return ssim_torch(a, b)


def ssim_torch(pred: torch.Tensor, target: torch.Tensor) -> float:
    module = _ssim_module(pred.shape[1])
    with torch.no_grad():
        return float(module(pred.clamp(0, 1), target.clamp(0, 1)).item())


@dataclass
class MetricAccumulator:
    sums: dict[str, float] = field(default_factory=dict)
    count: int = 0

    def update(self, **values: float) -> None:
        for k, v in values.items():
            self.sums[k] = self.sums.get(k, 0.0) + float(v)
        self.count += 1

    def mean(self) -> dict[str, float]:
        if self.count == 0:
            return {k: float("nan") for k in self.sums}
        return {k: v / self.count for k, v in self.sums.items()}

    def __repr__(self) -> str:
        m = self.mean()
        return " | ".join(f"{k}={v:.4f}" for k, v in m.items())


@dataclass
class CornerScores:
    n: int
    mce_px: float
    median_px: float
    mce_norm: float
    success: dict[float, float]
    iou: float
    worst_px: float
    per_image: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))

    def as_row(self) -> dict[str, float]:
        row = {
            "n": self.n,
            "MCE (px)": self.mce_px,
            "median (px)": self.median_px,
            "MCE (% diag)": self.mce_norm * 100.0,
            "quad IoU": self.iou,
            "worst (px)": self.worst_px,
        }
        for thr, frac in sorted(self.success.items()):
            row[f"success@{thr:g}px"] = frac * 100.0
        return row


def match_cyclic(preds: np.ndarray, gts: np.ndarray) -> np.ndarray:
    preds = np.asarray(preds, np.float64).reshape(-1, 4, 2)
    gts = np.asarray(gts, np.float64).reshape(-1, 4, 2)
    out = np.empty_like(gts)
    for i, (p, g) in enumerate(zip(preds, gts)):
        best, best_err = g, np.inf
        for k in range(4):
            cand = np.roll(g, -k, axis=0)
            err = np.linalg.norm(p - cand, axis=1).mean()
            if err < best_err:
                best, best_err = cand, err
        out[i] = best
    return out


def corner_metrics(
    preds: np.ndarray,
    gts: np.ndarray,
    image_sizes: np.ndarray | tuple[int, int],
    thresholds: tuple[float, ...] = (4.0, 8.0, 16.0, 32.0),
    cyclic: bool = True,
) -> CornerScores:
    preds = np.asarray(preds, np.float64).reshape(-1, 4, 2)
    gts = np.asarray(gts, np.float64).reshape(-1, 4, 2)
    if cyclic:
        gts = match_cyclic(preds, gts)
    n = len(preds)
    if isinstance(image_sizes, tuple):
        sizes = np.tile(np.asarray(image_sizes, np.float64), (n, 1))
    else:
        sizes = np.asarray(image_sizes, np.float64).reshape(-1, 2)

    per_corner = np.linalg.norm(preds - gts, axis=2)
    per_image = per_corner.mean(axis=1)
    diag = np.linalg.norm(sizes, axis=1)

    success = {}
    for thr in thresholds:
        success[thr] = float((per_corner.max(axis=1) <= thr).mean()) if n else float("nan")

    ious = np.array([quad_iou(p, g) for p, g in zip(preds, gts)]) if n else np.zeros(0)

    return CornerScores(
        n=n,
        mce_px=float(per_image.mean()) if n else float("nan"),
        median_px=float(np.median(per_image)) if n else float("nan"),
        mce_norm=float((per_image / np.maximum(diag, 1e-6)).mean()) if n else float("nan"),
        success=success,
        iou=float(ious.mean()) if n else float("nan"),
        worst_px=float(per_image.max()) if n else float("nan"),
        per_image=per_image,
    )


def summarize_corner_errors(preds: np.ndarray, gts: np.ndarray) -> np.ndarray:
    preds = np.asarray(preds, np.float64).reshape(-1, 4, 2)
    gts = np.asarray(gts, np.float64).reshape(-1, 4, 2)
    return np.array([corner_distance(p, g) for p, g in zip(preds, gts)]).mean(axis=0)
