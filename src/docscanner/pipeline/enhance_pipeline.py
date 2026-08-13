from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import torch

from ..models.enhance_unet import DocEnhanceNet
from ..utils.imageio import ensure_rgb, fit_within, pad_to_multiple

OutputMode = Literal["color", "gray", "bw", "whiteboard", "raw"]

__all__ = ["EnhancementPipeline", "EnhanceResult", "apply_output_mode",
           "recover_highlights", "paperness", "preserve_tone", "compress_highlights"]


@dataclass
class EnhanceResult:
    image: np.ndarray
    network_output: np.ndarray
    input_image: np.ndarray
    mode: str = "color"
    seconds: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


def apply_output_mode(img: np.ndarray, mode: OutputMode = "color") -> np.ndarray:
    img = ensure_rgb(img)
    if mode in ("color", "raw"):
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if mode == "gray":
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    if mode == "bw":
        win = max(15, (min(gray.shape[:2]) // 40) | 1)
        mean = cv2.boxFilter(gray.astype(np.float32), -1, (win, win))
        sq = cv2.boxFilter((gray.astype(np.float32)) ** 2, -1, (win, win))
        std = np.sqrt(np.maximum(sq - mean ** 2, 0))
        k, r = 0.20, 128.0
        thresh = mean * (1 + k * (std / r - 1))
        binar = np.where(gray.astype(np.float32) > thresh, 255, 0).astype(np.uint8)
        return cv2.cvtColor(binar, cv2.COLOR_GRAY2RGB)

    if mode == "whiteboard":
        bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
        bg = cv2.GaussianBlur(bg, (0, 0), 9).astype(np.float32) + 1.0
        norm = np.clip(img.astype(np.float32) / bg[..., None] * 255.0 * 0.94, 0, 255)
        out = np.clip((norm - 12.0) * 1.30, 0, 255).astype(np.uint8)
        return out

    raise ValueError(f"unknown output mode: {mode}")


def _feather(h: int, w: int, overlap: int) -> np.ndarray:
    def ramp(n: int) -> np.ndarray:
        v = np.ones(n, np.float32)
        k = min(overlap, n // 2)
        if k > 0:
            t = np.linspace(0, np.pi / 2, k, dtype=np.float32)
            v[:k] = np.sin(t) ** 2
            v[-k:] = np.cos(t) ** 2
        return v
    return ramp(h)[:, None] * ramp(w)[None, :]


def recover_highlights(inp: np.ndarray, out: np.ndarray,
                       knee: float = 0.95, ceiling: float = 0.995) -> np.ndarray:
    out = np.asarray(out, np.float32)
    inp = np.asarray(inp, np.float32)
    if out.shape != inp.shape:
        return np.clip(out, 0.0, 1.0)

    span = max(1.0 - knee, 1e-6)
    t = np.clip((out - knee) / span, 0.0, None)
    rolled = knee + (ceiling - knee) * (1.0 - np.exp(-t))
    out = np.where(out > knee, rolled, out)

    sat = np.clip((out - knee) / span, 0.0, 1.0)
    if float(sat.max()) <= 1e-6:
        return np.clip(out, 0.0, 1.0)
    blur = cv2.GaussianBlur(inp, (0, 0), 2.0)
    detail = inp - blur
    return np.clip(out + sat * detail * 1.6, 0.0, 1.0)


def paperness(img: np.ndarray) -> float:
    img = ensure_rgb(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    sat = float(hsv[..., 1].mean()) / 255.0
    v = hsv[..., 2]
    ref = float(np.percentile(v, 98)) + 1e-6
    bright = (v > 0.82 * ref) & (hsv[..., 1] < 60)
    frac = float(bright.mean())
    s_term = float(np.clip(1.0 - sat / 0.14, 0.0, 1.0))
    f_term = float(np.clip((frac - 0.25) / 0.45, 0.0, 1.0))
    return float(s_term * f_term)


def preserve_tone(inp: np.ndarray, out: np.ndarray, strength: float) -> np.ndarray:
    s = float(np.clip(strength, 0.0, 1.0))
    a = np.clip(inp, 0, 1).astype(np.float32)
    b = np.clip(out, 0, 1).astype(np.float32)
    lab_i = cv2.cvtColor(a, cv2.COLOR_RGB2LAB)
    lab_o = cv2.cvtColor(b, cv2.COLOR_RGB2LAB)

    lab_o[..., 1] = lab_i[..., 1]
    lab_o[..., 2] = lab_i[..., 2]

    if s > 1e-3:
        li, lo = lab_i[..., 0], lab_o[..., 0]
        mo, so = float(lo.mean()), float(lo.std()) + 1e-5
        mi, si = float(li.mean()), float(li.std()) + 1e-5
        scale = float(np.clip(si / so, 0.6, 1.8))
        toned = (lo - mo) * scale + mi
        lab_o[..., 0] = lo * (1.0 - s) + toned * s
    return np.clip(cv2.cvtColor(lab_o, cv2.COLOR_LAB2RGB), 0.0, 1.0)


def compress_highlights(img: np.ndarray, inp: np.ndarray | None = None,
                        target: float = 0.985, pct: float = 99.5,
                        min_destroyed: float = 0.01) -> np.ndarray:
    lab = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2LAB)
    l = lab[..., 0] / 100.0
    hi = float(np.percentile(l, pct))
    if hi <= target:
        return img

    if inp is not None:
        blown = l >= 0.98
        if not blown.any():
            return img
        src = cv2.cvtColor(np.clip(inp, 0, 1).astype(np.float32), cv2.COLOR_RGB2GRAY)
        detail = cv2.morphologyEx(src, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        destroyed = float((blown & (detail > 0.06)).mean())
        if destroyed < min_destroyed:
            return img

    knee = target * 0.85
    top = np.maximum(l - knee, 0.0)
    span = max(hi - knee, 1e-5)
    lab[..., 0] = (np.minimum(l, knee) + (target - knee) * np.tanh(top / span)) * 100.0
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0.0, 1.0)


class EnhancementPipeline:
    def __init__(
        self,
        checkpoint: str | Path | None = None,
        model: DocEnhanceNet | None = None,
        device: str = "cpu",
        max_side: int = 1600,
        tile: int = 768,
        overlap: int = 64,
        threads: int | None = None,
        prefer_ema: bool = True,
        recover_highlights: bool = True,
        preserve_tone: float = 0.0,
        compress_highlights: bool = True,
    ):
        if threads:
            torch.set_num_threads(threads)
        self.device = torch.device(device)
        self.max_side = int(max_side)
        self.tile = int(tile)
        self.overlap = int(overlap)
        self.recover_highlights = bool(recover_highlights)
        self.compress_highlights = bool(compress_highlights)
        self.preserve_tone = float(preserve_tone)

        if model is None:
            if checkpoint is None:
                raise ValueError("provide either a checkpoint or a model")
            model, self.ckpt_meta = self._load(checkpoint, prefer_ema)
        else:
            self.ckpt_meta = {}
        self.model = model.to(self.device).eval()
        self.model = self.model.to(memory_format=torch.channels_last)

    @staticmethod
    def _load(path: str | Path, prefer_ema: bool) -> tuple[DocEnhanceNet, dict]:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        extra = ckpt.get("config", {}).get("extra", {})
        model = DocEnhanceNet(
            base=extra.get("base", 32),
            depth=extra.get("depth", 4),
            dropout=0.0,
            use_bg_prior=extra.get("bg_prior", True),
        )
        state = ckpt.get("ema") if (prefer_ema and ckpt.get("ema")) else ckpt["model"]
        model.load_state_dict(state)
        meta = {"path": str(path), "epoch": ckpt.get("extra", {}).get("epoch"),
                "config": ckpt.get("config", {}), "val": ckpt.get("extra", {})}
        return model, meta

    @torch.no_grad()
    def _forward(self, arr: np.ndarray) -> np.ndarray:
        padded, (ph, pw) = pad_to_multiple(arr, 16)
        x = torch.from_numpy(padded.transpose(2, 0, 1))[None].to(self.device)
        x = x.contiguous(memory_format=torch.channels_last)
        y = self.model(x, clamp=True)[0].cpu().numpy().transpose(1, 2, 0)
        h, w = arr.shape[:2]
        return y[:h, :w]

    @torch.no_grad()
    def _forward_tiled(self, arr: np.ndarray) -> np.ndarray:
        h, w = arr.shape[:2]
        if max(h, w) <= self.tile:
            return self._forward(arr)

        step = max(32, self.tile - self.overlap)
        acc = np.zeros((h, w, 3), np.float32)
        wsum = np.zeros((h, w, 1), np.float32)
        ys = list(range(0, max(h - self.overlap, 1), step))
        xs = list(range(0, max(w - self.overlap, 1), step))
        for y in ys:
            for x in xs:
                y1, x1 = min(y + self.tile, h), min(x + self.tile, w)
                y0, x0 = max(0, y1 - self.tile), max(0, x1 - self.tile)
                patch = arr[y0:y1, x0:x1]
                out = self._forward(patch)
                fw = _feather(out.shape[0], out.shape[1], self.overlap)[..., None]
                acc[y0:y1, x0:x1] += out * fw
                wsum[y0:y1, x0:x1] += fw
        return acc / np.maximum(wsum, 1e-6)

    def __call__(self, image: np.ndarray, mode: OutputMode = "color",
                 keep_size: bool = True) -> EnhanceResult:
        t0 = time.time()
        src = ensure_rgb(np.asarray(image))
        if src.dtype != np.uint8:
            src = np.clip(src * 255 if src.max() <= 1.5 else src, 0, 255).astype(np.uint8)
        original_size = (src.shape[1], src.shape[0])

        work = fit_within(src, self.max_side)
        arr = work.astype(np.float32) / 255.0
        out = self._forward_tiled(arr)
        if self.recover_highlights:
            out = recover_highlights(arr, out)

        paper = paperness((np.clip(out, 0, 1) * 255).astype(np.uint8))
        tone_strength = self.preserve_tone * (1.0 - paper)
        if tone_strength > 1e-3:
            out = preserve_tone(arr, out, tone_strength)

        out8 = np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)

        if keep_size and (out8.shape[1], out8.shape[0]) != original_size:
            out8 = cv2.resize(out8, original_size, interpolation=cv2.INTER_CUBIC)

        styled_f = compress_highlights(out, inp=arr) if self.compress_highlights else out
        styled8 = np.clip(styled_f * 255.0 + 0.5, 0, 255).astype(np.uint8)
        if keep_size and (styled8.shape[1], styled8.shape[0]) != original_size:
            styled8 = cv2.resize(styled8, original_size, interpolation=cv2.INTER_CUBIC)
        styled = apply_output_mode(styled8, mode)
        return EnhanceResult(
            image=styled,
            network_output=out8,
            input_image=src,
            mode=mode,
            seconds=time.time() - t0,
            meta={"work_size": (work.shape[1], work.shape[0]),
                  "original_size": original_size,
                  "tiled": max(work.shape[:2]) > self.tile,
                  "paperness": paper,
                  "tone_restored": tone_strength,
                  **self.ckpt_meta.get("val", {})},
        )

    def batch(self, images: list[np.ndarray], mode: OutputMode = "color") -> list[EnhanceResult]:
        return [self(img, mode=mode) for img in images]
