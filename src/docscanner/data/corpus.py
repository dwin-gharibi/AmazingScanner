from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

from ..utils.imageio import ensure_rgb, fit_within, imread_rgb

__all__ = ["Split", "ImageCorpus", "ScanCorpus", "BackgroundCorpus", "procedural_background"]

Split = str
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")


def _stable_hash(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def hash_split(key: str, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)) -> Split:
    x = (_stable_hash(key) % 10_000) / 10_000.0
    if x < ratios[0]:
        return "train"
    if x < ratios[0] + ratios[1]:
        return "val"
    return "test"


class _LRUCache:
    def __init__(self, capacity: int = 64):
        self.capacity = capacity
        self._d: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, key: str):
        if key not in self._d:
            return None
        self._d.move_to_end(key)
        return self._d[key]

    def put(self, key: str, value: np.ndarray) -> None:
        if self.capacity <= 0:
            return
        self._d[key] = value
        self._d.move_to_end(key)
        while len(self._d) > self.capacity:
            self._d.popitem(last=False)


@dataclass
class ImageCorpus:
    paths: list[Path]
    name: str = "corpus"
    max_side: int | None = None
    cache_size: int = 48
    weights: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.paths = [Path(p) for p in self.paths]
        self._cache = _LRUCache(self.cache_size)
        if self.weights is not None:
            w = np.asarray(self.weights, np.float64).ravel()
            if len(w) != len(self.paths):
                raise ValueError("weights must have one entry per path")
            total = w.sum()
            self.weights = w / total if total > 0 else None

    def __len__(self) -> int:
        return len(self.paths)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}: {len(self)} images>"

    def load(self, index: int) -> np.ndarray:
        path = self.paths[int(index) % max(len(self.paths), 1)]
        key = str(path)
        img = self._cache.get(key)
        if img is None:
            img = ensure_rgb(imread_rgb(path, exif=False))
            if self.max_side:
                img = fit_within(img, self.max_side)
            self._cache.put(key, img)
        return img

    def sample_index(self, rng: np.random.Generator) -> int:
        if self.weights is None:
            return int(rng.integers(0, len(self.paths)))
        return int(rng.choice(len(self.paths), p=self.weights))

    def sample(self, rng: np.random.Generator) -> tuple[np.ndarray, int]:
        i = self.sample_index(rng)
        return self.load(i), i

    @classmethod
    def from_dir(cls, root: str | Path, name: str = "corpus", **kw) -> "ImageCorpus":
        root = Path(root)
        paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXT)
        return cls(paths=paths, name=name, **kw)


class ScanCorpus(ImageCorpus):
    ASPECTS: tuple[float, ...] = (
        0.707,
        0.773,
        0.85,
        1.0,
        1.294,
        1.414,
        0.5,
        1.7,
    )

    def load_document(
        self,
        index: int,
        rng: np.random.Generator | None = None,
        aspect: float | None = None,
        jitter: float = 0.25,
    ) -> np.ndarray:
        img = self.load(index)
        if rng is None:
            return img
        if aspect is None:
            aspect = float(rng.choice(self.ASPECTS))
        h, w = img.shape[:2]
        cw, ch = (w, int(round(w / aspect))) if w / aspect <= h else (int(round(h * aspect)), h)
        cw, ch = max(16, min(cw, w)), max(16, min(ch, h))
        max_dx, max_dy = w - cw, h - ch
        dx = int(rng.integers(0, max_dx + 1)) if max_dx > 0 else 0
        dy = int(rng.integers(0, max_dy + 1)) if max_dy > 0 else 0
        if max_dx > 0:
            dx = int(round(dx * jitter + (max_dx / 2) * (1 - jitter)))
        if max_dy > 0:
            dy = int(round(dy * jitter + (max_dy / 2) * (1 - jitter)))
        return np.ascontiguousarray(img[dy:dy + ch, dx:dx + cw])


class BackgroundCorpus(ImageCorpus):
    def canvas(
        self,
        width: int,
        height: int,
        rng: np.random.Generator,
        procedural_prob: float = 0.15,
    ) -> np.ndarray:
        if len(self.paths) == 0 or rng.random() < procedural_prob:
            return procedural_background(width, height, rng)
        img, _ = self.sample(rng)
        h, w = img.shape[:2]

        zoom = float(rng.uniform(1.0, 2.2))
        tw, th = int(width / zoom), int(height / zoom)
        tw, th = max(8, min(tw, w)), max(8, min(th, h))
        x = int(rng.integers(0, max(w - tw, 0) + 1))
        y = int(rng.integers(0, max(h - th, 0) + 1))
        crop = img[y:y + th, x:x + tw]

        k = int(rng.integers(0, 4))
        if k:
            crop = np.rot90(crop, k)
        out = cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)

        gain = float(rng.uniform(0.45, 1.05))
        out = np.clip(out.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(out)


def procedural_background(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    kind = int(rng.integers(0, 3))
    base = rng.uniform(0.18, 0.72)
    tint = rng.uniform(0.75, 1.25, size=3).astype(np.float32)
    small = (max(4, height // 16), max(4, width // 16))
    field = rng.random(small).astype(np.float32)
    field = cv2.resize(field, (width, height), interpolation=cv2.INTER_CUBIC)

    if kind == 0:
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        ang = float(rng.uniform(0, np.pi))
        u = np.cos(ang) * xx + np.sin(ang) * yy
        freq = float(rng.uniform(0.01, 0.06))
        grain = 0.5 + 0.5 * np.sin(u * freq + field * 6.0)
        img = base * (0.75 + 0.45 * grain)
        tint = tint * np.array([1.15, 0.92, 0.72], np.float32)
    elif kind == 1:
        img = base * (0.7 + 0.6 * field)
        speck = (rng.random((height, width)) > 0.997).astype(np.float32)
        img = img + cv2.GaussianBlur(speck, (0, 0), 1.2) * 0.35
    else:
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        p = float(rng.uniform(3.0, 12.0))
        weave = 0.5 + 0.25 * (np.sin(2 * np.pi * xx / p) + np.sin(2 * np.pi * yy / p))
        img = base * (0.8 + 0.35 * weave) * (0.85 + 0.3 * field)

    img = np.clip(img, 0, 1).astype(np.float32)
    rgb = np.clip(img[..., None] * tint[None, None, :], 0, 1)
    noise = rng.normal(0, 0.02, size=rgb.shape).astype(np.float32)
    rgb = np.clip(rgb + noise, 0, 1)
    return (rgb * 255).astype(np.uint8)


def split_paths_by_parent(
    paths: Iterable[Path], eval_fraction: float = 0.25, seed: int = 0
) -> dict[str, list[Path]]:
    paths = list(paths)
    families = sorted({p.parent.name for p in paths})
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(families))
    n_eval = max(1, int(round(len(families) * eval_fraction)))
    eval_families = {families[i] for i in order[:n_eval]}
    out: dict[str, list[Path]] = {"train": [], "eval": []}
    for p in paths:
        out["eval" if p.parent.name in eval_families else "train"].append(p)
    return out


def summarize(corpora: dict[str, Sequence]) -> str:
    lines = []
    for k, v in corpora.items():
        lines.append(f"  {k:<28s} {len(v):>6d}")
    return "\n".join(lines)
