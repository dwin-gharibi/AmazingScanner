from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils.geometry import homography, order_corners
from ..utils.imageio import imread_rgb, imwrite_rgb
from .generator import Sample, SyntheticSampleGenerator

__all__ = [
    "to_chw",
    "normalize_imagenet",
    "EnhancementDataset",
    "CornerDataset",
    "FrozenPairSet",
    "FrozenCornerSet",
    "RealPhotoSet",
    "freeze_pairs",
    "freeze_corners",
]

DOC_MEAN = (0.780, 0.762, 0.741)
DOC_STD = (0.204, 0.208, 0.213)


def to_chw(img: np.ndarray) -> torch.Tensor:
    if img.dtype == np.uint8:
        arr = img.astype(np.float32) / 255.0
    else:
        arr = img.astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))


def normalize_imagenet(t: torch.Tensor,
                       mean: Sequence[float] = DOC_MEAN,
                       std: Sequence[float] = DOC_STD) -> torch.Tensor:
    m = torch.tensor(mean, dtype=t.dtype).view(-1, 1, 1)
    s = torch.tensor(std, dtype=t.dtype).view(-1, 1, 1)
    return (t - m) / s

def _ink_map(document: np.ndarray, cells: int = 16) -> np.ndarray:
    g = cv2.cvtColor(document, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(g, (cells, cells), interpolation=cv2.INTER_AREA).astype(np.float32)
    ink = np.clip(1.0 - small / 255.0, 0.0, 1.0)
    ink = ink ** 0.5
    total = ink.sum()
    return ink / total if total > 1e-6 else np.full_like(ink, 1.0 / ink.size)


def sample_crop_origin(
    ink: np.ndarray,
    page_w: int,
    page_h: int,
    crop: int,
    rng: np.random.Generator,
    ink_prob: float = 0.75,
) -> tuple[int, int]:
    max_x, max_y = max(page_w - crop, 0), max(page_h - crop, 0)
    if rng.random() < ink_prob:
        cells = ink.shape[0]
        idx = int(rng.choice(ink.size, p=ink.ravel()))
        cy, cx = divmod(idx, cells)
        fx = (cx + 0.5 + rng.uniform(-0.5, 0.5)) / cells
        fy = (cy + 0.5 + rng.uniform(-0.5, 0.5)) / cells
        x = int(round(fx * page_w - crop / 2))
        y = int(round(fy * page_h - crop / 2))
    else:
        x = int(rng.integers(0, max_x + 1)) if max_x > 0 else 0
        y = int(rng.integers(0, max_y + 1)) if max_y > 0 else 0
    return int(np.clip(x, 0, max_x)), int(np.clip(y, 0, max_y))


def rectify_crop(
    sample: Sample,
    document: np.ndarray,
    page_size: tuple[int, int],
    origin: tuple[int, int],
    crop: int,
) -> tuple[np.ndarray, np.ndarray]:
    pw, ph = page_size
    x0, y0 = origin
    dh, dw = document.shape[:2]

    page_corners = np.array([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]], np.float32)
    doc_corners = np.array([[0, 0], [dw - 1, 0], [dw - 1, dh - 1], [0, dh - 1]], np.float32)
    shift = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float32)

    src_corners = sample.corners_content if sample.corners_content is not None else sample.corners
    h_in = shift @ homography(src_corners, page_corners)
    h_tg = shift @ homography(doc_corners, page_corners)

    inp = cv2.warpPerspective(sample.photo, h_in, (crop, crop),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    tgt = cv2.warpPerspective(document, h_tg, (crop, crop),
                              flags=cv2.INTER_AREA if dw > pw else cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    return inp, tgt


class EnhancementDataset(Dataset):
    def __init__(
        self,
        generator: SyntheticSampleGenerator,
        length: int = 4096,
        crop: int = 256,
        crops_per_photo: int = 6,
        page_long_side: int | None = None,
        page_size_range: tuple[int, int] | None = (640, 1152),
        seed: int = 0,
        ink_prob: float = 0.75,
    ):
        self.gen = generator
        self.length = int(length)
        self.crop = int(crop)
        self.crops_per_photo = max(1, int(crops_per_photo))
        self.page_long_side = page_long_side or generator.page_long_side
        self.page_size_range = page_size_range
        self.seed = int(seed)
        self.ink_prob = float(ink_prob)
        self._epoch = 0
        self._cache_key: int | None = None
        self._cache: tuple[Sample, np.ndarray, tuple[int, int], np.ndarray] | None = None

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        self._cache_key = None
        self._cache = None

    def __len__(self) -> int:
        return self.length

    def _photo_for(self, photo_id: int):
        if self._cache_key == photo_id and self._cache is not None:
            return self._cache
        rng = np.random.default_rng((self.seed, self._epoch, photo_id))
        sample = self.gen.generate(rng, want_rectified=False)
        document = sample.document
        assert document is not None
        dh, dw = document.shape[:2]
        long_side = self.page_long_side
        if self.page_size_range:
            lo, hi = self.page_size_range
            long_side = int(rng.integers(lo, hi + 1))
        long_side = max(long_side, self.crop)
        if dw >= dh:
            pw = long_side
            ph = max(16, int(round(pw * dh / dw)))
        else:
            ph = long_side
            pw = max(16, int(round(ph * dw / dh)))
        ink = _ink_map(document)
        self._cache_key = photo_id
        self._cache = (sample, document, (pw, ph), ink)
        return self._cache

    def __getitem__(self, index: int):
        photo_id, crop_id = divmod(int(index), self.crops_per_photo)
        sample, document, page_size, ink = self._photo_for(photo_id)
        rng = np.random.default_rng((self.seed, self._epoch, photo_id, crop_id, 11))
        origin = sample_crop_origin(ink, page_size[0], page_size[1], self.crop, rng,
                                    self.ink_prob)
        inp, tgt = rectify_crop(sample, document, page_size, origin, self.crop)
        return to_chw(inp), to_chw(tgt)


class CornerDataset(Dataset):
    def __init__(
        self,
        generator: SyntheticSampleGenerator,
        length: int = 4096,
        size: int = 256,
        seed: int = 0,
        normalize: bool = True,
    ):
        self.gen = generator
        self.length = int(length)
        self.size = int(size)
        self.seed = int(seed)
        self.normalize = normalize
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        rng = np.random.default_rng((self.seed, self._epoch, int(index)))
        s = self.gen.generate(rng, want_rectified=False)
        img, corners = resize_with_corners(s.photo, s.corners, self.size)
        t = to_chw(img)
        if self.normalize:
            t = normalize_imagenet(t)
        c = torch.from_numpy((corners / self.size).astype(np.float32))
        return t, c


def resize_with_corners(
    img: np.ndarray, corners: np.ndarray, size: int | tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(size, int):
        out_w = out_h = size
    else:
        out_w, out_h = size
    h, w = img.shape[:2]
    sx, sy = out_w / float(w), out_h / float(h)
    interp = cv2.INTER_AREA if (out_w * out_h) < (w * h) else cv2.INTER_LINEAR
    out = cv2.resize(img, (out_w, out_h), interpolation=interp)
    pts = np.asarray(corners, np.float32).reshape(-1, 2).copy()
    pts[:, 0] *= sx
    pts[:, 1] *= sy
    return out, pts


def freeze_pairs(
    generator: SyntheticSampleGenerator,
    out_dir: str | Path,
    n: int,
    seed: int,
    page_long_side: int = 768,
    jpeg_quality: int = 100,
) -> Path:
    out_dir = Path(out_dir)
    (out_dir / "input").mkdir(parents=True, exist_ok=True)
    (out_dir / "target").mkdir(parents=True, exist_ok=True)
    meta = []
    gen_page = generator.page_long_side
    generator.page_long_side = page_long_side
    try:
        for i in range(n):
            rng = np.random.default_rng((seed, i))
            s = generator.generate(rng, want_rectified=True)
            imwrite_rgb(out_dir / "input" / f"{i:04d}.png", s.rectified)
            imwrite_rgb(out_dir / "target" / f"{i:04d}.png", s.target)
            meta.append({
                "id": i,
                "scan_path": s.meta["scan_path"],
                "page_size": list(s.page_size),
                "ops": s.trace.names if s.trace else [],
            })
    finally:
        generator.page_long_side = gen_page
    (out_dir / "manifest.json").write_text(json.dumps({"n": n, "seed": seed,
                                                       "items": meta}, indent=1))
    return out_dir


def freeze_corners(
    generator: SyntheticSampleGenerator,
    out_dir: str | Path,
    n: int,
    seed: int,
    size: int = 512,
) -> Path:
    out_dir = Path(out_dir)
    (out_dir / "photo").mkdir(parents=True, exist_ok=True)
    items = []
    for i in range(n):
        rng = np.random.default_rng((seed, i, 3))
        s = generator.generate(rng, want_rectified=False)
        img, corners = resize_with_corners(s.photo, s.corners, size)
        imwrite_rgb(out_dir / "photo" / f"{i:04d}.png", img)
        items.append({
            "id": i, "file": f"photo/{i:04d}.png",
            "corners": np.round(corners, 3).tolist(),
            "size": [size, size],
            "scan_path": s.meta["scan_path"],
            "ops": s.trace.names if s.trace else [],
        })
    (out_dir / "annotations.json").write_text(json.dumps({"n": n, "seed": seed,
                                                          "items": items}, indent=1))
    return out_dir


@dataclass
class FrozenPairSet(Dataset):
    root: Path
    crop: int | None = None
    crops_per_page: int = 4
    seed: int = 1234

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.inputs = sorted((self.root / "input").glob("*.png"))
        self.targets = sorted((self.root / "target").glob("*.png"))
        if len(self.inputs) != len(self.targets):
            raise ValueError(f"{self.root}: input/target count mismatch")
        self._plan: list[tuple[int, tuple[int, int]]] = []
        if self.crop:
            for i, p in enumerate(self.inputs):
                rng = np.random.default_rng((self.seed, i))
                img = imread_rgb(p, exif=False)
                h, w = img.shape[:2]
                ink = _ink_map(img)
                for _ in range(self.crops_per_page):
                    self._plan.append((i, sample_crop_origin(ink, w, h, self.crop, rng)))

    def __len__(self) -> int:
        return len(self._plan) if self.crop else len(self.inputs)

    def page(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        return imread_rgb(self.inputs[i], exif=False), imread_rgb(self.targets[i], exif=False)

    def __getitem__(self, index: int):
        if self.crop:
            i, (x, y) = self._plan[index]
            a, b = self.page(i)
            c = self.crop
            return (to_chw(a[y:y + c, x:x + c]), to_chw(b[y:y + c, x:x + c]))
        a, b = self.page(index)
        return to_chw(a), to_chw(b)


@dataclass
class FrozenCornerSet(Dataset):
    root: Path
    size: int = 256
    normalize: bool = True

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        data = json.loads((self.root / "annotations.json").read_text())
        self.items = data["items"]

    def __len__(self) -> int:
        return len(self.items)

    def raw(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        it = self.items[index]
        img = imread_rgb(self.root / it["file"], exif=False)
        return img, np.asarray(it["corners"], np.float32)

    def __getitem__(self, index: int):
        img, corners = self.raw(index)
        img, corners = resize_with_corners(img, corners, self.size)
        t = to_chw(img)
        if self.normalize:
            t = normalize_imagenet(t)
        return t, torch.from_numpy((corners / self.size).astype(np.float32))


@dataclass
class RealPhotoSet(Dataset):
    manifest: Path
    size: int = 256
    normalize: bool = True

    def __post_init__(self) -> None:
        self.manifest = Path(self.manifest)
        data = json.loads(self.manifest.read_text())
        self.root = self.manifest.parent
        self.items = data["items"]
        self.name = data.get("name", self.manifest.stem)

    def __len__(self) -> int:
        return len(self.items)

    def raw(self, index: int) -> tuple[np.ndarray, np.ndarray, dict]:
        it = self.items[index]
        img = imread_rgb(self.root / it["file"])
        corners = order_corners(np.asarray(it["corners"], np.float32))
        return img, corners, it

    def reference(self, index: int) -> np.ndarray | None:
        ref = self.items[index].get("reference")
        return imread_rgb(self.root / ref) if ref else None

    def __getitem__(self, index: int):
        img, corners, _ = self.raw(index)
        img, corners = resize_with_corners(img, corners, self.size)
        t = to_chw(img)
        if self.normalize:
            t = normalize_imagenet(t)
        return t, torch.from_numpy((corners / self.size).astype(np.float32))
