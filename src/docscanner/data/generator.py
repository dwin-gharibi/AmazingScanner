from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from ..utils.geometry import homography, order_corners, quad_area, quad_is_convex
from .corpus import BackgroundCorpus, ScanCorpus
from .degrade import (
    DegradationConfig,
    DegradationPipeline,
    DegradationTrace,
    random_document_quad,
    to_uint8,
    warp_onto_background,
)

__all__ = ["Sample", "SyntheticSampleGenerator", "rot90_with_points"]


def rot90_matrix(width: int, height: int, k: int) -> np.ndarray:
    m = np.eye(3, dtype=np.float32)
    w, h = int(width), int(height)
    for _ in range(int(k) % 4):
        step = np.array([[0.0, 1.0, 0.0],
                         [-1.0, 0.0, w - 1.0],
                         [0.0, 0.0, 1.0]], np.float32)
        m = step @ m
        w, h = h, w
    return m


def _rot90_with_points(img: np.ndarray, pts: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    k = int(k) % 4
    if k == 0:
        return img, pts
    h, w = img.shape[:2]
    out = np.ascontiguousarray(np.rot90(img, k))
    p = np.asarray(pts, np.float32).reshape(-1, 2).copy()
    for _ in range(k):
        x, y = p[:, 0].copy(), p[:, 1].copy()
        p[:, 0] = y
        p[:, 1] = (w - 1) - x
        w, h = h, w
    return out, p


rot90_with_points = _rot90_with_points


@dataclass
class Sample:
    photo: np.ndarray
    corners: np.ndarray
    corners_content: np.ndarray | None = None
    document: np.ndarray | None = None
    rectified: np.ndarray | None = None
    target: np.ndarray | None = None
    h_rect: np.ndarray | None = None
    h_doc2photo: np.ndarray | None = None
    trace: DegradationTrace | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def page_size(self) -> tuple[int, int]:
        if self.target is None:
            return (0, 0)
        return (self.target.shape[1], self.target.shape[0])


class SyntheticSampleGenerator:
    def __init__(
        self,
        scans: ScanCorpus,
        backgrounds: BackgroundCorpus,
        cfg: DegradationConfig | None = None,
        photo_long_side: int = 1280,
        page_long_side: int = 1024,
        photo_aspects: tuple[float, ...] = (0.75, 1.0, 1.3333),
        min_page_area_frac: float = 0.10,
    ):
        self.scans = scans
        self.backgrounds = backgrounds
        self.cfg = cfg or DegradationConfig()
        self.pipeline = DegradationPipeline(self.cfg)
        self.photo_long_side = int(photo_long_side)
        self.page_long_side = int(page_long_side)
        self.photo_aspects = photo_aspects
        self.min_page_area_frac = float(min_page_area_frac)

    def _photo_canvas_size(self, rng: np.random.Generator) -> tuple[int, int]:
        aspect = float(rng.choice(self.photo_aspects))
        if aspect >= 1.0:
            w = self.photo_long_side
            h = int(round(w / aspect))
        else:
            h = self.photo_long_side
            w = int(round(h * aspect))
        return w, h

    def _distractor_source(self, rng: np.random.Generator) -> np.ndarray | None:
        cache = getattr(self, "_distractor_cache", None)
        if cache is None:
            cache = []
            n = min(6, len(self.scans.paths))
            for k in range(n):
                try:
                    img = self.scans.load(k * max(1, len(self.scans.paths) // max(n, 1)))
                except Exception:
                    continue
                scale = 420.0 / max(img.shape[:2])
                if scale < 1.0:
                    img = cv2.resize(img, (max(8, int(img.shape[1] * scale)),
                                           max(8, int(img.shape[0] * scale))),
                                     interpolation=cv2.INTER_AREA)
                cache.append(np.ascontiguousarray(img))
            self._distractor_cache = cache
        if not cache:
            return None
        return cache[int(rng.integers(0, len(cache)))]

    def _add_distractor_pages(self, background: np.ndarray,
                              rng: np.random.Generator,
                              main_quad: np.ndarray | None = None) -> np.ndarray:
        p = float(getattr(self.cfg, "p_distractor_page", 0.0))
        if p <= 0.0 or rng.random() >= p:
            return background

        h, w = background.shape[:2]
        out = background
        for _ in range(int(rng.integers(1, 3))):
            try:
                other = self._distractor_source(rng)
            except Exception:
                continue
            if other is None:
                continue
            quad = self._distractor_quad(rng, (h, w), main_quad)
            if quad is None or not quad_is_convex(quad):
                continue
            try:
                out, _ = warp_onto_background(other, out, order_corners(quad), rng,
                                              self.cfg, None)
            except Exception:
                continue

            out = np.clip(out * float(rng.uniform(0.82, 0.97)), 0.0, 1.0)
        return out.astype(np.float32)

    def _distractor_quad(self, rng: np.random.Generator, shape: tuple[int, int],
                         main: np.ndarray | None) -> np.ndarray | None:
        h, w = shape
        mode = rng.choice(["spread", "behind", "elsewhere"], p=[0.4, 0.35, 0.25])

        if main is not None and mode in ("spread", "behind"):
            q = np.asarray(main, np.float32).reshape(4, 2)
            if mode == "spread":
                i = int(rng.integers(0, 4))
                a, b = q[i], q[(i + 1) % 4]
                d = b - a
                n2 = float(d @ d)
                if n2 < 1e-6:
                    return None
                t = ((q - a) @ d) / n2
                proj = a + t[:, None] * d
                quad = 2.0 * proj - q
                quad = quad + rng.normal(0, 0.02, (4, 2)).astype(np.float32) * np.float32([w, h])
            else:
                centre = q.mean(axis=0)
                scale = float(rng.uniform(1.02, 1.35))
                shift = rng.uniform(-0.42, 0.42, 2).astype(np.float32) * np.float32(
                    [np.ptp(q[:, 0]), np.ptp(q[:, 1])])
                quad = (q - centre) * scale + centre + shift
            return quad.astype(np.float32)

        side = int(rng.integers(0, 4))
        cx = {0: 0.14, 1: 0.86}.get(side, float(rng.uniform(0.2, 0.8)))
        cy = {2: 0.14, 3: 0.86}.get(side, float(rng.uniform(0.2, 0.8)))
        scale = float(rng.uniform(0.34, 0.72))
        half_w, half_h = 0.5 * scale * w, 0.5 * scale * h
        jitter = rng.normal(0, 0.05, (4, 2)) * np.float32([w, h])
        return (np.float32([
            [cx * w - half_w, cy * h - half_h], [cx * w + half_w, cy * h - half_h],
            [cx * w + half_w, cy * h + half_h], [cx * w - half_w, cy * h + half_h],
        ]) + jitter.astype(np.float32))

    def _page_raster_size(self, doc_w: int, doc_h: int) -> tuple[int, int]:
        aspect = doc_w / max(doc_h, 1)
        if aspect >= 1.0:
            w = self.page_long_side
            h = max(16, int(round(w / aspect)))
        else:
            h = self.page_long_side
            w = max(16, int(round(h * aspect)))
        return w, h

    def _draw_inner_frame(self, document: np.ndarray,
                          rng: np.random.Generator) -> np.ndarray:
        out = document.copy()
        h, w = out.shape[:2]
        mlo, mhi = self.cfg.inner_frame_margin
        mx = int(w * rng.uniform(mlo, mhi))
        my = int(h * rng.uniform(mlo, mhi))
        if mx < 2 or my < 2 or w - 2 * mx < 8 or h - 2 * my < 8:
            return document
        tlo, thi = self.cfg.inner_frame_thickness
        thick = max(1, int(min(h, w) * rng.uniform(tlo, thi)))
        dlo, dhi = self.cfg.inner_frame_darkness

        patch = out[my:h - my, mx:w - mx]
        level = float(np.median(patch)) if patch.size else 200.0
        ink = int(np.clip(level * (1.0 - rng.uniform(dlo, dhi)), 0, 255))
        colour = (ink, ink, ink) if out.ndim == 3 else ink
        cv2.rectangle(out, (mx, my), (w - mx - 1, h - my - 1), colour, thick,
                      cv2.LINE_AA)
        return out

    def _tint_page(self, document: np.ndarray,
                   rng: np.random.Generator) -> np.ndarray:
        out = document.astype(np.float32)
        if out.ndim == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
        lo, hi = self.cfg.page_tint_saturation
        mix = float(rng.uniform(lo, hi))
        hue = rng.uniform(0, 180)
        colour = cv2.cvtColor(
            np.array([[[hue, 200, 255]]], np.uint8), cv2.COLOR_HSV2RGB
        ).astype(np.float32).reshape(1, 1, 3)
        out = out * (1.0 - mix) + (out / 255.0) * colour * mix
        dlo, dhi = self.cfg.page_tint_darken
        out *= float(rng.uniform(dlo, dhi))
        return np.clip(out, 0, 255).astype(np.uint8)

    def _draw_page_band(self, document: np.ndarray,
                        rng: np.random.Generator) -> np.ndarray:
        out = document.copy()
        if out.ndim == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
        h, w = out.shape[:2]
        lo, hi = self.cfg.page_band_extent
        extent = float(rng.uniform(lo, hi))
        dlo, dhi = self.cfg.page_band_darkness
        level = float(np.median(out))
        shade = np.clip(level * (1.0 - rng.uniform(dlo, dhi)), 0, 255)
        hue = rng.uniform(0, 180)
        colour = cv2.cvtColor(np.array([[[hue, int(rng.uniform(0, 180)), 255]]], np.uint8),
                              cv2.COLOR_HSV2RGB).astype(np.float32).reshape(3)
        colour = np.clip(colour / 255.0 * shade, 0, 255)

        side = int(rng.integers(0, 4))
        if side == 0:
            out[:max(1, int(h * extent))] = colour
        elif side == 1:
            out[h - max(1, int(h * extent)):] = colour
        elif side == 2:
            out[:, :max(1, int(w * extent))] = colour
        else:
            out[:, w - max(1, int(w * extent)):] = colour
        return out

    @staticmethod
    def _edge_normal(quad: np.ndarray, i: int) -> tuple[np.ndarray, np.ndarray]:
        a, b = quad[i], quad[(i + 1) % 4]
        d = b - a
        n = np.array([-d[1], d[0]], np.float32)
        ln = float(np.linalg.norm(n)) or 1.0
        n /= ln
        if float(np.dot(n, quad.mean(axis=0) - a)) > 0:
            n = -n
        return n, d / (float(np.linalg.norm(d)) or 1.0)

    def _add_page_stack(self, background: np.ndarray, quad: np.ndarray,
                        rng: np.random.Generator) -> np.ndarray:
        out = background
        lo, hi = self.cfg.page_stack_layers
        layers = int(rng.integers(lo, hi + 1))
        long_side = float(max(np.linalg.norm(quad[1] - quad[0]),
                              np.linalg.norm(quad[2] - quad[1])))
        olo, ohi = self.cfg.page_stack_offset
        centre = quad.mean(axis=0)
        drift = rng.normal(0, 1, 2).astype(np.float32)
        drift /= (float(np.linalg.norm(drift)) or 1.0)
        for k in range(layers, 0, -1):
            step = long_side * float(rng.uniform(olo, ohi)) * k
            grow = 1.0 + float(rng.uniform(0.002, 0.012)) * k
            sheet = centre + (quad - centre) * grow + drift * step
            shade = int(np.clip(rng.normal(228, 16), 150, 252))
            cv2.fillConvexPoly(out, np.round(sheet).astype(np.int32),
                               (shade, shade, int(np.clip(shade - rng.uniform(0, 10), 0, 255))),
                               cv2.LINE_AA)
        return out

    def _add_binding(self, background: np.ndarray, quad: np.ndarray,
                     rng: np.random.Generator) -> np.ndarray:
        out = background
        i = int(rng.integers(0, 4))
        a, b = quad[i], quad[(i + 1) % 4]
        n, _ = self._edge_normal(quad, i)
        short_side = float(min(np.linalg.norm(quad[1] - quad[0]),
                               np.linalg.norm(quad[2] - quad[1])))
        rlo, rhi = self.cfg.binding_reach
        reach = max(3.0, short_side * float(rng.uniform(rlo, rhi)))
        nlo, nhi = self.cfg.binding_rings
        rings = int(rng.integers(nlo, nhi + 1))
        dark = int(rng.integers(25, 95))
        light = int(rng.integers(160, 240))

        angle = float(np.degrees(np.arctan2(n[1], n[0])))
        lean = float(rng.uniform(-14, 14))
        ax = max(2, int(reach * 0.75))
        ay = max(1, int(reach * rng.uniform(0.16, 0.30)))
        thick = max(1, int(reach * rng.uniform(0.10, 0.22)))
        for t in np.linspace(0.03, 0.97, rings):
            p = a + (b - a) * float(t)

            c = tuple(np.round(p + n * reach * 0.45).astype(int))
            cv2.ellipse(out, c, (ax, ay), angle + lean, 0, 360,
                        (light, light, light), thick + 2, cv2.LINE_AA)
            cv2.ellipse(out, c, (ax, ay), angle + lean, 0, 360,
                        (dark, dark, dark), thick, cv2.LINE_AA)
        return out

    def _add_gutter(self, background: np.ndarray, quad: np.ndarray,
                    rng: np.random.Generator) -> np.ndarray:
        h, w = background.shape[:2]
        i = int(rng.integers(0, 4))
        a, b = quad[i], quad[(i + 1) % 4]
        n, _ = self._edge_normal(quad, i)
        short_side = float(min(np.linalg.norm(quad[1] - quad[0]),
                               np.linalg.norm(quad[2] - quad[1])))
        wlo, whi = self.cfg.gutter_width
        band = max(2.0, short_side * float(rng.uniform(wlo, whi)))
        poly = np.array([a, b, b + n * band, a + n * band], np.float32)

        mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask, np.round(poly).astype(np.int32), 255, cv2.LINE_AA)
        mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, band * 0.25))
        dlo, dhi = self.cfg.gutter_darkness
        strength = float(rng.uniform(dlo, dhi))
        m = (mask.astype(np.float32) / 255.0 * strength)[..., None]
        return np.clip(background.astype(np.float32) * (1.0 - m), 0, 255).astype(np.uint8)

    def _draw_ruling(self, document: np.ndarray,
                     rng: np.random.Generator) -> np.ndarray:
        out = document.copy()
        if out.ndim == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
        h, w = out.shape[:2]
        slo, shi = self.cfg.ruled_spacing
        step = max(3, int(h * rng.uniform(slo, shi)))
        dlo, dhi = self.cfg.ruled_darkness
        level = float(np.median(out))
        ink = int(np.clip(level * (1.0 - rng.uniform(dlo, dhi)), 0, 255))
        tint = rng.uniform(0.6, 1.0, 3)
        colour = tuple(int(np.clip(ink * t + 255 * (1 - t) * 0.15, 0, 255)) for t in tint)
        thick = 1 if min(h, w) < 700 else 2
        grid = rng.random() < 0.35
        for y in range(step, h - 1, step):
            cv2.line(out, (0, y), (w - 1, y), colour, thick, cv2.LINE_AA)
        if grid:
            for x in range(step, w - 1, step):
                cv2.line(out, (x, 0), (x, h - 1), colour, thick, cv2.LINE_AA)
        return out

    def _outside_colour(self, photo_f: np.ndarray, quad: np.ndarray,
                        i: int) -> np.ndarray:
        h, w = photo_f.shape[:2]
        centre = quad.mean(axis=0)
        d = quad[i] - centre
        d /= (float(np.linalg.norm(d)) or 1.0)
        p = quad[i] + d * max(4.0, 0.03 * float(np.linalg.norm(quad[2] - quad[0])))
        x, y = int(np.clip(p[0], 2, w - 3)), int(np.clip(p[1], 2, h - 3))
        return np.median(photo_f[y - 2:y + 3, x - 2:x + 3].reshape(-1, 3), axis=0)

    def _round_corners(self, photo_f: np.ndarray, quad: np.ndarray,
                       rng: np.random.Generator) -> np.ndarray:
        short = float(min(np.linalg.norm(quad[1] - quad[0]),
                          np.linalg.norm(quad[2] - quad[1])))
        rlo, rhi = self.cfg.round_corner_radius
        r = max(2.0, short * float(rng.uniform(rlo, rhi)))
        out = photo_f
        for i in range(4):
            prev, cur, nxt = quad[i - 1], quad[i], quad[(i + 1) % 4]
            u = (prev - cur) / (float(np.linalg.norm(prev - cur)) or 1.0)
            v = (nxt - cur) / (float(np.linalg.norm(nxt - cur)) or 1.0)
            tri = np.array([cur, cur + u * r, cur + v * r], np.float32)
            mask = np.zeros(photo_f.shape[:2], np.uint8)
            cv2.fillConvexPoly(mask, np.round(tri).astype(np.int32), 255, cv2.LINE_AA)
            m = (cv2.GaussianBlur(mask, (0, 0), 1.2).astype(np.float32) / 255.0)[..., None]
            out = out * (1 - m) + self._outside_colour(photo_f, quad, i)[None, None, :] * m
        return out

    def _dog_ear(self, photo_f: np.ndarray, quad: np.ndarray,
                 rng: np.random.Generator) -> np.ndarray:
        i = int(rng.integers(0, 4))
        prev, cur, nxt = quad[i - 1], quad[i], quad[(i + 1) % 4]
        slo, shi = self.cfg.dog_ear_size
        s = float(rng.uniform(slo, shi))
        a = cur + (prev - cur) * s
        b = cur + (nxt - cur) * s
        tri = np.array([a, cur, b], np.float32)
        mask = np.zeros(photo_f.shape[:2], np.uint8)
        cv2.fillConvexPoly(mask, np.round(tri).astype(np.int32), 255, cv2.LINE_AA)
        m = (cv2.GaussianBlur(mask, (0, 0), 1.0).astype(np.float32) / 255.0)[..., None]
        back = float(np.clip(np.median(photo_f) * rng.uniform(0.72, 1.06), 0, 1))
        return photo_f * (1 - m) + np.full(3, back, np.float32)[None, None, :] * m

    def _corner_occluder(self, photo_f: np.ndarray, quad: np.ndarray,
                         rng: np.random.Generator) -> np.ndarray:
        i = int(rng.integers(0, 4))
        short = float(min(np.linalg.norm(quad[1] - quad[0]),
                          np.linalg.norm(quad[2] - quad[1])))
        slo, shi = self.cfg.corner_occluder_size
        size = max(3.0, short * float(rng.uniform(slo, shi)))
        centre = quad[i] + (quad.mean(axis=0) - quad[i]) * float(rng.uniform(-0.25, 0.45))
        mask = np.zeros(photo_f.shape[:2], np.uint8)
        if rng.random() < 0.5:
            cv2.ellipse(mask, tuple(np.round(centre).astype(int)),
                        (int(size), int(size * rng.uniform(0.5, 1.4))),
                        float(rng.uniform(0, 180)), 0, 360, 255, -1, cv2.LINE_AA)
        else:
            box = cv2.boxPoints(((float(centre[0]), float(centre[1])),
                                 (size * 2, size * 2 * float(rng.uniform(0.6, 1.5))),
                                 float(rng.uniform(0, 180))))
            cv2.fillConvexPoly(mask, np.round(box).astype(np.int32), 255, cv2.LINE_AA)
        m = (cv2.GaussianBlur(mask, (0, 0), 1.5).astype(np.float32) / 255.0)[..., None]
        colour = rng.uniform(0.12, 0.92, 3).astype(np.float32)
        return photo_f * (1 - m) + colour[None, None, :] * m

    def _backlit(self, photo_f: np.ndarray, quad: np.ndarray,
                 rng: np.random.Generator) -> np.ndarray:
        mask = np.zeros(photo_f.shape[:2], np.uint8)
        cv2.fillConvexPoly(mask, np.round(quad).astype(np.int32), 255, cv2.LINE_AA)
        outside = 1.0 - (cv2.GaussianBlur(mask, (0, 0), 3.0).astype(np.float32) / 255.0)
        lo, hi = self.cfg.backlit_strength
        k = float(rng.uniform(lo, hi))
        out = photo_f + outside[..., None] * k * (1.0 - photo_f)
        inside = 1.0 - outside
        out = out * (1.0 - inside[..., None] * k * 0.35)
        return np.clip(out, 0.0, 1.0)

    def generate(
        self,
        rng: np.random.Generator | int,
        want_rectified: bool = True,
        keep_stages: bool = False,
        scan_index: int | None = None,
        page_size: tuple[int, int] | None = None,
        max_tries: int = 8,
    ) -> Sample:
        if isinstance(rng, (int, np.integer)):
            rng = np.random.default_rng(int(rng))

        idx = self.scans.sample_index(rng) if scan_index is None else int(scan_index)
        document = self.scans.load_document(idx, rng)
        def _maybe(attr: str) -> bool:
            p = float(getattr(self.cfg, attr, 0.0))
            return p > 0.0 and rng.random() < p

        if rng.random() < float(getattr(self.cfg, "p_inner_frame", 0.0)):
            document = self._draw_inner_frame(document, rng)

        if _maybe("p_page_band"):
            document = self._draw_page_band(document, rng)
        if _maybe("p_ruled_paper"):
            document = self._draw_ruling(document, rng)
        if _maybe("p_page_tint"):
            document = self._tint_page(document, rng)
        doc_h, doc_w = document.shape[:2]

        canvas_w, canvas_h = self._photo_canvas_size(rng)
        background = self.backgrounds.canvas(canvas_w, canvas_h, rng)

        quad = None
        for _ in range(max_tries):
            cand = random_document_quad(canvas_w, canvas_h, doc_w / doc_h, rng, self.cfg)
            if quad_is_convex(cand) and quad_area(cand) > self.min_page_area_frac * canvas_w * canvas_h:
                quad = cand
                break
        if quad is None:
            m = 0.12
            quad = order_corners(np.array([
                [canvas_w * m, canvas_h * m], [canvas_w * (1 - m), canvas_h * m],
                [canvas_w * (1 - m), canvas_h * (1 - m)], [canvas_w * m, canvas_h * (1 - m)],
            ], np.float32))

        if _maybe("p_pale_background"):
            lo, hi = self.cfg.pale_background_level
            k = float(rng.uniform(lo, hi))
            background = np.clip(background.astype(np.float32) * (1 - k) + 255.0 * k,
                                 0, 255).astype(np.uint8)

        background = self._add_distractor_pages(background, rng, quad)
        if _maybe("p_page_stack"):
            background = self._add_page_stack(background, quad, rng)
        if _maybe("p_gutter"):
            background = self._add_gutter(background, quad, rng)
        if _maybe("p_binding"):
            background = self._add_binding(background, quad, rng)

        trace = DegradationTrace()
        photo_f, h_doc2photo = warp_onto_background(
            document, background, quad, rng, self.cfg, trace
        )
        if _maybe("p_round_corners"):
            photo_f = self._round_corners(photo_f, quad, rng)
        if _maybe("p_dog_ear"):
            photo_f = self._dog_ear(photo_f, quad, rng)
        if _maybe("p_corner_occluder"):
            photo_f = self._corner_occluder(photo_f, quad, rng)
        if _maybe("p_backlit"):
            photo_f = self._backlit(photo_f, quad, rng)
        photo_f, trace = self.pipeline(photo_f, rng, keep_stages=keep_stages, trace=trace)
        photo = to_uint8(photo_f)

        corners_content = np.asarray(quad, np.float32)
        if getattr(self.cfg, "p_rot90", 0.0) > 0 and rng.random() < self.cfg.p_rot90:
            k = int(rng.integers(1, 4))
            rot = rot90_matrix(photo.shape[1], photo.shape[0], k)
            photo, corners_content = _rot90_with_points(photo, corners_content, k)
            h_doc2photo = rot @ h_doc2photo
            trace.add("rot90", turns=k)

        sample = Sample(
            photo=photo,
            corners=order_corners(corners_content),
            corners_content=corners_content,
            document=document,
            h_doc2photo=h_doc2photo,
            trace=trace,
            meta={
                "scan_index": idx,
                "scan_path": str(self.scans.paths[idx % len(self.scans.paths)]),
                "doc_size": (doc_w, doc_h),
                "photo_size": (canvas_w, canvas_h),
                "config": self.cfg.name,
            },
        )

        if want_rectified:
            pw, ph = page_size if page_size else self._page_raster_size(doc_w, doc_h)
            page_corners = np.array(
                [[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]], np.float32
            )
            doc_corners = np.array(
                [[0, 0], [doc_w - 1, 0], [doc_w - 1, doc_h - 1], [0, doc_h - 1]], np.float32
            )
            h_target = homography(doc_corners, page_corners)
            h_rect = homography(sample.corners_content, page_corners)

            target = cv2.warpPerspective(
                document, h_target, (pw, ph),
                flags=cv2.INTER_AREA if doc_w > pw else cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )
            rectified = cv2.warpPerspective(
                sample.photo, h_rect, (pw, ph),
                flags=cv2.INTER_AREA if quad_area(sample.corners) > pw * ph else cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )
            sample.rectified = rectified
            sample.target = target
            sample.h_rect = h_rect

        return sample

    def generate_many(self, n: int, seed: int = 0, **kw):
        for i in range(n):
            yield self.generate(np.random.default_rng(seed * 1_000_003 + i), **kw)
