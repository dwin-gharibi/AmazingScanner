from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import cv2
import numpy as np

from ..utils.geometry import homography, order_corners

Array = np.ndarray
RNG = np.random.Generator

__all__ = [
    "DegradationConfig",
    "DegradationTrace",
    "DegradationPipeline",
    "random_document_quad",
    "warp_onto_background",
    "to_float",
    "to_uint8",
]


def to_float(img: Array) -> Array:
    if img.dtype == np.float32:
        return img
    return img.astype(np.float32) / 255.0


def to_uint8(img: Array) -> Array:
    if img.dtype == np.uint8:
        return img
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _u(rng: RNG, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def _smooth_field(rng: RNG, h: int, w: int, ctrl: int = 4) -> Array:
    grid = rng.random((ctrl, ctrl)).astype(np.float32)
    field = cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
    lo, hi = float(field.min()), float(field.max())
    if hi - lo < 1e-6:
        return np.full((h, w), 0.5, np.float32)
    return (field - lo) / (hi - lo)


_LOWRES_DIV = 8
_MASK_DIV = 4


def _lowres_shape(h: int, w: int, div: int) -> tuple[int, int]:
    return max(8, h // div), max(8, w // div)


_GRID_CACHE: dict[tuple[int, int], tuple[Array, Array]] = {}


def _norm_grid(h: int, w: int) -> tuple[Array, Array]:
    key = (h, w)
    cached = _GRID_CACHE.get(key)
    if cached is None:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cached = (xx / max(w - 1, 1), yy / max(h - 1, 1))
        if len(_GRID_CACHE) > 32:
            _GRID_CACHE.clear()
        _GRID_CACHE[key] = cached
    return cached


def _clip01(img: Array) -> Array:
    return np.clip(img, 0.0, 1.0, out=img)


def _odd(n: int) -> int:
    n = int(n)
    return n if n % 2 == 1 else n + 1


@dataclass
class DegradationConfig:
    quad_margin: tuple[float, float] = (0.03, 0.16)
    quad_jitter: float = 0.16
    quad_rotation_deg: tuple[float, float] = (-40.0, 40.0)
    p_rot90: float = 0.35
    p_distractor_page: float = 0.0
    p_inner_frame: float = 0.0
    inner_frame_margin: tuple[float, float] = (0.04, 0.18)
    inner_frame_thickness: tuple[float, float] = (0.002, 0.012)
    inner_frame_darkness: tuple[float, float] = (0.15, 0.75)
    p_page_tint: float = 0.0
    page_tint_saturation: tuple[float, float] = (0.10, 0.55)
    page_tint_darken: tuple[float, float] = (0.25, 0.95)

    p_page_band: float = 0.0
    page_band_extent: tuple[float, float] = (0.15, 0.45)
    page_band_darkness: tuple[float, float] = (0.35, 0.92)
    p_page_stack: float = 0.0
    page_stack_layers: tuple[int, int] = (1, 3)
    page_stack_offset: tuple[float, float] = (0.004, 0.022)

    p_binding: float = 0.0
    binding_rings: tuple[int, int] = (9, 22)
    binding_reach: tuple[float, float] = (0.05, 0.15)

    p_gutter: float = 0.0
    gutter_width: tuple[float, float] = (0.02, 0.07)
    gutter_darkness: tuple[float, float] = (0.35, 0.85)
    p_round_corners: float = 0.0
    round_corner_radius: tuple[float, float] = (0.010, 0.055)

    p_dog_ear: float = 0.0
    dog_ear_size: tuple[float, float] = (0.05, 0.16)

    p_corner_occluder: float = 0.0
    corner_occluder_size: tuple[float, float] = (0.04, 0.14)

    p_backlit: float = 0.0
    backlit_strength: tuple[float, float] = (0.25, 0.75)

    p_pale_background: float = 0.0
    pale_background_level: tuple[float, float] = (0.55, 0.95)

    p_ruled_paper: float = 0.0
    ruled_spacing: tuple[float, float] = (0.03, 0.09)
    ruled_darkness: tuple[float, float] = (0.08, 0.42)
    page_scale: tuple[float, float] = (0.72, 0.99)
    p_drop_shadow: float = 0.85
    drop_shadow_strength: tuple[float, float] = (0.15, 0.55)
    drop_shadow_blur: tuple[float, float] = (0.006, 0.030)
    drop_shadow_offset: tuple[float, float] = (0.002, 0.020)
    p_paper_texture: float = 0.5
    paper_texture_strength: tuple[float, float] = (0.008, 0.035)

    p_downscale: float = 0.9
    downscale_factor: tuple[float, float] = (1.15, 2.4)

    brightness: tuple[float, float] = (-0.18, 0.16)
    contrast: tuple[float, float] = (0.72, 1.22)
    gamma: tuple[float, float] = (0.75, 1.35)
    color_cast_r: tuple[float, float] = (0.90, 1.10)
    color_cast_b: tuple[float, float] = (0.88, 1.12)
    saturation: tuple[float, float] = (0.75, 1.15)

    p_illumination: float = 0.95
    illumination_strength: tuple[float, float] = (0.10, 0.45)
    p_soft_shadow: float = 0.7
    soft_shadow_count: tuple[int, int] = (1, 3)
    soft_shadow_strength: tuple[float, float] = (0.12, 0.52)
    soft_shadow_blur: tuple[float, float] = (0.02, 0.14)
    p_glare: float = 0.25
    glare_strength: tuple[float, float] = (0.10, 0.40)
    p_vignette: float = 0.45
    vignette_strength: tuple[float, float] = (0.08, 0.32)

    p_tint: float = 0.35
    tint_strength: tuple[float, float] = (0.25, 0.95)

    p_lens_distortion: float = 0.0
    lens_k1: tuple[float, float] = (-0.16, 0.10)

    p_occlusion: float = 0.0
    p_crease: float = 0.22
    crease_strength: tuple[float, float] = (0.05, 0.22)
    p_show_through: float = 0.15
    show_through_strength: tuple[float, float] = (0.04, 0.16)
    p_moire: float = 0.10
    moire_strength: tuple[float, float] = (0.02, 0.09)

    p_auto_exposure: float = 0.92
    ae_target_mean: tuple[float, float] = (0.46, 0.74)
    ae_gain_range: tuple[float, float] = (0.55, 2.4)
    ae_strength: tuple[float, float] = (0.55, 1.0)
    p_blur: float = 0.85
    gaussian_sigma: tuple[float, float] = (0.4, 1.9)
    p_motion_blur: float = 0.30
    motion_len: tuple[float, float] = (0.002, 0.012)
    p_defocus: float = 0.12
    defocus_radius: tuple[float, float] = (0.0015, 0.006)
    p_chromatic: float = 0.25
    chromatic_shift: tuple[float, float] = (0.0005, 0.0028)
    p_noise: float = 0.9
    noise_sigma: tuple[float, float] = (0.002, 0.032)
    noise_grain: tuple[float, float] = (0.0, 1.1)
    p_jpeg: float = 0.9
    jpeg_quality: tuple[int, int] = (30, 85)
    p_page_curl: float = 0.0
    page_curl_strength: tuple[float, float] = (0.01, 0.05)
    name: str = "default"

    @classmethod
    def mild(cls) -> "DegradationConfig":
        return cls(
            name="mild",
            quad_jitter=0.06,
            quad_rotation_deg=(-7.0, 7.0),
            downscale_factor=(1.2, 2.2),
            brightness=(-0.10, 0.10),
            contrast=(0.85, 1.12),
            gamma=(0.9, 1.15),
            illumination_strength=(0.05, 0.22),
            soft_shadow_strength=(0.08, 0.28),
            gaussian_sigma=(0.3, 1.0),
            noise_sigma=(0.002, 0.014),
            jpeg_quality=(55, 92),
            p_motion_blur=0.15,
            p_defocus=0.05,
        )

    @classmethod
    def for_corners(cls) -> "DegradationConfig":
        return cls(
            name="corners",
            quad_margin=(0.02, 0.20),
            quad_jitter=0.26,
            quad_rotation_deg=(-60.0, 60.0),
            page_scale=(0.32, 1.02),
            p_rot90=0.45,
            p_distractor_page=0.55,
            p_inner_frame=0.45,
            p_page_tint=0.5,
            p_page_band=0.4,
            p_page_stack=0.40,
            p_binding=0.28,
            p_gutter=0.25,
            p_round_corners=0.35,
            p_dog_ear=0.16,
            p_corner_occluder=0.20,
            p_backlit=0.18,
            p_pale_background=0.22,
            p_ruled_paper=0.25,
            brightness=(-0.34, 0.16),
            contrast=(0.62, 1.24),
            gamma=(0.72, 1.95),
            illumination_strength=(0.10, 0.60),
            p_auto_exposure=0.80,
            ae_target_mean=(0.26, 0.78),
            downscale_factor=(1.2, 3.0),
            gaussian_sigma=(0.3, 2.2),
            noise_sigma=(0.002, 0.05),
            jpeg_quality=(25, 90),
            p_soft_shadow=0.75,
            soft_shadow_strength=(0.12, 0.60),
            p_glare=0.3,
            p_vignette=0.5,
            p_occlusion=0.20,
            p_tint=0.40,
            p_crease=0.18,
        )

    @classmethod
    def for_enhancement(cls) -> "DegradationConfig":
        return cls(
            name="enhancement",
            quad_jitter=0.14,
            quad_rotation_deg=(-30.0, 30.0),
            page_scale=(0.74, 0.99),
            p_rot90=0.0,
            downscale_factor=(1.15, 2.4),
            brightness=(-0.30, 0.20),
            contrast=(0.62, 1.28),
            gamma=(0.70, 1.62),
            color_cast_r=(0.86, 1.14),
            color_cast_b=(0.84, 1.16),
            p_auto_exposure=0.88,
            ae_target_mean=(0.38, 0.78),
            p_illumination=0.97,
            illumination_strength=(0.12, 0.60),
            p_soft_shadow=0.78,
            soft_shadow_strength=(0.14, 0.58),
            p_glare=0.28,
            p_vignette=0.5,
            gaussian_sigma=(0.4, 2.1),
            p_motion_blur=0.34,
            noise_sigma=(0.002, 0.038),
            jpeg_quality=(28, 88),
            p_tint=0.40,
            tint_strength=(0.25, 1.0),
            p_crease=0.30,
            crease_strength=(0.05, 0.26),
            p_show_through=0.22,
            p_moire=0.12,
            p_occlusion=0.0,
        )

    @classmethod
    def hard(cls) -> "DegradationConfig":
        return cls(
            name="hard",
            quad_margin=(0.01, 0.22),
            quad_jitter=0.23,
            quad_rotation_deg=(-45.0, 45.0),
            page_scale=(0.55, 0.99),
            downscale_factor=(2.0, 4.0),
            brightness=(-0.30, 0.26),
            contrast=(0.55, 1.40),
            gamma=(0.6, 1.6),
            color_cast_r=(0.82, 1.18),
            color_cast_b=(0.80, 1.20),
            illumination_strength=(0.25, 0.65),
            p_soft_shadow=0.9,
            soft_shadow_count=(1, 4),
            soft_shadow_strength=(0.25, 0.70),
            p_glare=0.5,
            glare_strength=(0.15, 0.55),
            p_vignette=0.7,
            vignette_strength=(0.15, 0.45),
            ae_target_mean=(0.38, 0.80),
            ae_strength=(0.35, 0.95),
            gaussian_sigma=(0.6, 2.6),
            p_motion_blur=0.5,
            motion_len=(0.004, 0.020),
            p_defocus=0.25,
            p_chromatic=0.5,
            noise_sigma=(0.006, 0.05),
            jpeg_quality=(22, 65),
            p_page_curl=0.45,
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class DegradationTrace:
    ops: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    stages: list[tuple[str, Array]] = field(default_factory=list)

    def add(self, name: str, **params: Any) -> None:
        self.ops.append((name, params))

    @property
    def names(self) -> list[str]:
        return [n for n, _ in self.ops]

    def summary(self) -> str:
        parts = []
        for name, params in self.ops:
            if params:
                inner = ", ".join(
                    f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in params.items()
                )
                parts.append(f"{name}({inner})")
            else:
                parts.append(f"{name}()")
        return "\n".join(parts)


def random_document_quad(
    canvas_w: int,
    canvas_h: int,
    doc_aspect: float,
    rng: RNG,
    cfg: DegradationConfig,
) -> Array:
    scale = _u(rng, *cfg.page_scale)
    box_w = canvas_w * scale
    box_h = canvas_h * scale
    if box_w / box_h > doc_aspect:
        box_w = box_h * doc_aspect
    else:
        box_h = box_w / doc_aspect

    cx = canvas_w * 0.5 + _u(rng, -1, 1) * (canvas_w - box_w) * 0.35
    cy = canvas_h * 0.5 + _u(rng, -1, 1) * (canvas_h - box_h) * 0.35

    hw, hh = box_w * 0.5, box_h * 0.5
    rect = np.array(
        [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]], dtype=np.float32
    )

    theta = np.radians(_u(rng, *cfg.quad_rotation_deg))
    rot = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float32,
    )
    quad = rect @ rot.T + np.array([cx, cy], dtype=np.float32)

    jitter = cfg.quad_jitter * np.array([box_w, box_h], dtype=np.float32)
    quad = quad + rng.uniform(-1.0, 1.0, size=(4, 2)).astype(np.float32) * jitter

    margin = _u(rng, *cfg.quad_margin)
    lo = np.array([-canvas_w * margin, -canvas_h * margin], dtype=np.float32)
    hi = np.array(
        [canvas_w * (1 + margin), canvas_h * (1 + margin)], dtype=np.float32
    )
    quad = np.clip(quad, lo, hi)
    return order_corners(quad)


def _page_curl_map(h: int, w: int, rng: RNG, strength: float) -> tuple[Array, Array]:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    phase = _u(rng, 0.0, 2 * np.pi)
    freq = _u(rng, 0.7, 1.8)
    axis = rng.random() < 0.5
    amp = strength * (h if axis else w)
    if axis:
        dy = amp * np.sin(freq * np.pi * xx / max(w - 1, 1) + phase)
        dx = np.zeros_like(dy)
    else:
        dx = amp * np.sin(freq * np.pi * yy / max(h - 1, 1) + phase)
        dy = np.zeros_like(dx)
    return (xx + dx).astype(np.float32), (yy + dy).astype(np.float32)


def warp_onto_background(
    document: Array,
    background: Array,
    quad: Array,
    rng: RNG,
    cfg: DegradationConfig,
    trace: DegradationTrace | None = None,
) -> tuple[Array, Array]:
    bh, bw = background.shape[:2]
    dh, dw = document.shape[:2]
    src = np.array([[0, 0], [dw - 1, 0], [dw - 1, dh - 1], [0, dh - 1]], np.float32)
    h_mat = homography(src, quad)

    doc = to_float(document)
    bg = to_float(background)

    if cfg.p_page_curl > 0 and rng.random() < cfg.p_page_curl:
        s = _u(rng, *cfg.page_curl_strength)
        mx, my = _page_curl_map(dh, dw, rng, s)
        doc = cv2.remap(doc, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        if trace is not None:
            trace.add("page_curl", strength=s)

    if cfg.p_paper_texture > 0 and rng.random() < cfg.p_paper_texture:
        s = _u(rng, *cfg.paper_texture_strength)
        grain = rng.standard_normal((dh, dw), dtype=np.float32)
        grain *= s
        grain = cv2.GaussianBlur(grain, (0, 0), 0.8)[..., None]
        doc = _clip01(doc + grain)
        if trace is not None:
            trace.add("paper_texture", strength=s)

    warped = cv2.warpPerspective(
        doc, h_mat, (bw, bh), flags=cv2.INTER_AREA if dw > bw else cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )
    mask = cv2.warpPerspective(
        np.ones((dh, dw), np.float32), h_mat, (bw, bh),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    photo = bg
    if cfg.p_drop_shadow > 0 and rng.random() < cfg.p_drop_shadow:
        strength = _u(rng, *cfg.drop_shadow_strength)
        off = _u(rng, *cfg.drop_shadow_offset) * max(bw, bh)
        ang = _u(rng, 0, 2 * np.pi)

        lh, lw = _lowres_shape(bh, bw, _MASK_DIV)
        sx, sy = lw / bw, lh / bh
        shifted = quad * np.array([sx, sy], np.float32) + np.array(
            [off * np.cos(ang) * sx, off * np.sin(ang) * sy], np.float32
        )
        low = np.zeros((lh, lw), np.float32)
        cv2.fillConvexPoly(low, shifted.astype(np.int32), 1.0)
        blur_px = _odd(max(3, _u(rng, *cfg.drop_shadow_blur) * max(lw, lh)))
        low = cv2.GaussianBlur(low, (blur_px, blur_px), 0)
        shadow = cv2.resize(low, (bw, bh), interpolation=cv2.INTER_LINEAR)
        photo = photo * (1.0 - strength * shadow[..., None])
        if trace is not None:
            trace.add("page_drop_shadow", strength=strength, blur_px=int(blur_px * _MASK_DIV))

    alpha = cv2.GaussianBlur(mask, (3, 3), 0)[..., None]
    photo = photo * (1.0 - alpha) + warped * alpha
    if trace is not None:
        trace.add("warp_onto_background", quad=np.round(quad, 1).tolist())
    return np.clip(photo, 0.0, 1.0), h_mat


def downscale_upscale(img: Array, rng: RNG, cfg: DegradationConfig,
                      trace: DegradationTrace | None = None) -> Array:
    f = _u(rng, *cfg.downscale_factor)
    h, w = img.shape[:2]
    sh, sw = max(8, int(h / f)), max(8, int(w / f))
    down = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    up_interp = rng.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_NEAREST],
                           p=[0.5, 0.4, 0.1])
    out = cv2.resize(down, (w, h), interpolation=int(up_interp))
    if trace is not None:
        trace.add("downscale_upscale", factor=f)
    return out


def photometric(img: Array, rng: RNG, cfg: DegradationConfig,
                trace: DegradationTrace | None = None) -> Array:
    b = _u(rng, *cfg.brightness)
    c = _u(rng, *cfg.contrast)
    g = _u(rng, *cfg.gamma)
    cr = _u(rng, *cfg.color_cast_r)
    cb = _u(rng, *cfg.color_cast_b)
    sat = _u(rng, *cfg.saturation)

    out = cv2.addWeighted(img, c, img, 0.0, 0.5 * (1.0 - c) + b)
    _clip01(out)
    if abs(g - 1.0) > 1e-3:
        out = cv2.pow(out, g)

    luma = np.array([0.299, 0.587, 0.114], np.float32)
    m_sat = sat * np.eye(3, dtype=np.float32) + (1.0 - sat) * np.tile(luma, (3, 1))
    m = np.diag(np.array([cr, 1.0, cb], np.float32)) @ m_sat
    out = cv2.transform(out, m)
    _clip01(out)
    if trace is not None:
        trace.add("photometric", brightness=b, contrast=c, gamma=g,
                  cast_r=cr, cast_b=cb, saturation=sat)
    return out


def illumination_field(img: Array, rng: RNG, cfg: DegradationConfig,
                       trace: DegradationTrace | None = None) -> Array:
    h, w = img.shape[:2]
    s = _u(rng, *cfg.illumination_strength)
    ang = _u(rng, 0, 2 * np.pi)
    lh, lw = _lowres_shape(h, w, _LOWRES_DIV)
    xn, yn = _norm_grid(lh, lw)
    ramp = float(np.cos(ang)) * xn + float(np.sin(ang)) * yn
    ramp = (ramp - ramp.min()) / max(float(ramp.max() - ramp.min()), 1e-6)
    blob = _smooth_field(rng, lh, lw, ctrl=int(rng.integers(3, 6)))
    mix = _u(rng, 0.35, 0.85)
    fieldv = mix * ramp + (1.0 - mix) * blob
    gain = (1.0 - s) + s * fieldv
    gain = cv2.resize(gain, (w, h), interpolation=cv2.INTER_CUBIC)
    out = _clip01(img * gain[..., None])
    if trace is not None:
        trace.add("illumination_field", strength=s, angle_deg=float(np.degrees(ang)))
    return out


def soft_shadow(img: Array, rng: RNG, cfg: DegradationConfig,
                trace: DegradationTrace | None = None) -> Array:
    h, w = img.shape[:2]
    n = int(rng.integers(cfg.soft_shadow_count[0], cfg.soft_shadow_count[1] + 1))
    lh, lw = _lowres_shape(h, w, _MASK_DIV)
    acc = np.zeros((lh, lw), np.float32)
    for _ in range(n):
        kind = rng.random()
        m = np.zeros((lh, lw), np.float32)
        if kind < 0.45:
            ang = _u(rng, 0, 2 * np.pi)
            xn, yn = _norm_grid(lh, lw)
            d = float(np.cos(ang)) * xn + float(np.sin(ang)) * yn
            thr = _u(rng, 0.25, 0.75)
            m = (d < thr).astype(np.float32)
        elif kind < 0.85:
            k = int(rng.integers(3, 8))
            cx, cy = _u(rng, 0, lw), _u(rng, 0, lh)
            rad = _u(rng, 0.15, 0.75) * max(lw, lh)
            ang0 = _u(rng, 0, 2 * np.pi)
            pts = []
            for i in range(k):
                a = ang0 + 2 * np.pi * i / k + _u(rng, -0.3, 0.3)
                r = rad * _u(rng, 0.45, 1.15)
                pts.append([cx + r * np.cos(a), cy + r * np.sin(a)])
            cv2.fillPoly(m, [np.array(pts, np.int32)], 1.0)
        else:
            cx, cy = _u(rng, 0, lw), _u(rng, 0, lh)
            ln = _u(rng, 0.3, 1.2) * max(lw, lh)
            th = _u(rng, 0.02, 0.16) * max(lw, lh)
            ang = _u(rng, 0, 180)
            box = ((cx, cy), (ln, th), ang)
            cv2.fillPoly(m, [cv2.boxPoints(box).astype(np.int32)], 1.0)

        blur = _odd(max(3, _u(rng, *cfg.soft_shadow_blur) * max(lw, lh)))
        m = cv2.GaussianBlur(m, (blur, blur), 0)
        acc = np.maximum(acc, m * _u(rng, *cfg.soft_shadow_strength))

    acc = cv2.resize(acc, (w, h), interpolation=cv2.INTER_CUBIC)
    out = _clip01(img * (1.0 - acc[..., None]))
    if trace is not None:
        trace.add("soft_shadow", count=n, max_strength=float(acc.max()))
    return out


def specular_glare(img: Array, rng: RNG, cfg: DegradationConfig,
                   trace: DegradationTrace | None = None) -> Array:
    h, w = img.shape[:2]
    s = _u(rng, *cfg.glare_strength)
    lh, lw = _lowres_shape(h, w, _MASK_DIV)
    m = np.zeros((lh, lw), np.float32)
    cx, cy = int(_u(rng, 0, lw)), int(_u(rng, 0, lh))
    ax = int(_u(rng, 0.08, 0.42) * lw)
    ay = int(_u(rng, 0.08, 0.42) * lh)
    cv2.ellipse(m, (cx, cy), (max(ax, 3), max(ay, 3)), _u(rng, 0, 180), 0, 360, 1.0, -1)
    blur = _odd(max(9, 0.09 * max(lw, lh)))
    m = cv2.GaussianBlur(m, (blur, blur), 0) * s
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_CUBIC)
    out = _clip01(img + m[..., None] * (1.0 - img))
    if trace is not None:
        trace.add("specular_glare", strength=s)
    return out


def vignette(img: Array, rng: RNG, cfg: DegradationConfig,
             trace: DegradationTrace | None = None) -> Array:
    h, w = img.shape[:2]
    s = _u(rng, *cfg.vignette_strength)
    lh, lw = _lowres_shape(h, w, _LOWRES_DIV)
    xn, yn = _norm_grid(lh, lw)
    r = np.sqrt((2.0 * xn - 1.0) ** 2 + (2.0 * yn - 1.0) ** 2) / np.sqrt(2.0)
    gain = 1.0 - s * np.power(r, _u(rng, 1.5, 3.0))
    gain = cv2.resize(gain, (w, h), interpolation=cv2.INTER_CUBIC)
    out = _clip01(img * gain[..., None])
    if trace is not None:
        trace.add("vignette", strength=s)
    return out


def auto_exposure(img: Array, rng: RNG, cfg: DegradationConfig,
                  trace: DegradationTrace | None = None) -> Array:
    target = _u(rng, *cfg.ae_target_mean)
    lum = float(np.dot(img.reshape(-1, 3).mean(axis=0), [0.299, 0.587, 0.114]))
    gain = float(np.clip(target / max(lum, 1e-3), *cfg.ae_gain_range))
    alpha = _u(rng, *cfg.ae_strength)
    g = 1.0 + alpha * (gain - 1.0)
    out = _clip01(img * g)
    if trace is not None:
        trace.add("auto_exposure", scene_mean=lum, gain=g)
    return out


def ambient_tint(img: Array, rng: RNG, cfg: DegradationConfig,
                 trace: DegradationTrace | None = None) -> Array:
    h, w = img.shape[:2]
    strength = _u(rng, *cfg.tint_strength)

    def _light() -> Array:
        warm = _u(rng, -1.0, 1.0)
        gain = np.array([1.0 + 0.34 * warm, 1.0, 1.0 - 0.30 * warm], np.float32)
        gain *= np.array([_u(rng, 0.88, 1.14), _u(rng, 0.93, 1.07),
                          _u(rng, 0.88, 1.14)], np.float32)
        return gain / float(gain.mean())

    g0, g1 = _light(), _light()
    ramp = _smooth_field(rng, *_lowres_shape(h, w, _MASK_DIV), ctrl=2)
    ramp = cv2.resize(ramp, (w, h), interpolation=cv2.INTER_CUBIC)[..., None]
    field = g0[None, None, :] * (1.0 - ramp) + g1[None, None, :] * ramp
    field = 1.0 + strength * (field - 1.0)
    out = _clip01(img * field)
    if trace is not None:
        trace.add("ambient_tint", strength=strength,
                  warm=float(g0[0] - g0[2]), cool=float(g1[0] - g1[2]))
    return out


def lens_distortion(img: Array, rng: RNG, cfg: DegradationConfig,
                    trace: DegradationTrace | None = None) -> Array:
    h, w = img.shape[:2]
    k1 = _u(rng, *cfg.lens_k1)
    if abs(k1) < 1e-4:
        return img
    xs, ys = _norm_grid(h, w)
    r2 = xs * xs + ys * ys
    scale = 1.0 + k1 * r2
    cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
    map_x = (xs * scale * cx + cx).astype(np.float32)
    map_y = (ys * scale * cy + cy).astype(np.float32)
    out = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101)
    if trace is not None:
        trace.add("lens_distortion", k1=k1)
    return out


def occlusion(img: Array, rng: RNG, cfg: DegradationConfig,
              trace: DegradationTrace | None = None) -> Array:
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.float32)
    for _ in range(int(rng.integers(1, 3))):
        edge = int(rng.integers(0, 4))
        length = _u(rng, 0.14, 0.40) * (h if edge % 2 == 0 else w)
        width = _u(rng, 0.05, 0.13) * min(h, w)
        t = _u(rng, 0.1, 0.9)
        if edge == 0:      p0, p1 = (t * w, 0.0), (t * w + _u(rng, -.2, .2) * w, length)
        elif edge == 1:    p0, p1 = (float(w), t * h), (w - length, t * h + _u(rng, -.2, .2) * h)
        elif edge == 2:    p0, p1 = (t * w, float(h)), (t * w + _u(rng, -.2, .2) * w, h - length)
        else:              p0, p1 = (0.0, t * h), (length, t * h + _u(rng, -.2, .2) * h)
        cv2.line(mask, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])),
                 1.0, int(max(3, width)), cv2.LINE_AA)
        cv2.circle(mask, (int(p1[0]), int(p1[1])), int(max(3, width * 0.55)), 1.0, -1)
    blur = _odd(max(5, 0.012 * max(h, w)))
    mask = cv2.GaussianBlur(mask, (blur, blur), 0)[..., None]

    if rng.random() < 0.65:
        skin = np.array([_u(rng, 0.45, 0.86), _u(rng, 0.30, 0.66), _u(rng, 0.24, 0.56)],
                        np.float32)
        patch = np.broadcast_to(skin, img.shape).astype(np.float32)
        patch = _clip01(patch * _u(rng, 0.75, 1.15))
        kind = "finger"
    else:
        patch = _clip01(img * _u(rng, 0.30, 0.62))
        kind = "shadow"
    out = _clip01(img * (1.0 - mask) + patch * mask)
    if trace is not None:
        trace.add("occlusion", kind=kind)
    return out


def fold_crease(img: Array, rng: RNG, cfg: DegradationConfig,
                trace: DegradationTrace | None = None) -> Array:
    h, w = img.shape[:2]
    strength = _u(rng, *cfg.crease_strength)
    xs, ys = _norm_grid(h, w)
    theta = _u(rng, 0.0, np.pi)
    d = xs * np.cos(theta) + ys * np.sin(theta) - _u(rng, -0.55, 0.55)
    width = _u(rng, 0.02, 0.09)
    ridge = np.exp(-0.5 * (d / width) ** 2) * np.sign(d)
    field = 1.0 + strength * ridge.astype(np.float32)
    out = _clip01(img * field[..., None])
    if trace is not None:
        trace.add("fold_crease", strength=strength)
    return out


def show_through(img: Array, rng: RNG, cfg: DegradationConfig,
                 trace: DegradationTrace | None = None) -> Array:
    s = _u(rng, *cfg.show_through_strength)
    back = cv2.flip(img, 1)
    k = _odd(max(3, 0.006 * max(img.shape[:2])))
    back = cv2.GaussianBlur(back, (k, k), 0)
    out = _clip01(img * (1.0 - s) + np.minimum(img, back) * s)
    if trace is not None:
        trace.add("show_through", strength=s)
    return out


def moire(img: Array, rng: RNG, cfg: DegradationConfig,
          trace: DegradationTrace | None = None) -> Array:
    h, w = img.shape[:2]
    s = _u(rng, *cfg.moire_strength)
    xs, ys = _norm_grid(h, w)
    theta = _u(rng, 0.0, np.pi)
    freq = _u(rng, 40.0, 190.0)
    phase = xs * np.cos(theta) + ys * np.sin(theta)
    pattern = np.sin(phase * freq) * np.sin(phase * freq * _u(rng, 1.02, 1.18))
    out = _clip01(img * (1.0 + s * pattern.astype(np.float32))[..., None])
    if trace is not None:
        trace.add("moire", strength=s, frequency=freq)
    return out


def chromatic_aberration(img: Array, rng: RNG, cfg: DegradationConfig,
                         trace: DegradationTrace | None = None) -> Array:
    h, w = img.shape[:2]
    s = _u(rng, *cfg.chromatic_shift)
    out = img.copy()
    for ch, sign in ((0, 1.0), (2, -1.0)):
        k = 1.0 + sign * s
        m = np.float32([[k, 0, (1 - k) * w * 0.5], [0, k, (1 - k) * h * 0.5]])
        out[..., ch] = cv2.warpAffine(img[..., ch], m, (w, h),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT)
    if trace is not None:
        trace.add("chromatic_aberration", shift=s)
    return out


def _motion_kernel(length: int, angle_deg: float) -> Array:
    length = max(3, _odd(length))
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0
    m = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle_deg, 1.0)
    k = cv2.warpAffine(k, m, (length, length))
    s = k.sum()
    return k / s if s > 1e-6 else k


def _disk_kernel(radius: int) -> Array:
    radius = max(1, int(radius))
    size = 2 * radius + 1
    k = np.zeros((size, size), np.float32)
    cv2.circle(k, (radius, radius), radius, 1.0, -1)
    return k / max(k.sum(), 1e-6)


def blur(img: Array, rng: RNG, cfg: DegradationConfig,
         trace: DegradationTrace | None = None) -> Array:
    h, w = img.shape[:2]
    out = img
    if rng.random() < cfg.p_motion_blur:
        ln = int(_u(rng, *cfg.motion_len) * max(w, h))
        ang = _u(rng, 0, 180)
        out = cv2.filter2D(out, -1, _motion_kernel(ln, ang), borderType=cv2.BORDER_REFLECT)
        if trace is not None:
            trace.add("motion_blur", length_px=max(3, _odd(ln)), angle_deg=ang)
    if rng.random() < cfg.p_defocus:
        rad = int(_u(rng, *cfg.defocus_radius) * max(w, h))
        out = cv2.filter2D(out, -1, _disk_kernel(rad), borderType=cv2.BORDER_REFLECT)
        if trace is not None:
            trace.add("defocus_blur", radius_px=max(1, rad))
    sigma = _u(rng, *cfg.gaussian_sigma)
    out = cv2.GaussianBlur(out, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    if trace is not None:
        trace.add("gaussian_blur", sigma=sigma)
    return out


def sensor_noise(img: Array, rng: RNG, cfg: DegradationConfig,
                 trace: DegradationTrace | None = None) -> Array:
    sigma = _u(rng, *cfg.noise_sigma)
    grain = _u(rng, *cfg.noise_grain)
    n = rng.standard_normal(img.shape, dtype=np.float32)
    n *= sigma
    if grain > 0.05:
        n = cv2.GaussianBlur(n, (0, 0), grain, borderType=cv2.BORDER_REFLECT)
        n *= 1.0 + grain
    out = _clip01(img + n)
    if trace is not None:
        trace.add("sensor_noise", sigma=sigma, grain=grain)
    return out


def jpeg(img: Array, rng: RNG, cfg: DegradationConfig,
         trace: DegradationTrace | None = None) -> Array:
    q = int(rng.integers(cfg.jpeg_quality[0], cfg.jpeg_quality[1] + 1))
    bgr = cv2.cvtColor(to_uint8(img), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        return img
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    out = to_float(cv2.cvtColor(dec, cv2.COLOR_BGR2RGB))
    if trace is not None:
        trace.add("jpeg", quality=q)
    return out


class DegradationPipeline:
    STAGES: Sequence[tuple[str, str | None, Callable[..., Array]]] = (
        ("show_through", "p_show_through", show_through),
        ("fold_crease", "p_crease", fold_crease),
        ("downscale_upscale", "p_downscale", downscale_upscale),
        ("photometric", None, photometric),
        ("ambient_tint", "p_tint", ambient_tint),
        ("illumination_field", "p_illumination", illumination_field),
        ("soft_shadow", "p_soft_shadow", soft_shadow),
        ("occlusion", "p_occlusion", occlusion),
        ("specular_glare", "p_glare", specular_glare),
        ("lens_distortion", "p_lens_distortion", lens_distortion),
        ("vignette", "p_vignette", vignette),
        ("auto_exposure", "p_auto_exposure", auto_exposure),
        ("chromatic_aberration", "p_chromatic", chromatic_aberration),
        ("moire", "p_moire", moire),
        ("blur", "p_blur", blur),
        ("sensor_noise", "p_noise", sensor_noise),
        ("jpeg", "p_jpeg", jpeg),
    )

    def __init__(self, cfg: DegradationConfig | None = None):
        self.cfg = cfg or DegradationConfig()

    def __call__(
        self,
        photo: Array,
        rng: RNG,
        keep_stages: bool = False,
        trace: DegradationTrace | None = None,
    ) -> tuple[Array, DegradationTrace]:
        trace = trace if trace is not None else DegradationTrace()
        out = to_float(photo)
        if keep_stages:
            trace.stages.append(("composite", to_uint8(out)))
        for name, prob_field, fn in self.STAGES:
            if prob_field is not None and rng.random() >= getattr(self.cfg, prob_field):
                continue
            before = out
            out = fn(out, rng, self.cfg, trace)
            if keep_stages and out is not before:
                trace.stages.append((name, to_uint8(out)))
        return out, trace
