from __future__ import annotations

import cv2
import numpy as np

__all__ = [
    "order_corners",
    "quad_is_convex",
    "quad_is_plausible",
    "quad_area",
    "quad_side_lengths",
    "estimate_page_size",
    "homography",
    "warp_points",
    "rectify",
    "corner_distance",
    "normalize_corners",
    "denormalize_corners",
    "expand_quad",
    "quad_iou",
]


def order_corners(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    centroid = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    order = np.argsort(angles)
    cyc = pts[order]

    ref = np.array([pts[:, 0].min(), pts[:, 1].min()])
    start = int(np.argmin(((cyc - ref) ** 2).sum(axis=1)))
    cyc = np.roll(cyc, -start, axis=0)

    if _signed_area(cyc) < 0:
        cyc = np.vstack([cyc[0], cyc[1:][::-1]])
    return cyc.astype(np.float32)


def _signed_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def quad_area(pts: np.ndarray) -> float:
    return abs(_signed_area(np.asarray(pts, dtype=np.float64).reshape(4, 2)))


def quad_is_convex(pts: np.ndarray, min_angle_deg: float = 20.0) -> bool:
    pts = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    signs = []
    for i in range(4):
        a, b, c = pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4]
        v1, v2 = b - a, c - b
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        signs.append(np.sign(cross))
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return False
        cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        interior = 180.0 - np.degrees(np.arccos(cosang))
        if interior < min_angle_deg or interior > 180.0 - min_angle_deg * 0.25:
            return False
    return len(set(signs)) == 1


def quad_is_plausible(
    pts: np.ndarray,
    shape: tuple[int, int],
    min_area_frac: float = 0.02,
    margin_frac: float = 0.25,
) -> bool:
    pts = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    if not np.isfinite(pts).all():
        return False
    h, w = int(shape[0]), int(shape[1])
    if h <= 0 or w <= 0:
        return False
    if not quad_is_convex(pts):
        return False
    if quad_area(pts) < min_area_frac * w * h:
        return False
    mx, my = margin_frac * w, margin_frac * h
    within_x = (pts[:, 0] >= -mx).all() and (pts[:, 0] <= w + mx).all()
    within_y = (pts[:, 1] >= -my).all() and (pts[:, 1] <= h + my).all()
    return bool(within_x and within_y)


def quad_side_lengths(pts: np.ndarray) -> tuple[float, float, float, float]:
    pts = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    return tuple(float(np.linalg.norm(pts[(i + 1) % 4] - pts[i])) for i in range(4))


USE_PROJECTIVE_ASPECT = False


def estimate_page_size(
    quad: np.ndarray,
    image_shape: tuple[int, int] | None = None,
    max_aspect: float = 3.0,
    projective: bool | None = None,
) -> tuple[int, int]:
    quad = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    top, right, bottom, left = quad_side_lengths(quad)
    w = max((top + bottom) * 0.5, 1.0)
    h = max((left + right) * 0.5, 1.0)

    if projective if projective is not None else USE_PROJECTIVE_ASPECT:
        aspect = _projective_aspect(quad, image_shape)
        if aspect is not None and 1.0 / max_aspect <= aspect <= max_aspect:
            diag = float(np.hypot(w, h))
            h_new = diag / np.hypot(aspect, 1.0)
            w_new = aspect * h_new
            w, h = w_new, h_new

    return int(round(max(w, 8.0))), int(round(max(h, 8.0)))


def _projective_aspect(quad: np.ndarray, image_shape: tuple[int, int] | None) -> float | None:
    if image_shape is None:
        return None
    h_img, w_img = image_shape[:2]
    u0, v0 = w_img / 2.0, h_img / 2.0

    m = np.hstack([quad - np.array([u0, v0]), np.ones((4, 1))])
    tl, tr, br, bl = m

    def _coeffs(target, basis1, basis2, basis3):
        a = np.stack([basis1, basis2, basis3], axis=1)
        try:
            return np.linalg.solve(a, target)
        except np.linalg.LinAlgError:
            return None

    c_bl = _coeffs(bl, tl, tr, br)
    c_tr = _coeffs(tr, tl, bl, br)
    if c_bl is None or c_tr is None:
        return None
    k2 = c_bl[2] / c_bl[0] if abs(c_bl[0]) > 1e-9 else None
    k3 = c_tr[2] / c_tr[0] if abs(c_tr[0]) > 1e-9 else None
    if k2 is None or k3 is None:
        return None

    if abs(k2 - 1.0) < 1e-3 or abs(k3 - 1.0) < 1e-3:
        return None

    n2 = k2 * tr - tl
    n3 = k3 * bl - tl

    n21, n22, n23 = n2
    n31, n32, n33 = n3
    denom = n23 * n33
    if abs(denom) < 1e-9:
        return None
    f2 = -(1.0 / denom) * ((n21 * n31 - (n21 * n33 + n23 * n31) * u0 + n23 * n33 * u0 * u0)
                           + (n22 * n32 - (n22 * n33 + n23 * n32) * v0 + n23 * n33 * v0 * v0))
    if not np.isfinite(f2) or f2 <= 1e-6:
        return None
    f = np.sqrt(f2)
    a_mat = np.array([[f, 0, u0], [0, f, v0], [0, 0, 1.0]])
    try:
        ai = np.linalg.inv(a_mat)
    except np.linalg.LinAlgError:
        return None
    num = float(n2 @ ai.T @ ai @ n2)
    den = float(n3 @ ai.T @ ai @ n3)
    if den <= 1e-9 or num <= 1e-9:
        return None
    return float(np.sqrt(num / den))


def homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src = np.asarray(src, dtype=np.float32).reshape(4, 2)
    dst = np.asarray(dst, dtype=np.float32).reshape(4, 2)
    return cv2.getPerspectiveTransform(src, dst)


def warp_points(pts: np.ndarray, h_mat: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, np.asarray(h_mat, dtype=np.float32))
    return out.reshape(-1, 2).astype(np.float32)


def rectify(
    image: np.ndarray,
    quad: np.ndarray,
    out_size: tuple[int, int] | None = None,
    max_side: int | None = None,
    border_value: int = 255,
) -> tuple[np.ndarray, np.ndarray]:
    quad = order_corners(quad)
    if out_size is None:
        w, h = estimate_page_size(quad, image.shape[:2])
    else:
        w, h = out_size
    if max_side is not None and max(w, h) > max_side:
        s = max_side / float(max(w, h))
        w, h = max(int(round(w * s)), 8), max(int(round(h * s)), 8)

    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    h_mat = homography(quad, dst)
    interp = cv2.INTER_AREA if quad_area(quad) > w * h * 1.2 else cv2.INTER_CUBIC
    out = cv2.warpPerspective(
        image, h_mat, (w, h), flags=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(border_value,) * (image.shape[2] if image.ndim == 3 else 1),
    )
    return out, h_mat


def corner_distance(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64).reshape(4, 2)
    gt = np.asarray(gt, dtype=np.float64).reshape(4, 2)
    return np.linalg.norm(pred - gt, axis=1)


def normalize_corners(corners: np.ndarray, width: int, height: int) -> np.ndarray:
    out = np.asarray(corners, dtype=np.float32).reshape(-1, 2).copy()
    out[:, 0] /= float(width)
    out[:, 1] /= float(height)
    return out


def denormalize_corners(corners: np.ndarray, width: int, height: int) -> np.ndarray:
    out = np.asarray(corners, dtype=np.float32).reshape(-1, 2).copy()
    out[:, 0] *= float(width)
    out[:, 1] *= float(height)
    return out


def expand_quad(quad: np.ndarray, ratio: float = 0.02) -> np.ndarray:
    quad = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    c = quad.mean(axis=0, keepdims=True)
    return (c + (quad - c) * (1.0 + ratio)).astype(np.float32)


def quad_iou(a: np.ndarray, b: np.ndarray, canvas: int = 512) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(4, 2)
    b = np.asarray(b, dtype=np.float32).reshape(4, 2)
    pts = np.vstack([a, b])
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    span = float(max(hi[0] - lo[0], hi[1] - lo[1], 1e-6))
    scale = (canvas - 2) / span

    def _mask(q):
        m = np.zeros((canvas, canvas), dtype=np.uint8)
        pp = ((q - lo) * scale + 1).astype(np.int32)
        cv2.fillConvexPoly(m, pp, 1)
        return m

    ma, mb = _mask(a), _mask(b)
    inter = float(np.count_nonzero(ma & mb))
    union = float(np.count_nonzero(ma | mb))
    return inter / union if union > 0 else 0.0
