from __future__ import annotations

import cv2
import numpy as np
import pytest

from docscanner.utils.geometry import (
    corner_distance,
    denormalize_corners,
    estimate_page_size,
    homography,
    normalize_corners,
    order_corners,
    quad_area,
    quad_iou,
    quad_is_convex,
    rectify,
    warp_points,
)

RECT = np.array([[10, 20], [110, 20], [110, 90], [10, 90]], np.float32)


def test_order_corners_is_idempotent_and_canonical():
    assert np.allclose(order_corners(RECT), RECT)
    for k in range(4):
        rolled = np.roll(RECT, k, axis=0)
        assert np.allclose(order_corners(rolled), RECT), f"roll {k}"


def test_order_corners_handles_reversed_winding():
    reversed_quad = RECT[::-1].copy()
    assert np.allclose(order_corners(reversed_quad), RECT)


@pytest.mark.parametrize("angle", [0, 10, 25, 40, 60, 80, 100, 170, 200, 280, 350])
def test_order_corners_is_clockwise_for_any_rotation(angle):
    c = RECT.mean(axis=0)
    th = np.radians(angle)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], np.float32)
    rotated = (RECT - c) @ rot.T + c
    out = order_corners(rotated)
    x, y = out[:, 0], out[:, 1]
    area = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    assert area > 0
    for p in rotated:
        assert np.min(np.linalg.norm(out - p, axis=1)) < 1e-3


def test_quad_area_and_convexity():
    assert quad_area(RECT) == pytest.approx(100 * 70)
    assert quad_is_convex(RECT)
    bowtie = np.array([[0, 0], [10, 10], [10, 0], [0, 10]], np.float32)
    assert not quad_is_convex(bowtie)


def test_homography_round_trip():
    dst = np.array([[0, 0], [255, 0], [255, 127], [0, 127]], np.float32)
    h = homography(RECT, dst)
    assert np.allclose(warp_points(RECT, h), dst, atol=1e-3)
    h_inv = np.linalg.inv(h)
    assert np.allclose(warp_points(dst, h_inv), RECT, atol=1e-3)


def test_rectify_recovers_a_known_warp():
    src = np.zeros((120, 180, 3), np.uint8)
    cv2.rectangle(src, (10, 10), (170, 110), (255, 255, 255), -1)
    for i in range(6):
        cv2.line(src, (20 + i * 25, 20), (20 + i * 25, 100), (0, 0, 0), 3)

    quad = np.array([[40, 30], [300, 15], [320, 210], [25, 190]], np.float32)
    src_corners = np.array([[0, 0], [179, 0], [179, 119], [0, 119]], np.float32)
    h = homography(src_corners, quad)
    photo = cv2.warpPerspective(src, h, (360, 240))

    back, _ = rectify(photo, quad, out_size=(180, 120))
    a = back[15:-15, 15:-15].astype(np.float32)
    b = src[15:-15, 15:-15].astype(np.float32)
    assert np.abs(a - b).mean() < 12.0


def test_estimate_page_size_recovers_aspect_under_perspective():
    quad = np.array([[120, 90], [520, 60], [560, 250], [95, 235]], np.float32)
    est_w, est_h = estimate_page_size(quad, (400, 700))
    aspect = est_w / est_h
    assert 1.3 < aspect < 3.2, aspect


def test_normalize_denormalize_round_trip():
    n = normalize_corners(RECT, 200, 100)
    assert n.max() <= 1.0
    assert np.allclose(denormalize_corners(n, 200, 100), RECT, atol=1e-4)


def test_corner_distance_and_iou():
    assert corner_distance(RECT, RECT).max() == pytest.approx(0.0)
    shifted = RECT + np.array([3.0, 4.0])
    assert corner_distance(shifted, RECT) == pytest.approx([5.0] * 4)
    assert quad_iou(RECT, RECT) > 0.99
    assert quad_iou(RECT, RECT + 1000) == pytest.approx(0.0)
