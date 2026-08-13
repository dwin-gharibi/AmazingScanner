from __future__ import annotations

import cv2
import numpy as np
import torch

from docscanner.models.warp import get_perspective_transform, rectify_batch, warp_perspective

SRC = np.array([[10.0, 20.0], [180.0, 12.0], [195.0, 140.0], [5.0, 128.0]], np.float32)
DST = np.array([[0.0, 0.0], [255.0, 0.0], [255.0, 127.0], [0.0, 127.0]], np.float32)


def test_transform_matches_opencv():
    ours = get_perspective_transform(torch.from_numpy(SRC)[None].double(),
                                     torch.from_numpy(DST)[None].double())[0].numpy()
    theirs = cv2.getPerspectiveTransform(SRC, DST)
    assert np.allclose(ours / ours[2, 2], theirs / theirs[2, 2], atol=1e-6)


def test_transform_maps_the_correspondences():
    m = get_perspective_transform(torch.from_numpy(SRC)[None].double(),
                                  torch.from_numpy(DST)[None].double())
    pts = torch.cat([torch.from_numpy(SRC).double(),
                     torch.ones(4, 1, dtype=torch.float64)], dim=1)
    out = (pts @ m[0].T)
    out = out[:, :2] / out[:, 2:3]
    assert torch.allclose(out, torch.from_numpy(DST).double(), atol=1e-6)


def test_warp_matches_opencv_pixels():
    img = np.zeros((150, 200, 3), np.uint8)
    cv2.rectangle(img, (20, 20), (180, 130), (240, 240, 240), -1)
    for i in range(5):
        cv2.line(img, (35 + i * 30, 30), (35 + i * 30, 120), (20, 20, 30), 4)

    m = cv2.getPerspectiveTransform(SRC, DST)
    ref = cv2.warpPerspective(img, m, (256, 128), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)

    t = torch.from_numpy(img.transpose(2, 0, 1)).float()[None] / 255.0
    mt = torch.from_numpy(m).float()[None]
    ours = warp_perspective(t, mt, (128, 256))[0].numpy().transpose(1, 2, 0) * 255.0

    a = ours[4:-4, 4:-4].astype(np.float32)
    b = ref[4:-4, 4:-4].astype(np.float32)
    assert np.abs(a - b).mean() < 6.0, np.abs(a - b).mean()


def test_warp_is_differentiable_wrt_corners():
    img = torch.rand(1, 3, 96, 96, requires_grad=False)
    corners = torch.tensor([[[0.10, 0.10], [0.90, 0.12], [0.88, 0.90], [0.08, 0.88]]],
                           requires_grad=True)
    out = rectify_batch(img, corners, (48, 48))
    out.mean().backward()
    assert corners.grad is not None
    assert torch.isfinite(corners.grad).all()
    assert float(corners.grad.abs().sum()) > 0


def test_rectify_batch_recovers_a_known_page():
    page = np.zeros((120, 90, 3), np.uint8)
    page[:] = 240
    for i in range(6):
        cv2.line(page, (10, 15 + i * 18), (80, 15 + i * 18), (25, 25, 35), 3)

    quad = np.array([[30, 25], [210, 40], [200, 180], [20, 160]], np.float32)
    src_corners = np.array([[0, 0], [89, 0], [89, 119], [0, 119]], np.float32)
    photo = cv2.warpPerspective(page, cv2.getPerspectiveTransform(src_corners, quad),
                                (240, 200), borderMode=cv2.BORDER_REPLICATE)

    t = torch.from_numpy(photo.transpose(2, 0, 1)).float()[None] / 255.0
    norm = torch.from_numpy(quad / np.array([239.0, 199.0], np.float32)).float()[None]
    back = rectify_batch(t, norm, (120, 90))[0].numpy().transpose(1, 2, 0) * 255.0

    a = back[8:-8, 8:-8].astype(np.float32)
    b = page[8:-8, 8:-8].astype(np.float32)
    assert np.abs(a - b).mean() < 18.0, np.abs(a - b).mean()


def test_warp_handles_degenerate_corners_without_nan():
    img = torch.rand(1, 3, 64, 64)
    corners = torch.tensor([[[0.5, 0.5], [0.5001, 0.5], [0.5001, 0.5001], [0.5, 0.5001]]])
    out = rectify_batch(img, corners, (32, 32))
    assert torch.isfinite(out).all()
