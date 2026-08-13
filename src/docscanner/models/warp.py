from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["get_perspective_transform", "warp_perspective", "normalize_homography",
           "rectify_batch"]


def get_perspective_transform(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    if src.shape[-2:] != (4, 2) or dst.shape[-2:] != (4, 2):
        raise ValueError("src and dst must be (B, 4, 2)")
    b = src.shape[0]
    device, dtype = src.device, src.dtype

    x, y = src[..., 0], src[..., 1]
    u, v = dst[..., 0], dst[..., 1]
    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)

    row_u = torch.stack([x, y, ones, zeros, zeros, zeros, -u * x, -u * y], dim=-1)
    row_v = torch.stack([zeros, zeros, zeros, x, y, ones, -v * x, -v * y], dim=-1)

    a = torch.cat([row_u, row_v], dim=1)
    rhs = torch.cat([u, v], dim=1).unsqueeze(-1)

    h = torch.linalg.solve(a, rhs).squeeze(-1)
    h = torch.cat([h, torch.ones(b, 1, device=device, dtype=dtype)], dim=1)
    return h.view(b, 3, 3)


def normalize_homography(h: torch.Tensor, src_hw: tuple[int, int],
                         dst_hw: tuple[int, int]) -> torch.Tensor:
    sh, sw = src_hw
    dh, dw = dst_hw
    device, dtype = h.device, h.dtype

    def denorm(w: int, height: int) -> torch.Tensor:
        return torch.tensor([[(w - 1) / 2, 0, (w - 1) / 2],
                             [0, (height - 1) / 2, (height - 1) / 2],
                             [0, 0, 1]], device=device, dtype=dtype)

    def norm(w: int, height: int) -> torch.Tensor:
        return torch.tensor([[2 / max(w - 1, 1), 0, -1],
                             [0, 2 / max(height - 1, 1), -1],
                             [0, 0, 1]], device=device, dtype=dtype)

    return norm(dw, dh) @ h @ denorm(sw, sh)


def warp_perspective(image: torch.Tensor, h: torch.Tensor, out_hw: tuple[int, int],
                     mode: str = "bilinear", padding_mode: str = "border",
                     align_corners: bool = True) -> torch.Tensor:
    b, _, sh, sw = image.shape
    dh, dw = out_hw
    h_norm = normalize_homography(h, (sh, sw), (dh, dw))
    h_inv = torch.inverse(h_norm)

    ys = torch.linspace(-1, 1, dh, device=image.device, dtype=image.dtype)
    xs = torch.linspace(-1, 1, dw, device=image.device, dtype=image.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    ones = torch.ones_like(grid_x)
    grid = torch.stack([grid_x, grid_y, ones], dim=-1).view(1, -1, 3)
    grid = grid.expand(b, -1, -1)

    warped = grid @ h_inv.transpose(1, 2)
    denom = warped[..., 2:3]
    denom = torch.where(denom.abs() < 1e-8, torch.full_like(denom, 1e-8), denom)
    sample = (warped[..., :2] / denom).view(b, dh, dw, 2)

    return F.grid_sample(image, sample, mode=mode, padding_mode=padding_mode,
                         align_corners=align_corners)


def rectify_batch(images: torch.Tensor, corners_norm: torch.Tensor,
                  out_hw: tuple[int, int]) -> torch.Tensor:
    b, _, h, w = images.shape
    dh, dw = out_hw
    scale = torch.tensor([w - 1, h - 1], device=images.device, dtype=images.dtype)
    src = corners_norm * scale
    dst = torch.tensor([[0, 0], [dw - 1, 0], [dw - 1, dh - 1], [0, dh - 1]],
                       device=images.device, dtype=images.dtype)
    dst = dst.unsqueeze(0).expand(b, -1, -1)
    mat = get_perspective_transform(src, dst)
    return warp_perspective(images, mat, out_hw)
