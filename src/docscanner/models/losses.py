from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "CharbonnierLoss", "SSIM", "MSSSIM", "SobelGradientLoss", "EnhancementLoss",
    "CornerCoordLoss", "HeatmapLoss", "build_enhancement_loss", "gaussian_window",
]


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps * eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((pred - target) ** 2 + self.eps2).mean()


def gaussian_window(size: int = 11, sigma: float = 1.5,
                    channels: int = 3, device=None, dtype=None) -> torch.Tensor:
    coords = torch.arange(size, dtype=dtype or torch.float32, device=device) - (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    win2d = g[:, None] @ g[None, :]
    return win2d.expand(channels, 1, size, size).contiguous()


class SSIM(nn.Module):
    def __init__(self, window_size: int = 11, sigma: float = 1.5,
                 data_range: float = 1.0, channels: int = 3, k1: float = 0.01,
                 k2: float = 0.03):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self.channels = channels
        self.c1 = (k1 * data_range) ** 2
        self.c2 = (k2 * data_range) ** 2
        self.register_buffer("window", gaussian_window(window_size, sigma, channels),
                             persistent=False)

    def _filter(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        w = self.window if c == self.channels else gaussian_window(
            self.window_size, self.sigma, c, x.device, x.dtype)
        return F.conv2d(x, w.to(x.dtype).to(x.device), padding=0, groups=c)

    def map(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.float()
        target = target.float()
        mu_x = self._filter(pred)
        mu_y = self._filter(target)
        mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
        sigma_x = (self._filter(pred * pred) - mu_x2).clamp_min(0.0)
        sigma_y = (self._filter(target * target) - mu_y2).clamp_min(0.0)
        sigma_xy = self._filter(pred * target) - mu_xy
        cs = (2 * sigma_xy + self.c2) / (sigma_x + sigma_y + self.c2)
        return ((2 * mu_xy + self.c1) / (mu_x2 + mu_y2 + self.c1)) * cs, cs

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ssim_map, _ = self.map(pred, target)
        return ssim_map.mean()


class MSSSIM(nn.Module):
    WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)

    def __init__(self, window_size: int = 11, sigma: float = 1.5,
                 data_range: float = 1.0, channels: int = 3, levels: int = 5):
        super().__init__()
        self.ssim = SSIM(window_size, sigma, data_range, channels)
        self.levels = levels
        self.window_size = window_size

    EPS = 1e-4

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.float()
        target = target.float()
        size = min(pred.shape[-2:])
        max_levels = max(1, int(math.floor(math.log2(size / (self.window_size - 1)))) + 1)
        levels = max(1, min(self.levels, max_levels))
        weights = torch.tensor(self.WEIGHTS[:levels], device=pred.device, dtype=pred.dtype)
        weights = weights / weights.sum()

        mcs = []
        x, y = pred, target
        for i in range(levels):
            ssim_map, cs = self.ssim.map(x, y)
            if i < levels - 1:
                mcs.append(cs.mean().clamp_min(self.EPS))
                x = F.avg_pool2d(x, 2)
                y = F.avg_pool2d(y, 2)
            else:
                final = ssim_map.mean().clamp_min(self.EPS)
        out = final ** weights[-1]
        for i, m in enumerate(mcs):
            out = out * (m ** weights[i])
        return out


class SobelGradientLoss(nn.Module):
    def __init__(self, channels: int = 3, magnitude: bool = True):
        super().__init__()
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        ky = kx.t().contiguous()
        self.register_buffer("kx", kx.view(1, 1, 3, 3).repeat(channels, 1, 1, 1),
                             persistent=False)
        self.register_buffer("ky", ky.view(1, 1, 3, 3).repeat(channels, 1, 1, 1),
                             persistent=False)
        self.channels = channels
        self.magnitude = magnitude

    def _grad(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c = x.shape[1]
        kx, ky = self.kx.to(x.dtype), self.ky.to(x.dtype)
        if c != self.channels:
            kx, ky = kx[:1].repeat(c, 1, 1, 1), ky[:1].repeat(c, 1, 1, 1)
        return (F.conv2d(x, kx, padding=1, groups=c),
                F.conv2d(x, ky, padding=1, groups=c))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.float()
        target = target.float()
        gx_p, gy_p = self._grad(pred)
        gx_t, gy_t = self._grad(target)
        if self.magnitude:
            mp = torch.sqrt(gx_p ** 2 + gy_p ** 2 + 1e-6)
            mt = torch.sqrt(gx_t ** 2 + gy_t ** 2 + 1e-6)
            return F.l1_loss(mp, mt)
        return F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t)


@dataclass
class LossWeights:
    charbonnier: float = 1.0
    l2: float = 0.0
    msssim: float = 0.0
    sobel: float = 0.0

    @property
    def name(self) -> str:
        parts = []
        if self.charbonnier:
            parts.append(f"l1x{self.charbonnier:g}")
        if self.l2:
            parts.append(f"l2x{self.l2:g}")
        if self.msssim:
            parts.append(f"msssim x{self.msssim:g}")
        if self.sobel:
            parts.append(f"sobel x{self.sobel:g}")
        return "+".join(parts) or "none"


class EnhancementLoss(nn.Module):
    def __init__(self, weights: LossWeights | None = None, channels: int = 3):
        super().__init__()
        self.w = weights or LossWeights()
        self.charb = CharbonnierLoss()
        self.msssim = MSSSIM(channels=channels) if self.w.msssim else None
        self.sobel = SobelGradientLoss(channels=channels) if self.w.sobel else None

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        pred = pred.float()
        target = target.float()
        comps: dict[str, torch.Tensor] = {}
        total = pred.new_zeros(())
        if self.w.charbonnier:
            c = self.charb(pred, target)
            comps["charbonnier"] = c.detach()
            total = total + self.w.charbonnier * c
        if self.w.l2:
            c = F.mse_loss(pred, target)
            comps["l2"] = c.detach()
            total = total + self.w.l2 * c
        if self.msssim is not None:
            c = 1.0 - self.msssim(pred.clamp(0, 1), target)
            comps["msssim"] = c.detach()
            total = total + self.w.msssim * c
        if self.sobel is not None:
            c = self.sobel(pred, target)
            comps["sobel"] = c.detach()
            total = total + self.w.sobel * c
        return total, comps


def build_enhancement_loss(kind: str = "combined") -> EnhancementLoss:
    kind = kind.lower()
    if kind in ("mse", "l2"):
        return EnhancementLoss(LossWeights(charbonnier=0.0, l2=1.0))
    if kind in ("l1", "charbonnier"):
        return EnhancementLoss(LossWeights(charbonnier=1.0))
    if kind in ("l1_ssim", "l1+msssim"):
        return EnhancementLoss(LossWeights(charbonnier=1.0, msssim=0.25))
    if kind in ("combined", "full"):
        return EnhancementLoss(LossWeights(charbonnier=1.0, msssim=0.25, sobel=0.35))
    raise ValueError(f"unknown loss preset: {kind}")


def cyclic_permutations(coords: torch.Tensor) -> torch.Tensor:
    return torch.stack([coords.roll(-k, dims=1) for k in range(4)], dim=1)


class CornerCoordLoss(nn.Module):
    def __init__(self, kind: str = "wing", omega: float = 0.06, epsilon: float = 0.01,
                 permutation_invariant: bool = False):
        super().__init__()
        self.kind = kind
        self.omega = omega
        self.epsilon = epsilon
        self.permutation_invariant = permutation_invariant
        self._c = omega - omega * math.log(1 + omega / epsilon)

    def _elementwise(self, d: torch.Tensor) -> torch.Tensor:
        if self.kind == "l1":
            return d
        if self.kind == "l2":
            return d ** 2
        return torch.where(d < self.omega,
                           self.omega * torch.log1p(d / self.epsilon),
                           d - self._c)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self.permutation_invariant:
            return self._elementwise((pred - target).abs()).mean()
        perms = cyclic_permutations(target)
        d = (pred.unsqueeze(1) - perms).abs()
        per_perm = self._elementwise(d).mean(dim=(2, 3))
        return per_perm.min(dim=1).values.mean()


class HeatmapLoss(nn.Module):
    def __init__(self, heatmap_weight: float = 1.0, coord_weight: float = 0.15,
                 pos_weight: float = 24.0, pad: float = 0.12, sigma: float = 1.6,
                 permutation_invariant: bool = False, seg_weight: float = 0.5):
        super().__init__()
        self.hm_w = heatmap_weight
        self.coord_w = coord_weight
        self.pos_weight = pos_weight
        self.pad = pad
        self.sigma = sigma
        self.permutation_invariant = permutation_invariant
        self.seg_w = seg_weight

    def _weighted_bce(self, logits: torch.Tensor, target_hm: torch.Tensor) -> torch.Tensor:
        weight = 1.0 + self.pos_weight * target_hm
        el = F.binary_cross_entropy_with_logits(logits, target_hm, reduction="none") * weight
        return el.mean(dim=(1, 2, 3))

    def forward(self, logits: torch.Tensor, target_coords: torch.Tensor,
                seg_logits: torch.Tensor | None = None):
        from .corner_nets import coords_to_heatmaps, coords_to_mask, soft_argmax_2d

        size = logits.shape[-1]
        b = logits.shape[0]

        if self.permutation_invariant:
            perms = cyclic_permutations(target_coords)
            flat = perms.reshape(b * 4, 4, 2)
            target_hm = coords_to_heatmaps(flat, size, sigma=self.sigma, pad=self.pad)
            rep = logits.unsqueeze(1).expand(b, 4, *logits.shape[1:]).reshape(
                b * 4, *logits.shape[1:])
            per = self._weighted_bce(rep, target_hm).view(b, 4)
            best = per.argmin(dim=1)
            hm_loss = per.gather(1, best[:, None]).squeeze(1).mean()
            matched = perms[torch.arange(b, device=perms.device), best]
        else:
            target_hm = coords_to_heatmaps(target_coords, size, sigma=self.sigma,
                                           pad=self.pad)
            hm_loss = self._weighted_bce(logits, target_hm).mean()
            matched = target_coords

        comps = {"heatmap": hm_loss.detach()}
        total = self.hm_w * hm_loss
        if self.coord_w:
            coords = soft_argmax_2d(torch.sigmoid(logits), pad=self.pad)
            c_loss = F.l1_loss(coords, matched)
            comps["coord"] = c_loss.detach()
            total = total + self.coord_w * c_loss
        if seg_logits is not None and self.seg_w:
            target_mask = coords_to_mask(matched, seg_logits.shape[-1], pad=self.pad)
            s_loss = F.binary_cross_entropy_with_logits(seg_logits, target_mask)
            comps["seg"] = s_loss.detach()
            total = total + self.seg_w * s_loss
        return total, comps
