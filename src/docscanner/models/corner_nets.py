from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import ConvBNAct, DilatedContext, DoubleConv, Down, Up, init_weights

__all__ = ["CornerRegressor", "CornerHeatmapNet", "soft_argmax_2d", "coords_to_heatmaps",
           "coords_to_mask"]


class CornerRegressor(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        base: int = 24,
        depth: int = 5,
        max_ch: int = 256,
        head_dim: int = 256,
        dropout: float = 0.0,
        norm: str = "batch",
        input_size: int = 256,
        se: bool = True,
    ):
        super().__init__()
        chs = [min(base * 2 ** i, max_ch) for i in range(depth + 1)]
        self.stem = DoubleConv(in_ch, chs[0], norm=norm)
        self.downs = nn.ModuleList([
            Down(chs[i], chs[i + 1], norm=norm, se=se,
                 dropout=dropout * 0.5 if i >= depth - 2 else 0.0)
            for i in range(depth)
        ])
        self.context = DilatedContext(chs[-1], dilations=(1, 2, 4), norm=norm)
        grid = max(1, input_size // (2 ** depth))
        self.reduce = ConvBNAct(chs[-1], 64, 1, norm=norm)
        self.flat_dim = 64 * grid * grid
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(self.flat_dim, head_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_dim, head_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(head_dim // 2, 8),
        )
        self.input_size = input_size
        init_weights(self)
        last = self.head[-1]
        nn.init.zeros_(last.weight)
        with torch.no_grad():
            last.bias.copy_(torch.tensor(
                [0.15, 0.15, 0.85, 0.15, 0.85, 0.85, 0.15, 0.85], dtype=torch.float32
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.stem(x)
        for down in self.downs:
            y = down(y)
        y = self.context(y)
        y = self.reduce(y)
        out = self.head(y)
        return out.view(-1, 4, 2)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(x).clamp(-0.25, 1.25)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def coords_to_heatmaps(
    coords: torch.Tensor,
    size: int,
    sigma: float = 1.6,
    pad: float = 0.12,
) -> torch.Tensor:
    b = coords.shape[0]
    device, dtype = coords.device, coords.dtype
    lin = torch.linspace(-pad, 1.0 + pad, size, device=device, dtype=dtype)
    gx = lin.view(1, 1, 1, size)
    gy = lin.view(1, 1, size, 1)
    cx = coords[..., 0].view(b, 4, 1, 1)
    cy = coords[..., 1].view(b, 4, 1, 1)
    span = (1.0 + 2 * pad)
    s = sigma * span / size
    return torch.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * s * s))


def coords_to_mask(coords: torch.Tensor, size: int, pad: float = 0.12) -> torch.Tensor:
    b = coords.shape[0]
    device, dtype = coords.device, coords.dtype
    lin = torch.linspace(-pad, 1.0 + pad, size, device=device, dtype=dtype)
    gx = lin.view(1, 1, size).expand(b, size, size)
    gy = lin.view(1, size, 1).expand(b, size, size)

    inside = torch.ones(b, size, size, device=device, dtype=dtype)
    x, y = coords[..., 0], coords[..., 1]
    area2 = (x * torch.roll(y, -1, dims=1) - y * torch.roll(x, -1, dims=1)).sum(1)
    sign = torch.where(area2 >= 0, 1.0, -1.0).view(b, 1, 1).to(dtype)
    for i in range(4):
        p0 = coords[:, i]
        p1 = coords[:, (i + 1) % 4]
        ex = (p1[:, 0] - p0[:, 0]).view(b, 1, 1)
        ey = (p1[:, 1] - p0[:, 1]).view(b, 1, 1)
        cross = ex * (gy - p0[:, 1].view(b, 1, 1)) - ey * (gx - p0[:, 0].view(b, 1, 1))

        inside = inside * torch.sigmoid(sign * cross * float(size) * 4.0)
    return inside.unsqueeze(1)


def soft_argmax_2d(
    heatmaps: torch.Tensor,
    pad: float = 0.12,
    temperature: float = 0.02,
    window: int = 7,
) -> torch.Tensor:
    b, k, h, w = heatmaps.shape
    flat = heatmaps.reshape(b, k, -1)
    idx = flat.argmax(dim=-1)
    py = torch.div(idx, w, rounding_mode="floor")
    px = idx % w

    r = window // 2
    dy = torch.arange(-r, r + 1, device=heatmaps.device)
    dx = torch.arange(-r, r + 1, device=heatmaps.device)
    yy = (py.unsqueeze(-1).unsqueeze(-1) + dy.view(1, 1, -1, 1)).clamp(0, h - 1)
    xx = (px.unsqueeze(-1).unsqueeze(-1) + dx.view(1, 1, 1, -1)).clamp(0, w - 1)

    gather_idx = (yy * w + xx).reshape(b, k, -1)
    patch = torch.gather(flat, 2, gather_idx)
    weights = torch.softmax(patch / temperature, dim=-1)

    span = 1.0 + 2 * pad
    xs = (xx.to(heatmaps.dtype) / max(w - 1, 1)) * span - pad
    ys = (yy.to(heatmaps.dtype) / max(h - 1, 1)) * span - pad
    xs = xs.expand(b, k, window, window).reshape(b, k, -1)
    ys = ys.expand(b, k, window, window).reshape(b, k, -1)

    cx = (weights * xs).sum(-1)
    cy = (weights * ys).sum(-1)
    return torch.stack([cx, cy], dim=-1)


class CornerHeatmapNet(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        base: int = 24,
        depth: int = 4,
        out_stride: int = 4,
        max_ch: int = 192,
        dropout: float = 0.0,
        norm: str = "batch",
        heatmap_pad: float = 0.12,
        se: bool = True,
        seg_head: bool = True,
    ):
        super().__init__()
        self.heatmap_pad = heatmap_pad
        self.out_stride = out_stride
        self.depth = depth
        self.has_seg = bool(seg_head)
        chs = [min(base * 2 ** i, max_ch) for i in range(depth + 1)]

        self.stem = DoubleConv(in_ch, chs[0], norm=norm)
        self.downs = nn.ModuleList([
            Down(chs[i], chs[i + 1], norm=norm, se=se,
                 dropout=dropout if i >= depth - 2 else 0.0)
            for i in range(depth)
        ])
        self.context = DilatedContext(chs[-1], norm=norm, dropout=dropout)

        n_up = max(0, depth - (out_stride.bit_length() - 1))
        self.ups = nn.ModuleList([
            Up(chs[depth - i], chs[depth - i - 1], chs[depth - i - 1],
               bilinear=True, norm=norm, se=se)
            for i in range(n_up)
        ])
        self.head = nn.Sequential(
            ConvBNAct(chs[depth - n_up], 64, 3, norm=norm),
            nn.Conv2d(64, 4, 1),
        )
        self.seg = nn.Sequential(
            ConvBNAct(chs[depth - n_up], 32, 3, norm=norm),
            nn.Conv2d(32, 1, 1),
        ) if seg_head else None
        init_weights(self)
        nn.init.constant_(self.head[-1].bias, -4.0)

    def _trunk(self, x: torch.Tensor) -> torch.Tensor:
        feats = [self.stem(x)]
        for down in self.downs:
            feats.append(down(feats[-1]))
        y = self.context(feats[-1])
        for i, up in enumerate(self.ups):
            y = up(y, feats[self.depth - i - 1])
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self._trunk(x))

    def forward_with_seg(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        y = self._trunk(x)
        return self.head(y), (self.seg(y) if self.seg is not None else None)

    def heatmaps(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))

    def coords(self, x: torch.Tensor, window: int = 7) -> torch.Tensor:
        return soft_argmax_2d(self.heatmaps(x), pad=self.heatmap_pad, window=window)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.coords(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
