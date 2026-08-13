from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBNAct, DilatedContext, DoubleConv, Down, Up, init_weights

__all__ = ["DocEnhanceNet", "background_prior"]


def background_prior(x: torch.Tensor, kernel: int = 33, blur: int = 5) -> torch.Tensor:
    h, w = x.shape[-2:]
    stride = max(1, kernel // 2)
    p = F.max_pool2d(x, kernel_size=kernel, stride=stride, padding=kernel // 2)
    p = F.avg_pool2d(p, kernel_size=blur, stride=1, padding=blur // 2)
    return F.interpolate(p, size=(h, w), mode="bilinear", align_corners=False)


class DocEnhanceNet(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 3,
        base: int = 32,
        depth: int = 4,
        dropout: float = 0.0,
        max_ch: int = 256,
        use_bg_prior: bool = True,
        bilinear: bool = True,
        se: bool = True,
        norm: str = "batch",
        bg_kernel: int = 33,
    ):
        super().__init__()
        self.use_bg_prior = use_bg_prior
        self.bg_kernel = bg_kernel
        self.depth = depth
        self.divisor = 2 ** depth

        stem_in = in_ch * (2 if use_bg_prior else 1)
        chs = [min(base * 2 ** i, max_ch) for i in range(depth + 1)]

        self.stem = DoubleConv(stem_in, chs[0], norm=norm, se=False)
        self.downs = nn.ModuleList([
            Down(chs[i], chs[i + 1], norm=norm, se=se,
                 dropout=dropout if i >= depth - 2 else 0.0)
            for i in range(depth)
        ])
        self.context = DilatedContext(chs[-1], norm=norm, dropout=dropout)
        self.ups = nn.ModuleList([
            Up(chs[depth - i], chs[depth - i - 1], chs[depth - i - 1],
               bilinear=bilinear, norm=norm, se=se,
               dropout=dropout if i == 0 else 0.0)
            for i in range(depth)
        ])
        self.refine = ConvBNAct(chs[0], chs[0], 3, norm=norm)
        self.head = nn.Conv2d(chs[0], out_ch, 1)

        init_weights(self)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, clamp: bool = False) -> torch.Tensor:
        inp = x
        if self.use_bg_prior:
            bg = background_prior(x, kernel=self.bg_kernel)
            ratio = torch.clamp(x / (bg + 1e-3), 0.0, 1.5) / 1.5
            inp = torch.cat([x, ratio], dim=1)

        feats = [self.stem(inp)]
        for down in self.downs:
            feats.append(down(feats[-1]))

        y = self.context(feats[-1])
        for i, up in enumerate(self.ups):
            y = up(y, feats[self.depth - i - 1])

        y = self.refine(y)
        out = x + self.head(y)
        return out.clamp(0.0, 1.0) if clamp else out

    @torch.no_grad()
    def enhance(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(x, clamp=True)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def describe(self) -> str:
        return (f"DocEnhanceNet(base={self.stem.conv1[0].out_channels}, depth={self.depth}, "
                f"bg_prior={self.use_bg_prior}, params={self.num_parameters() / 1e6:.2f}M)")
