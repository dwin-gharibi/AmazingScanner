from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ConvBNAct", "DoubleConv", "Down", "Up", "DilatedContext", "SEBlock", "init_weights"]


class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1,
                 dilation: int = 1, norm: str = "batch", act: bool = True,
                 groups: int = 1):
        pad = dilation * (kernel - 1) // 2
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=pad,
                      dilation=dilation, groups=groups, bias=norm == "none")
        ]
        if norm == "batch":
            layers.append(nn.BatchNorm2d(out_ch))
        elif norm == "instance":
            layers.append(nn.InstanceNorm2d(out_ch, affine=True))
        elif norm == "group":
            layers.append(nn.GroupNorm(min(8, out_ch), out_ch))
        if act:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: str = "batch",
                 dropout: float = 0.0, se: bool = False, residual: bool = True):
        super().__init__()
        self.conv1 = ConvBNAct(in_ch, out_ch, 3, norm=norm)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = ConvBNAct(out_ch, out_ch, 3, norm=norm, act=False)
        self.se = SEBlock(out_ch) if se else nn.Identity()
        self.act = nn.ReLU(inplace=True)
        self.residual = residual
        self.skip = (nn.Identity() if in_ch == out_ch
                     else nn.Conv2d(in_ch, out_ch, 1, bias=False)) if residual else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.drop(y)
        y = self.conv2(y)
        y = self.se(y)
        if self.residual and self.skip is not None:
            y = y + self.skip(x)
        return self.act(y)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, **kw):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = DoubleConv(in_ch, out_ch, **kw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 bilinear: bool = True, **kw):
        super().__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                ConvBNAct(in_ch, in_ch // 2, 3),
            )
            up_ch = in_ch // 2
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
            up_ch = in_ch // 2
        self.block = DoubleConv(up_ch + skip_ch, out_ch, **kw)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        dy, dx = skip.shape[-2] - x.shape[-2], skip.shape[-1] - x.shape[-1]
        if dy or dx:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.block(torch.cat([skip, x], dim=1))


class DilatedContext(nn.Module):
    def __init__(self, channels: int, dilations: tuple[int, ...] = (1, 2, 4, 8),
                 norm: str = "batch", dropout: float = 0.0):
        super().__init__()
        branch = max(8, channels // len(dilations))
        self.branches = nn.ModuleList(
            [ConvBNAct(channels, branch, 3, dilation=d, norm=norm) for d in dilations]
        )
        self.pool_branch = ConvBNAct(channels, branch, 1, norm=norm)
        self.fuse = ConvBNAct(branch * (len(dilations) + 1), channels, 1, norm=norm)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [b(x) for b in self.branches]
        g = F.adaptive_avg_pool2d(x, 1)
        g = self.pool_branch(g).expand(-1, -1, x.shape[-2], x.shape[-1])
        feats.append(g)
        return self.drop(self.fuse(torch.cat(feats, dim=1)))


def init_weights(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
