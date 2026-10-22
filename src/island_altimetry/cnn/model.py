"""Stride-1, valid-convolution dilated regressor (plan: 'required model contract').

No padding, striding, pooling, normalization, dropout or dense layers: an
R x R input yields exactly one output (its center), and an H x W input yields
(H - R + 1) x (W - R + 1) outputs whose (0, 0) element belongs to input pixel
(offset, offset), offset = (R - 1) / 2. The same weights therefore serve
center-label training and fully convolutional raster inference.
"""

from __future__ import annotations

KERNEL = 3


def receptive_field(dilations: list[int], kernel: int = KERNEL) -> int:
    """R = 1 + sum((k - 1) * d) over the 3x3 layers (the 1x1 head adds nothing)."""
    return 1 + sum((kernel - 1) * d for d in dilations)


def output_offset(dilations: list[int]) -> int:
    r = receptive_field(dilations)
    if r % 2 != 1:
        raise ValueError(f"receptive field {r} must be odd to have a center pixel")
    return (r - 1) // 2


def build(in_channels: int, channels: list[int], dilations: list[int]):
    import torch.nn as nn

    class TerrainFCN(nn.Module):
        def __init__(self):
            super().__init__()
            if len(channels) != len(dilations):
                raise ValueError("channels and dilations must have equal length")
            layers, c = [], in_channels
            for co, d in zip(channels, dilations):
                layers += [nn.Conv2d(c, co, KERNEL, stride=1, padding=0, dilation=d), nn.ReLU(inplace=True)]
                c = co
            layers.append(nn.Conv2d(c, 1, 1))
            self.net = nn.Sequential(*layers)
            self.receptive_field = receptive_field(dilations)
            self.offset = output_offset(dilations)

        def forward(self, x):
            return self.net(x)[:, 0]   # (N, H - R + 1, W - R + 1)

    return TerrainFCN()
