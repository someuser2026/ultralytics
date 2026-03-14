# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""LEGNet backbone blocks and multi-scale feature extractor."""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .block import DropPath

__all__ = ("LWEGNet",)


class FrozenDepthwiseConv2d(nn.Module):
    """Depthwise convolution backed by a non-trainable kernel buffer."""

    def __init__(self, channels: int, kernel: Tensor, padding: int) -> None:
        super().__init__()
        self.channels = channels
        self.padding = padding
        self.register_buffer("weight", kernel.repeat(channels, 1, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        return F.conv2d(x, self.weight.to(dtype=x.dtype), padding=self.padding, groups=self.channels)


def _gaussian_kernel(size: int, sigma: float) -> Tensor:
    radius = size // 2
    kernel = torch.tensor(
        [
            [
                (1 / (2 * math.pi * sigma**2)) * math.exp(-(x**2 + y**2) / (2 * sigma**2))
                for x in range(-radius, radius + 1)
            ]
            for y in range(-radius, radius + 1)
        ],
        dtype=torch.float32,
    )
    kernel = kernel / kernel.sum()
    return kernel.unsqueeze(0).unsqueeze(0)


def _log_kernel(size: int, sigma: float) -> Tensor:
    radius = size // 2
    ax = torch.arange(-radius, radius + 1, dtype=torch.float32)
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    kernel = (xx**2 + yy**2 - 2 * sigma**2) / (2 * math.pi * sigma**4) * torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel - kernel.mean()
    kernel_sum = kernel.sum()
    if kernel_sum.abs() < 1e-6:
        kernel_sum = kernel.abs().sum().clamp_min(1e-6)
    kernel = kernel / kernel_sum
    return kernel.unsqueeze(0).unsqueeze(0)


class ConvExtra(nn.Module):
    def __init__(self, channels: int, act_layer: type[nn.Module]) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, 64, 1),
            nn.BatchNorm2d(64),
            act_layer(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            act_layer(),
            nn.Conv2d(64, channels, 1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class Gaussian(nn.Module):
    def __init__(
        self,
        channels: int,
        size: int,
        sigma: float,
        act_layer: type[nn.Module],
        feature_extra: bool = True,
    ) -> None:
        super().__init__()
        self.gaussian = FrozenDepthwiseConv2d(channels, _gaussian_kernel(size, sigma), padding=size // 2)
        self.norm = nn.BatchNorm2d(channels)
        self.act = act_layer()
        self.conv_extra = ConvExtra(channels, act_layer) if feature_extra else None

    def forward(self, x: Tensor) -> Tensor:
        gaussian = self.act(self.norm(self.gaussian(x)))
        if self.conv_extra is None:
            return gaussian
        return self.conv_extra(x + gaussian)


class Scharr(nn.Module):
    def __init__(self, channels: int, act_layer: type[nn.Module]) -> None:
        super().__init__()
        scharr_x = torch.tensor([[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]], dtype=torch.float32)
        scharr_y = torch.tensor([[-3.0, -10.0, -3.0], [0.0, 0.0, 0.0], [3.0, 10.0, 3.0]], dtype=torch.float32)
        self.conv_x = FrozenDepthwiseConv2d(channels, scharr_x.unsqueeze(0).unsqueeze(0), padding=1)
        self.conv_y = FrozenDepthwiseConv2d(channels, scharr_y.unsqueeze(0).unsqueeze(0), padding=1)
        self.norm = nn.BatchNorm2d(channels)
        self.act = act_layer()
        self.conv_extra = ConvExtra(channels, act_layer)

    def forward(self, x: Tensor) -> Tensor:
        edges_x = self.conv_x(x)
        edges_y = self.conv_y(x)
        scharr_edge = torch.sqrt(edges_x.pow(2) + edges_y.pow(2) + 1e-6)
        scharr_edge = self.act(self.norm(scharr_edge))
        return self.conv_extra(x + scharr_edge)


class LFEA(nn.Module):
    def __init__(self, channels: int, act_layer: type[nn.Module]) -> None:
        super().__init__()
        t = int(abs((math.log(channels, 2) + 1) / 2))
        kernel_size = t if t % 2 else t + 1
        self.conv2d = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            act_layer(),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, c: Tensor, att: Tensor) -> Tensor:
        att = c * att + c
        att = self.conv2d(att)
        weight = self.avg_pool(att)
        weight = self.conv1d(weight.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        weight = self.sigmoid(weight)
        return self.norm(c + att * weight)


class LFEModule(nn.Module):
    def __init__(
        self,
        dim: int,
        stage: int,
        mlp_ratio: float,
        drop_path: float,
        act_layer: type[nn.Module],
    ) -> None:
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            nn.BatchNorm2d(mlp_hidden_dim),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False),
        )
        self.lfea = LFEA(dim, act_layer)
        self.edge = Scharr(dim, act_layer) if stage == 0 else Gaussian(dim, 5, 1.0, act_layer)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x: Tensor) -> Tensor:
        x_att = self.lfea(x, self.edge(x))
        return x + self.norm(self.drop_path(self.mlp(x_att)))


class BasicStage(nn.Module):
    def __init__(
        self,
        dim: int,
        stage: int,
        depth: int,
        mlp_ratio: float,
        drop_path: Iterable[float],
        act_layer: type[nn.Module],
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *(LFEModule(dim, stage, mlp_ratio, float(dp), act_layer) for dp in drop_path[:depth])
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


class LoGFilter(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, sigma: float, act_layer: type[nn.Module]) -> None:
        super().__init__()
        self.conv_init = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=1, padding=3)
        self.log = FrozenDepthwiseConv2d(out_channels, _log_kernel(kernel_size, sigma), padding=kernel_size // 2)
        self.act = act_layer()
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_init(x)
        log_edge = self.act(self.norm1(self.log(x)))
        return self.norm2(x + log_edge)


class DRFD(nn.Module):
    def __init__(self, dim: int, act_layer: type[nn.Module]) -> None:
        super().__init__()
        self.outdim = dim * 2
        self.conv = nn.Conv2d(dim, self.outdim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.conv_c = nn.Conv2d(self.outdim, self.outdim, kernel_size=3, stride=2, padding=1, groups=self.outdim)
        self.act_c = act_layer()
        self.norm_c = nn.BatchNorm2d(self.outdim)
        self.max_m = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.norm_m = nn.BatchNorm2d(self.outdim)
        self.fusion = nn.Conv2d(self.outdim * 2, self.outdim, kernel_size=1, stride=1)
        self.gaussian = Gaussian(self.outdim, 5, 0.5, act_layer, feature_extra=False)
        self.norm_g = nn.BatchNorm2d(self.outdim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.norm_g(x + self.gaussian(x))
        pooled = self.norm_m(self.max_m(x))
        conv = self.norm_c(self.act_c(self.conv_c(x)))
        return self.fusion(torch.cat([conv, pooled], dim=1))


class Stem(nn.Module):
    def __init__(self, in_channels: int, stem_dim: int, act_layer: type[nn.Module]) -> None:
        super().__init__()
        out_c14 = stem_dim // 4
        out_c12 = stem_dim // 2
        self.conv_d = nn.Sequential(
            nn.Conv2d(out_c14, out_c12, kernel_size=3, stride=1, padding=1, groups=out_c14),
            nn.Conv2d(out_c12, out_c12, kernel_size=3, stride=2, padding=1, groups=out_c12),
            nn.BatchNorm2d(out_c12),
        )
        self.log = LoGFilter(in_channels, out_c14, 7, 1.0, act_layer)
        self.gaussian = Gaussian(out_c12, 9, 0.5, act_layer)
        self.norm = nn.BatchNorm2d(out_c12)
        self.drfd = DRFD(out_c12, act_layer)

    def forward(self, x: Tensor) -> Tensor:
        x = self.log(x)
        x = self.conv_d(x)
        x = self.norm(x + self.gaussian(x))
        return self.drfd(x)


class LWEGNet(nn.Module):
    """LEGNet multi-scale backbone returning P2-P5 features."""

    def __init__(
        self,
        in_chans: int = 3,
        stem_dim: int = 64,
        depths: tuple[int, int, int, int] = (1, 4, 4, 2),
        mlp_ratio: float = 2.0,
        drop_path_rate: float = 0.1,
        act_layer: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.channels = [int(stem_dim * 2**i) for i in range(len(depths))]
        self.strides = [4, 8, 16, 32]
        self.num_features = self.channels[-1]
        self.stem = Stem(in_chans, stem_dim, act_layer)

        drop_path = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        start = 0
        for i, depth in enumerate(depths):
            stage_drop = drop_path[start : start + depth]
            self.stages.append(BasicStage(self.channels[i], i, depth, mlp_ratio, stage_drop, act_layer))
            start += depth
            if i < len(depths) - 1:
                self.downsamples.append(DRFD(self.channels[i], act_layer))
        self.out_norms = nn.ModuleList(nn.BatchNorm2d(channel) for channel in self.channels)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        x = self.stem(x)
        outs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            outs.append(self.out_norms[i](x))
            if i < len(self.downsamples):
                x = self.downsamples[i](x)
        return tuple(outs)
