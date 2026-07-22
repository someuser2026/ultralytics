"""YAML-level building blocks for the reference HRVMamba architecture."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("HRAdd", "HRBottleneck", "HRConv", "HRFusion")


class _HRBatchNorm2d(nn.SyncBatchNorm):
    """Reference SyncBN whose defaults are preserved by Ultralytics initialization."""


class _HRConvBNAct(nn.Module):
    """Reference HRNet convolution followed by BatchNorm and an optional ReLU."""

    def __init__(self, c1: int, c2: int, k: int, s: int = 1, groups: int = 1, act: bool = True):
        super().__init__()
        if c1 % groups or c2 % groups:
            raise ValueError(f"groups={groups} must divide both input channels {c1} and output channels {c2}.")
        self.conv = nn.Conv2d(c1, c2, k, s, padding=k // 2, groups=groups, bias=False)
        self.norm = _HRBatchNorm2d(c2, eps=1e-5, momentum=0.1)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()
        nn.init.normal_(self.conv.weight, std=0.001)
        nn.init.constant_(self.norm.weight, 1.0)
        nn.init.constant_(self.norm.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class HRConv(_HRConvBNAct):
    """YAML-visible reference HRNet Conv-BN-ReLU transition."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, groups: int = 1, act: bool = True):
        super().__init__(c1, c2, k, s, groups, act)


class HRBottleneck(nn.Module):
    """Reference HRNet/ResNet bottleneck whose output width is ``c2``."""

    expansion = 4

    def __init__(self, c1: int, c2: int, s: int = 1):
        super().__init__()
        if c2 % self.expansion:
            raise ValueError(f"HRBottleneck output channels {c2} must be divisible by {self.expansion}.")
        hidden = c2 // self.expansion
        self.conv1 = _HRConvBNAct(c1, hidden, 1, act=True)
        self.conv2 = _HRConvBNAct(hidden, hidden, 3, s=s, act=True)
        self.conv3 = _HRConvBNAct(hidden, c2, 1, act=False)
        self.shortcut = _HRConvBNAct(c1, c2, 1, s=s, act=False) if s != 1 or c1 != c2 else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv3(self.conv2(self.conv1(x))) + self.shortcut(x))


class _HRExchangeTransform(nn.Module):
    """Transform one HR branch to a target branch's channels and resolution."""

    def __init__(self, source_index: int, target_index: int, channels: Sequence[int]):
        super().__init__()
        self.source_index = source_index
        self.target_index = target_index
        source_channels = int(channels[source_index])
        target_channels = int(channels[target_index])

        if source_index == target_index:
            self.transform = nn.Identity()
        elif source_index > target_index:
            self.transform = _HRConvBNAct(source_channels, target_channels, 1, act=False)
        else:
            layers = []
            current_channels = source_channels
            for step in range(target_index - source_index):
                is_last = step == target_index - source_index - 1
                output_channels = target_channels if is_last else source_channels
                layers.extend(
                    (
                        _HRConvBNAct(current_channels, current_channels, 3, s=2, groups=current_channels, act=False),
                        _HRConvBNAct(current_channels, output_channels, 1, act=not is_last),
                    )
                )
                current_channels = output_channels
            self.transform = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        x = self.transform(x)
        if self.source_index > self.target_index:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return x


class HRFusion(nn.Module):
    """Produce one additively fused HR branch from all current branch tensors."""

    def __init__(self, in_channels: Sequence[int], target_index: int):
        super().__init__()
        if not 0 <= target_index < len(in_channels):
            raise ValueError(f"target_index={target_index} is invalid for {len(in_channels)} HR branches.")
        self.in_channels = tuple(int(c) for c in in_channels)
        self.target_index = int(target_index)
        self.transforms = nn.ModuleList(
            _HRExchangeTransform(source_index, self.target_index, self.in_channels)
            for source_index in range(len(self.in_channels))
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        if len(x) != len(self.transforms):
            raise ValueError(f"HRFusion expected {len(self.transforms)} inputs, got {len(x)}.")
        target_size = x[self.target_index].shape[-2:]
        output = self.transforms[0](x[0], target_size)
        for source, transform in zip(x[1:], self.transforms[1:]):
            output = output + transform(source, target_size)
        return self.act(output)


class HRAdd(nn.Module):
    """Elementwise addition used by the reference HRFuseScales neck."""

    def __init__(self, in_channels: Sequence[int], act: bool = False):
        super().__init__()
        if not in_channels or len(set(in_channels)) != 1:
            raise ValueError(f"HRAdd requires equal, non-empty input channels, got {list(in_channels)}.")
        self.in_channels = tuple(int(c) for c in in_channels)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        if len(x) != len(self.in_channels):
            raise ValueError(f"HRAdd expected {len(self.in_channels)} inputs, got {len(x)}.")
        output = x[0]
        for source in x[1:]:
            if source.shape != output.shape:
                raise ValueError(f"HRAdd input shapes must match, got {tuple(output.shape)} and {tuple(source.shape)}.")
            output = output + source
        return self.act(output)
