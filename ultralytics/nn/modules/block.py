# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Block modules."""

from __future__ import annotations
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad, SE
from .transformer import TransformerBlock

__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ELAN1",
    "ADown",
    "AConv",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "C3k2",
    "C2fPSA",
    "C2PSA",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
    "SCDown",
    "TorchVision",
    # ConvNeXt family
    "ConvNeXtLayerNorm",
    "DropPath",
    "ConvNeXtStem",
    "ConvNeXtDownsample",
    "ConvNeXtBlock",
    "GRN",
    "Timm",
    # "DinoV3Backbone",
    "MaxMBConv",
    "WindowSA",
    "GridSA",
    "MaxViTBlock",
)


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1: int = 16):
        """
        Initialize a convolutional layer with a given number of input channels.

        Args:
            c1 (int): Number of input channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the DFL module to input tensor and return transformed output."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """Ultralytics YOLO models mask Proto module for segmentation models."""

    def __init__(self, c1: int, c_: int = 256, c2: int = 32):
        """
        Initialize the Ultralytics YOLO models mask Proto module with specified number of protos and masks.

        Args:
            c1 (int): Input channels.
            c_ (int): Intermediate channels.
            c2 (int): Output channels (number of protos).
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1: int, cm: int, c2: int):
        """
        Initialize the StemBlock of PPHGNetV2.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(
        self,
        c1: int,
        cm: int,
        c2: int,
        k: int = 3,
        n: int = 6,
        lightconv: bool = False,
        shortcut: bool = False,
        act: nn.Module = nn.ReLU(),
    ):
        """
        Initialize HGBlock with specified parameters.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            n (int): Number of LightConv or Conv blocks.
            lightconv (bool): Whether to use LightConv.
            shortcut (bool): Whether to use shortcut connection.
            act (nn.Module): Activation function.
        """
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1: int, c2: int, k: tuple[int, ...] = (5, 9, 13)):
        """
        Initialize the SPP layer with input/output channels and pooling kernel sizes.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (tuple): Kernel sizes for max pooling.
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1: int, c2: int, k: int = 5):
        """
        Initialize the SPPF layer with given input/output channels and kernel size.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.

        Notes:
            This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sequential pooling operations to input and return concatenated feature maps."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1: int, c2: int, n: int = 1):
        """
        Initialize the CSP Bottleneck with 1 convolution.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of convolutions.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution and residual connection to input tensor."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """
        Initialize a CSP Bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """
        Initialize a CSP bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """
        Initialize the CSP Bottleneck with 3 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 3 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """
        Initialize C3 module with cross-convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1: int, c2: int, n: int = 3, e: float = 1.0):
        """
        Initialize CSP Bottleneck with a single convolution.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepConv blocks.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of RepC3 module."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """
        Initialize C3 module with TransformerBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Transformer blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """
        Initialize C3 module with GhostBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Ghost bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/Efficient-AI-Backbones."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1):
        """
        Initialize Ghost Bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """
        Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """
        Initialize CSP Bottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1: int, c2: int, s: int = 1, e: int = 4):
        """
        Initialize ResNet block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            e (int): Expansion ratio.
        """
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1: int, c2: int, s: int = 1, is_first: bool = False, n: int = 1, e: int = 4):
        """
        Initialize ResNet layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            is_first (bool): Whether this is the first layer.
            n (int): Number of ResNet blocks.
            e (int): Expansion ratio.
        """
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1: int, c2: int, nh: int = 1, ec: int = 128, gc: int = 512, scale: bool = False):
        """
        Initialize MaxSigmoidAttnBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            nh (int): Number of heads.
            ec (int): Embedding channels.
            gc (int): Guide channels.
            scale (bool): Whether to use learnable scale parameter.
        """
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of MaxSigmoidAttnBlock.

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor.

        Returns:
            (torch.Tensor): Output tensor after attention.
        """
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, guide.shape[1], self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        ec: int = 128,
        nh: int = 1,
        gc: int = 512,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ):
        """
        Initialize C2f module with attention mechanism.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            ec (int): Embedding channels for attention.
            nh (int): Number of heads for attention.
            gc (int): Guide channels for attention.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through C2f layer with attention.

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor for attention.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using split() instead of chunk().

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor for attention.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(
        self, ec: int = 256, ch: tuple[int, ...] = (), ct: int = 512, nh: int = 8, k: int = 3, scale: bool = False
    ):
        """
        Initialize ImagePoolingAttn module.

        Args:
            ec (int): Embedding channels.
            ch (tuple): Channel dimensions for feature maps.
            ct (int): Channel dimension for text embeddings.
            nh (int): Number of attention heads.
            k (int): Kernel size for pooling.
            scale (bool): Whether to use learnable scale parameter.
        """
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x: list[torch.Tensor], text: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of ImagePoolingAttn.

        Args:
            x (list[torch.Tensor]): List of input feature maps.
            text (torch.Tensor): Text embeddings.

        Returns:
            (torch.Tensor): Enhanced text embeddings.
        """
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Forward function of contrastive learning.

        Args:
            x (torch.Tensor): Image features.
            w (torch.Tensor): Text features.

        Returns:
            (torch.Tensor): Similarity scores.
        """
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """
        Initialize BNContrastiveHead.

        Args:
            embed_dims (int): Embedding dimensions for features.
        """
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def fuse(self):
        """Fuse the batch normalization layer in the BNContrastiveHead module."""
        del self.norm
        del self.bias
        del self.logit_scale
        self.forward = self.forward_fuse

    def forward_fuse(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Passes input out unchanged."""
        return x

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Forward function of contrastive learning with batch normalization.

        Args:
            x (torch.Tensor): Image features.
            w (torch.Tensor): Text features.

        Returns:
            (torch.Tensor): Similarity scores.
        """
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)

        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """
        Initialize RepBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """
        Initialize RepCSP layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepBottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1: int, c2: int, c3: int, c4: int, n: int = 1):
        """
        Initialize CSP-ELAN layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for RepCSP.
            n (int): Number of RepCSP blocks.
        """
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1: int, c2: int, c3: int, c4: int):
        """
        Initialize ELAN1 layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for convolutions.
        """
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1: int, c2: int):
        """
        Initialize AConv module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1: int, c2: int):
        """
        Initialize ADown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1: int, c2: int, c3: int, k: int = 5):
        """
        Initialize SPP-ELAN block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            k (int): Kernel size for max pooling.
        """
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1: int, c2s: list[int], k: int = 1, s: int = 1, p: int | None = None, g: int = 1):
        """
        Initialize CBLinear module.

        Args:
            c1 (int): Input channels.
            c2s (list[int]): List of output channel sizes.
            k (int): Kernel size.
            s (int): Stride.
            p (int | None): Padding.
            g (int): Groups.
        """
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx: list[int]):
        """
        Initialize CBFuse module.

        Args:
            idx (list[int]): Indices for feature selection.
        """
        super().__init__()
        self.idx = idx

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through CBFuse layer.

        Args:
            xs (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Fused output tensor.
        """
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """
        Initialize CSP bottleneck layer with two convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C3f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(
        self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, g: int = 1, shortcut: bool = True
    ):
        """
        Initialize C3k2 module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            c3k (bool): Whether to use C3k blocks.
            e (float): Expansion ratio.
            g (int): Groups for convolutions.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 3):
        """
        Initialize C3k module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
            k (int): Kernel size.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""

    def __init__(self, ed: int) -> None:
        """
        Initialize RepVGGDW module.

        Args:
            ed (int): Input and output channels.
        """
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """
        Fuse the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """
    Conditional Identity Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5, lk: bool = False):
        """
        Initialize the CIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            e (float): Expansion ratio.
            lk (bool): Whether to use RepVGGDW.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """
    C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use local key connection. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(
        self, c1: int, c2: int, n: int = 1, shortcut: bool = False, lk: bool = False, g: int = 1, e: float = 0.5
    ):
        """
        Initialize C2fCIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of CIB modules.
            shortcut (bool): Whether to use shortcut connection.
            lk (bool): Whether to use local key connection.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        """
        Initialize multi-head attention module.

        Args:
            dim (int): Input dimension.
            num_heads (int): Number of attention heads.
            attn_ratio (float): Attention ratio for key dimension.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True) -> None:
        """
        Initialize the PSABlock.

        Args:
            c (int): Input and output channels.
            attn_ratio (float): Attention ratio for key dimension.
            num_heads (int): Number of attention heads.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute a forward pass through PSABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """
    PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1: int, c2: int, e: float = 0.5):
        """
        Initialize PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute forward pass in PSA module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """
        Initialize C2PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the input tensor through a series of PSA blocks.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """
    C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.ModuleList): List of PSA blocks for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.models.common import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """
        Initialize C2fPSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))


class SCDown(nn.Module):
    """
    SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1: int, c2: int, k: int, s: int):
        """
        Initialize SCDown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply convolution and downsampling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Downsampled output tensor.
        """
        return self.cv2(self.cv1(x))


class TorchVision(nn.Module):
    """
    TorchVision module to allow loading any torchvision model.

    This class provides a way to load a model from the torchvision library, optionally load pre-trained weights, and customize the model by truncating or unwrapping layers.

    Attributes:
        m (nn.Module): The loaded torchvision model, possibly truncated and unwrapped.

    Args:
        model (str): Name of the torchvision model to load.
        weights (str, optional): Pre-trained weights to load. Default is "DEFAULT".
        unwrap (bool, optional): If True, unwraps the model to a sequential containing all but the last `truncate` layers. Default is True.
        truncate (int, optional): Number of layers to truncate from the end if `unwrap` is True. Default is 2.
        split (bool, optional): Returns output from intermediate child modules as list. Default is False.
    """

    def __init__(
        self, model: str, weights: str = "DEFAULT", unwrap: bool = True, truncate: int = 2, split: bool = False
    ):
        """
        Load the model and weights from torchvision.

        Args:
            model (str): Name of the torchvision model to load.
            weights (str): Pre-trained weights to load.
            unwrap (bool): Whether to unwrap the model.
            truncate (int): Number of layers to truncate.
            split (bool): Whether to split the output.
        """
        import torchvision  # scope for faster 'import ultralytics'

        super().__init__()
        if hasattr(torchvision.models, "get_model"):
            self.m = torchvision.models.get_model(model, weights=weights)
        else:
            self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        if unwrap:
            layers = list(self.m.children())
            if isinstance(layers[0], nn.Sequential):  # Second-level for some models like EfficientNet, Swin
                layers = [*list(layers[0].children()), *layers[1:]]
            self.m = nn.Sequential(*(layers[:-truncate] if truncate else layers))
            self.split = split
        else:
            self.split = False
            self.m.head = self.m.heads = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor | list[torch.Tensor]): Output tensor or list of tensors.
        """
        if self.split:
            y = [x]
            y.extend(m(y[-1]) for m in self.m)
        else:
            y = self.m(x)
        return y

# class DinoV3Backbone(nn.Module):
#     """
#     Load any pre-trained backbone from .pt or .pth file.

#     This class is designed to easily load any SSL backbone for transfer learning tasks.

#     Attributes:
#         m (nn.module): The loaded SSL backbone model.

#     Args:
#         weights (str): Pre-trained weights to load.
#         unwrap (bool): Whether to unwrap the model.
#         truncate (int): Number of layers to truncate.
#         split (bool): Whether to split the output.
#     """
    

class AAttn(nn.Module):
    """
    Area-attention module for YOLO models, providing efficient attention mechanisms.

    This module implements an area-based attention mechanism that processes input features in a spatially-aware manner,
    making it particularly effective for object detection tasks.

    Attributes:
        area (int): Number of areas the feature map is divided.
        num_heads (int): Number of heads into which the attention mechanism is divided.
        head_dim (int): Dimension of each attention head.
        qkv (Conv): Convolution layer for computing query, key and value tensors.
        proj (Conv): Projection convolution layer.
        pe (Conv): Position encoding convolution layer.

    Methods:
        forward: Applies area-attention to input tensor.

    Examples:
        >>> attn = AAttn(dim=256, num_heads=8, area=4)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = attn(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, area: int = 1):
        """
        Initialize an Area-attention module for YOLO models.

        Args:
            dim (int): Number of hidden channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            area (int): Number of areas the feature map is divided.
        """
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qkv = Conv(dim, all_head_dim * 3, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        self.pe = Conv(all_head_dim, dim, 7, 1, 3, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the input tensor through the area-attention.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention.
        """
        B, C, H, W = x.shape
        N = H * W

        qkv = self.qkv(x).flatten(2).transpose(1, 2)
        if self.area > 1:
            qkv = qkv.reshape(B * self.area, N // self.area, C * 3)
            B, N, _ = qkv.shape
        q, k, v = (
            qkv.view(B, N, self.num_heads, self.head_dim * 3)
            .permute(0, 2, 3, 1)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )
        attn = (q.transpose(-2, -1) @ k) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        x = v @ attn.transpose(-2, -1)
        x = x.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            v = v.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        v = v.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        x = x + self.pe(v)
        return self.proj(x)


class ABlock(nn.Module):
    """
    Area-attention block module for efficient feature extraction in YOLO models.

    This module implements an area-attention mechanism combined with a feed-forward network for processing feature maps.
    It uses a novel area-based attention approach that is more efficient than traditional self-attention while
    maintaining effectiveness.

    Attributes:
        attn (AAttn): Area-attention module for processing spatial features.
        mlp (nn.Sequential): Multi-layer perceptron for feature transformation.

    Methods:
        _init_weights: Initializes module weights using truncated normal distribution.
        forward: Applies area-attention and feed-forward processing to input tensor.

    Examples:
        >>> block = ABlock(dim=256, num_heads=8, mlp_ratio=1.2, area=1)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 1.2, area: int = 1):
        """
        Initialize an Area-attention block module.

        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            area (int): Number of areas the feature map is divided.
        """
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """
        Initialize weights using a truncated normal distribution.

        Args:
            m (nn.Module): Module to initialize.
        """
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention and feed-forward processing.
        """
        x = x + self.attn(x)
        return x + self.mlp(x)


class A2C2f(nn.Module):
    """
    Area-Attention C2f module for enhanced feature extraction with area-based attention mechanisms.

    This module extends the C2f architecture by incorporating area-attention and ABlock layers for improved feature
    processing. It supports both area-attention and standard convolution modes.

    Attributes:
        cv1 (Conv): Initial 1x1 convolution layer that reduces input channels to hidden channels.
        cv2 (Conv): Final 1x1 convolution layer that processes concatenated features.
        gamma (nn.Parameter | None): Learnable parameter for residual scaling when using area attention.
        m (nn.ModuleList): List of either ABlock or C3k modules for feature processing.

    Methods:
        forward: Processes input through area-attention or standard convolution pathway.

    Examples:
        >>> m = A2C2f(512, 512, n=1, a2=True, area=1)
        >>> x = torch.randn(1, 512, 32, 32)
        >>> output = m(x)
        >>> print(output.shape)
        torch.Size([1, 512, 32, 32])
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        a2: bool = True,
        area: int = 1,
        residual: bool = False,
        mlp_ratio: float = 2.0,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
    ):
        """
        Initialize Area-Attention C2f module.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of ABlock or C3k modules to stack.
            a2 (bool): Whether to use area attention blocks. If False, uses C3k blocks instead.
            area (int): Number of areas the feature map is divided.
            residual (bool): Whether to use residual connections with learnable gamma parameter.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            e (float): Channel expansion ratio for hidden channels.
            g (int): Number of groups for grouped convolutions.
            shortcut (bool): Whether to use shortcut connections in C3k blocks.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if a2 and residual else None
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through A2C2f layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.view(-1, self.gamma.shape[0], 1, 1) * y
        return y


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network for transformer-based architectures."""

    def __init__(self, gc: int, ec: int, e: int = 4) -> None:
        """
        Initialize SwiGLU FFN with input dimension, output dimension, and expansion factor.

        Args:
            gc (int): Guide channels.
            ec (int): Embedding channels.
            e (int): Expansion factor.
        """
        super().__init__()
        self.w12 = nn.Linear(gc, e * ec)
        self.w3 = nn.Linear(e * ec // 2, ec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU transformation to input features."""
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class Residual(nn.Module):
    """Residual connection wrapper for neural network modules."""

    def __init__(self, m: nn.Module) -> None:
        """
        Initialize residual module with the wrapped module.

        Args:
            m (nn.Module): Module to wrap with residual connection.
        """
        super().__init__()
        self.m = m
        nn.init.zeros_(self.m.w3.bias)
        # For models with l scale, please change the initialization to
        # nn.init.constant_(self.m.w3.weight, 1e-6)
        nn.init.zeros_(self.m.w3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual connection to input features."""
        return x + self.m(x)


class SAVPE(nn.Module):
    """Spatial-Aware Visual Prompt Embedding module for feature enhancement."""

    def __init__(self, ch: list[int], c3: int, embed: int):
        """
        Initialize SAVPE module with channels, intermediate channels, and embedding dimension.

        Args:
            ch (list[int]): List of input channel dimensions.
            c3 (int): Intermediate channels.
            embed (int): Embedding dimension.
        """
        super().__init__()
        self.cv1 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3), Conv(c3, c3, 3), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity()
            )
            for i, x in enumerate(ch)
        )

        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 1), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity())
            for i, x in enumerate(ch)
        )

        self.c = 16
        self.cv3 = nn.Conv2d(3 * c3, embed, 1)
        self.cv4 = nn.Conv2d(3 * c3, self.c, 3, padding=1)
        self.cv5 = nn.Conv2d(1, self.c, 3, padding=1)
        self.cv6 = nn.Sequential(Conv(2 * self.c, self.c, 3), nn.Conv2d(self.c, self.c, 3, padding=1))

    def forward(self, x: list[torch.Tensor], vp: torch.Tensor) -> torch.Tensor:
        """Process input features and visual prompts to generate enhanced embeddings."""
        y = [self.cv2[i](xi) for i, xi in enumerate(x)]
        y = self.cv4(torch.cat(y, dim=1))

        x = [self.cv1[i](xi) for i, xi in enumerate(x)]
        x = self.cv3(torch.cat(x, dim=1))

        B, C, H, W = x.shape

        Q = vp.shape[1]

        x = x.view(B, C, -1)

        y = y.reshape(B, 1, self.c, H, W).expand(-1, Q, -1, -1, -1).reshape(B * Q, self.c, H, W)
        vp = vp.reshape(B, Q, 1, H, W).reshape(B * Q, 1, H, W)

        y = self.cv6(torch.cat((y, self.cv5(vp)), dim=1))

        y = y.reshape(B, Q, self.c, -1)
        vp = vp.reshape(B, Q, 1, -1)

        score = y * vp + torch.logical_not(vp) * torch.finfo(y.dtype).min
        score = F.softmax(score, dim=-1).to(y.dtype)
        aggregated = score.transpose(-2, -3) @ x.reshape(B, self.c, C // self.c, -1).transpose(-1, -2)

        return F.normalize(aggregated.transpose(-2, -3).reshape(B, Q, -1), dim=-1, p=2)


class ConvNeXtLayerNorm(nn.Module):
    """
    LayerNorm supporting channels_last (NHWC) and channels_first (NCHW).
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6, data_format: str = "channels_last") -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        assert data_format in ("channels_last", "channels_first")
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # channels_first
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class DropPath(nn.Module):
    """Stochastic Depth per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class ConvNeXtStem(nn.Module):
    """ConvNeXt stem: Conv4x4 stride 4 followed by LayerNorm (channels_first)."""

    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size=4, stride=4)
        self.norm = ConvNeXtLayerNorm(c2, data_format="channels_first")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.norm(x)


class ConvNeXtDownsample(nn.Module):
    """ConvNeXt downsample: LayerNorm (channels_first) then Conv2d 2x2 stride 2."""

    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.norm = ConvNeXtLayerNorm(c1, data_format="channels_first")
        self.conv = nn.Conv2d(c1, c2, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.conv(x)


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt/ConvNeXtV2 Block (unified):
    DWConv(k=7) -> channels_last LayerNorm -> 1x1 MLP (4x, GELU) ->
      - if use_grn: GRN -> 1x1 -> residual -> DropPath (ConvNeXtV2)
      - else: 1x1 -> layer-scale gamma -> residual -> DropPath (ConvNeXt)
    Keeps channel dimension; if c1 != c2, input is projected to c2 first.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        layer_scale_init_value: float = 1e-6,
        drop_path: float = 0.0,
        use_grn: bool = False,
        max_drop_path: float = 0.5,
        drop_path_method: str = "linear",
    ):
        super().__init__()
        self.use_grn = use_grn
        self.proj_in = nn.Conv2d(c1, c2, kernel_size=1) if c1 != c2 else nn.Identity()
        self.dwconv = nn.Conv2d(c2, c2, kernel_size=7, padding=3, groups=c2)
        self.norm = ConvNeXtLayerNorm(c2, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Conv2d(c2, 4 * c2, kernel_size=1)
        self.act = nn.GELU()
        self.grn = GRN(4 * c2) if use_grn else None
        self.pwconv2 = nn.Conv2d(4 * c2, c2, kernel_size=1)
        self.gamma = (
            None if use_grn else (nn.Parameter(layer_scale_init_value * torch.ones((c2))) if layer_scale_init_value > 0 else None)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.proj_in(x)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.pwconv1(x)
        x = self.act(x)
        if self.grn is not None:
            x = x.permute(0, 2, 3, 1)
            x = self.grn(x)
            x = x.permute(0, 3, 1, 2)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma.view(1, -1, 1, 1) * x
        x = self.drop_path(x)
        return shortcut + x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x
class Timm(nn.Module):
    """
    Timm module to allow loading any timm model as a feature extractor.

    This class provides a way to load a model from the timm library with pretrained weights,
    customize input channels with intelligent weight adaptation, and extract multi-scale 
    features for tasks like detection and segmentation.

    When features_only=True, returns a list of feature tensors at different scales.
    Use with Index module to select specific scales in YAML configurations.

    Attributes:
        m (nn.Module): The loaded timm model configured as a feature extractor.
        out_indices (tuple): Indices of stages to extract features from.
        channels (list): Output channel dimensions for each feature stage.
        strides (list): Spatial reduction factors for each feature stage.

    Args:
        model (str): Name of the timm model to load (e.g., 'convnext_base', 'efficientnet_b0').
        pretrained (bool, optional): Whether to load pretrained weights. Default is True.
        in_chans (int, optional): Number of input channels. Default is 3.
        features_only (bool, optional): If True, extracts multi-scale features as a list. Default is True.
        out_indices (tuple, optional): Indices of feature stages to output. Default is (1, 2, 3, 4).
        output_stride (int, optional): Output stride for the model. Default is 32.
        norm_layer (str, optional): Normalization layer to use. Default is None (model default).
        stem_mode (str, optional): How to handle non-3 channel inputs when pretrained=True:
            - 'auto': Automatically adjust stem - repeat for >3 channels, mean for <3 (default)
            - 'repeat': Repeat RGB weights across channels
            - 'mean': Average RGB weights across channels
            - 'kaiming': Reinitialize stem with Kaiming initialization
            Note: Ignored if in_chans=3 or pretrained=False
        freeze_stem (bool, optional): Whether to freeze stem weights. Default is False.
        freeze (bool, optional): Whether to freeze entire backbone. Default is False.
        pure_transformers (bool, optional): Whether model is a pure transformer. Default is True.
        drop_path_rate (bool, optional): Stochastic drop path rate in the backbone. Default is 0.15

    Example:
        YAML usage with Index to select specific feature scales:
        ```yaml
        backbone:
          - [-1, 1, Timm, ['convnext_base', True, 3, True, [1,2,3,4]]]  # Layer 0: [P2, P3, P4, P5]
        
        head:
          - [0, 1, Index, [3]]  # Layer 1: Select P5 (32x downsample)
          - [0, 1, Index, [2]]  # Layer 2: Select P4 (16x downsample)
          - [[1, 2], 1, Concat, [1]]  # Layer 3: Concatenate P5 and P4
        ```
    """

    def __init__(
        self,
        model: str,
        pretrained: bool = True,
        in_chans: int = 3,
        features_only: bool = True,
        out_indices: tuple = (1, 2, 3, 4),
        output_stride: int = 32,
        norm_layer: Optional[str] = None,
        stem_mode: str = 'auto',
        freeze_stem: bool = False,
        freeze: bool = False,
        pure_transformers: bool = True,
        dynamic_img_size: bool = False,
        drop_path_rate: float = 0.15,
        dropout: float = 0.0
    ):
        """
        Load the model from timm with specified configuration.

        Args:
            model (str): Name of the timm model to load.
            pretrained (bool): Whether to load pretrained weights.
            in_chans (int): Number of input channels (RGB=3, RGBD=4, grayscale=1, etc.).
            features_only (bool): Whether to extract multi-scale features.
            out_indices (tuple): Which feature stages to output (typically 1-4 for P2-P5).
            output_stride (int): Output stride for the model.
            norm_layer (str): Normalization layer override.
            stem_mode (str): Method for adapting pretrained weights to non-3 channel inputs.
            freeze_stem (bool): Whether to freeze stem weights.
            freeze (bool): Whether to freeze all backbone parameters.
            pure_transformers (bool): Whether the model is a pure transformer (affects output_stride usage).
            dynamic_img_size (bool): Allow dynamic input image sizes (disables strict size checking).
        """
        import timm  # scope for faster 'import ultralytics'

        super().__init__()
        
        # Store configuration first
        self.features_only = features_only
        self.out_indices = out_indices
        self.model_name = model
        self.drop_path_rate = drop_path_rate
        self.dynamic_img_size = dynamic_img_size
        
        # Try to create model with features_only first to validate out_indices
        try:
            # Handle non-standard input channels with pretrained weights
            if pretrained and in_chans != 3:
                # Load with 3 channels first to get pretrained weights
                model_kwargs = {
                    'pretrained': True,
                    'in_chans': 3,
                    'features_only': features_only,
                }
                
                if features_only:
                    model_kwargs['out_indices'] = out_indices
                    if not pure_transformers:
                        model_kwargs['output_stride'] = output_stride
                
                if norm_layer is not None:
                    model_kwargs['norm_layer'] = norm_layer
                
                if drop_path_rate > 0.0:
                    model_kwargs["drop_path_rate"] = drop_path_rate
                if dropout > 0.0:
                    model_kwargs["drop_rate"] = dropout
                
                self.m = timm.create_model(model, **model_kwargs)
                
                # Adapt the stem for different input channels
                self._adapt_stem(in_chans, stem_mode)
            else:
                # Standard loading (no stem adaptation needed)
                model_kwargs = {
                    'pretrained': pretrained,
                    'in_chans': in_chans,
                    'features_only': features_only,
                }
                
                if features_only:
                    model_kwargs['out_indices'] = out_indices
                    if not pure_transformers:
                        model_kwargs['output_stride'] = output_stride
                
                if norm_layer is not None:
                    model_kwargs['norm_layer'] = norm_layer
                
                if drop_path_rate > 0.0:
                    model_kwargs["drop_path_rate"] = drop_path_rate
                if dropout > 0.0:
                    model_kwargs["drop_rate"] = dropout
                
                self.m = timm.create_model(model, **model_kwargs)
                
        except Exception as e:
            print(f"Error creating model with features_only={features_only}, out_indices={out_indices}: {e}")
            if features_only:
                print(f"Attempting to create model WITHOUT features_only (will use custom feature extraction)...")
                # Try WITHOUT features_only - load as full model
                try:
                    model_kwargs = {
                        'pretrained': pretrained,
                        'in_chans': in_chans,
                        'features_only': False,  # KEY CHANGE: Load as full model
                    }
                    if norm_layer is not None:
                        model_kwargs['norm_layer'] = norm_layer
                    
                    self.m = timm.create_model(model, **model_kwargs)
                    print(f"✓ Model loaded as full model (not feature extractor)")
                    print(f"⚠ Warning: Will attempt to extract features using forward hooks")
                    
                    # Set up feature extraction using hooks
                    self._setup_feature_hooks(out_indices)
                    
                except Exception as e2:
                    raise ValueError(f"Failed to create model '{model}': {e2}")
            else:
                # features_only=False, just re-raise the error
                raise
        
        # Apply freezing
        if freeze:
            self._freeze_backbone()
        elif freeze_stem:
            self._freeze_stem()
        
        # Enable dynamic image size if requested (for Vision Transformers)
        if dynamic_img_size:
            self._enable_dynamic_img_size()
        
        # Get feature info with robust fallback
        self._extract_feature_info(in_chans)
    
    def _setup_feature_hooks(self, out_indices):
        """
        Set up forward hooks to extract intermediate features from models that don't support features_only.
        
        Args:
            out_indices (tuple): Indices of layers to extract features from.
        """
        self.feature_outputs = {}
        self.hooks = []
        
        # Get all modules
        modules = list(self.m.named_modules())
        
        # For transformers, we typically want to hook into the blocks
        target_modules = []
        for name, module in modules:
            # Look for transformer blocks or stages
            if 'block' in name.lower() or 'stage' in name.lower() or 'layer' in name.lower():
                # Skip nested modules
                if '.' not in name.split('block')[-1].split('stage')[-1].split('layer')[-1]:
                    target_modules.append((name, module))
        
        # If no blocks found, try to use sequential children
        if not target_modules:
            for i, (name, module) in enumerate(self.m.named_children()):
                target_modules.append((name, module))
        
        print(f"Found {len(target_modules)} potential feature extraction points")
        
        # Register hooks for requested indices
        for idx in out_indices:
            if idx < len(target_modules):
                name, module = target_modules[idx]
                print(f"  Hooking layer {idx}: {name}")
                
                def hook_fn(idx):
                    def fn(module, input, output):
                        self.feature_outputs[idx] = output
                    return fn
                
                handle = module.register_forward_hook(hook_fn(idx))
                self.hooks.append(handle)
        
        self._using_hooks = True
    
    def _enable_dynamic_img_size(self):
        """
        Enable dynamic image size support for Vision Transformers and other models
        that have strict image size requirements.
        """
        # Find and modify PatchEmbed modules to allow dynamic sizes
        modified_count = 0
        for name, module in self.m.named_modules():
            # Handle timm PatchEmbed
            if module.__class__.__name__ == 'PatchEmbed':
                if hasattr(module, 'strict_img_size'):
                    module.strict_img_size = False
                    modified_count += 1
                if hasattr(module, 'dynamic_img_pad'):
                    module.dynamic_img_pad = True
                # Set img_size to None to allow any size
                if hasattr(module, 'img_size'):
                    module.img_size = None
        
        if modified_count > 0:
            print(f"✓ Enabled dynamic image size for model '{self.model_name}' ({modified_count} PatchEmbed modules modified)")
    
    def _extract_feature_info(self, in_chans: int):
        """
        Extract feature information from the model with robust fallback mechanisms.
        
        Args:
            in_chans (int): Number of input channels.
        """
        # If using hooks (features_only=True but model doesn't support it), probe to get info
        if hasattr(self, '_using_hooks') and self._using_hooks:
            print("Using forward hooks for feature extraction, probing model...")
            self._probe_model_features(in_chans)
            return
        
        # If features_only=False, just probe to get single output info
        if not self.features_only:
            print(f"Model '{self.model_name}' loaded with features_only=False (single output mode)")
            self._probe_single_output(in_chans)
            return
        
        # Try to get feature info from timm's feature_info (features_only=True and supported)
        if hasattr(self.m, 'feature_info'):
            try:
                self.feature_info = self.m.feature_info
                
                # Try different methods to extract channels and strides
                if hasattr(self.feature_info, 'channels') and callable(self.feature_info.channels):
                    self.channels = self.feature_info.channels()
                elif hasattr(self.feature_info, 'info') and isinstance(self.feature_info.info, list):
                    # Older timm versions
                    self.channels = [info['num_chs'] for info in self.feature_info.info]
                elif isinstance(self.feature_info, list):
                    # Fallback: directly access as list
                    self.channels = [info['num_chs'] for info in self.feature_info]
                else:
                    raise AttributeError("Cannot extract channels from feature_info")
                
                if hasattr(self.feature_info, 'reduction') and callable(self.feature_info.reduction):
                    self.strides = self.feature_info.reduction()
                elif hasattr(self.feature_info, 'info') and isinstance(self.feature_info.info, list):
                    self.strides = [info['reduction'] for info in self.feature_info.info]
                elif isinstance(self.feature_info, list):
                    self.strides = [info['reduction'] for info in self.feature_info]
                else:
                    raise AttributeError("Cannot extract strides from feature_info")
                
                print(f"✓ Extracted feature info from model '{self.model_name}':")
                print(f"  Channels: {self.channels}")
                print(f"  Strides: {self.strides}")
                return
                
            except Exception as e:
                print(f"Warning: Could not extract feature info from feature_info attribute: {e}")
        
        # Fallback: probe the model with a dummy input
        print(f"Warning: Model '{self.model_name}' does not have feature_info")
        print("Probing with dummy input...")
        self._probe_model_features(in_chans)
    
    def _probe_single_output(self, in_chans: int):
        """
        Probe the model to get single output info (for features_only=False).
        
        Args:
            in_chans (int): Number of input channels.
        """
        import torch
        
        try:
            # Try common input sizes
            for size in [224, 256, 384, 512]:
                try:
                    dummy_input = torch.randn(1, in_chans, size, size)
                    
                    # Run forward pass
                    was_training = self.m.training
                    self.m.eval()
                    with torch.no_grad():
                        output = self.m(dummy_input)
                    if was_training:
                        self.m.train()
                    
                    # Get output shape - should be single tensor
                    if isinstance(output, (list, tuple)):
                        # Take last output if multiple
                        output = output[-1]
                    
                    # Determine channels based on output shape
                    if len(output.shape) == 4:  # (B, C, H, W) or (B, H, W, C)
                        if output.shape[1] < output.shape[-1]:  # (B, H, W, C)
                            self.channels = [output.shape[-1]]
                            self.strides = [size // output.shape[1]]
                        else:  # (B, C, H, W)
                            self.channels = [output.shape[1]]
                            self.strides = [size // output.shape[2]]
                    elif len(output.shape) == 2:  # (B, num_classes) - classification output
                        self.channels = [output.shape[1]]
                        self.strides = [size]  # Full reduction
                    elif len(output.shape) == 3:  # (B, N, C) - transformer output
                        self.channels = [output.shape[-1]]
                        import math
                        h = int(math.sqrt(output.shape[1]))
                        self.strides = [size // h if h > 0 else size]
                    
                    self.feature_info = {
                        'channels': self.channels,
                        'reduction': self.strides,
                        'method': 'probed_single_output',
                        'input_size': size
                    }
                    
                    print(f"✓ Single output info (input_size={size}):")
                    print(f"  Channels: {self.channels}")
                    print(f"  Strides: {self.strides}")
                    return
                    
                except Exception as e:
                    if size == 512:  # Last attempt
                        raise e
                    continue
                    
        except Exception as e:
            print(f"✗ Error probing single output: {e}")
            self.feature_info = None
            self.channels = None
            self.strides = None
    
    def _probe_model_features(self, in_chans: int):
        """
        Probe the model with a dummy input to determine output channels and strides.
        
        Args:
            in_chans (int): Number of input channels.
        """
        import torch
        
        try:
            # Create a dummy input (must match expected input size for some models)
            # Try common sizes
            for size in [224, 256, 384, 512]:
                try:
                    dummy_input = torch.randn(1, in_chans, size, size)
                    
                    # Clear previous outputs if using hooks
                    if hasattr(self, '_using_hooks') and self._using_hooks:
                        self.feature_outputs = {}
                    
                    # Run forward pass
                    was_training = self.m.training
                    self.m.eval()
                    with torch.no_grad():
                        outputs = self.m(dummy_input)
                    if was_training:
                        self.m.train()
                    
                    # Extract features from hooks if we're using them
                    if hasattr(self, '_using_hooks') and self._using_hooks:
                        # Sort by index
                        sorted_indices = sorted(self.feature_outputs.keys())
                        outputs = [self.feature_outputs[idx] for idx in sorted_indices]
                    
                    if isinstance(outputs, (list, tuple)):
                        self.channels = []
                        self.strides = []
                        for out in outputs:
                            # Handle transformer outputs (B, H, W, C) or conv outputs (B, C, H, W)
                            if len(out.shape) == 4:
                                if out.shape[1] < out.shape[-1]:  # (B, H, W, C)
                                    self.channels.append(out.shape[-1])
                                    self.strides.append(size // out.shape[1])
                                else:  # (B, C, H, W)
                                    self.channels.append(out.shape[1])
                                    self.strides.append(size // out.shape[2])
                            elif len(out.shape) == 3:  # (B, N, C) - flatten tokens
                                self.channels.append(out.shape[-1])
                                # Estimate stride from number of tokens
                                import math
                                h = int(math.sqrt(out.shape[1]))
                                self.strides.append(size // h if h > 0 else size)
                    else:
                        # Single output
                        if len(outputs.shape) == 4:
                            if outputs.shape[1] < outputs.shape[-1]:  # (B, H, W, C)
                                self.channels = [outputs.shape[-1]]
                                self.strides = [size // outputs.shape[1]]
                            else:  # (B, C, H, W)
                                self.channels = [outputs.shape[1]]
                                self.strides = [size // outputs.shape[2]]
                        elif len(outputs.shape) == 3:  # (B, N, C)
                            self.channels = [outputs.shape[-1]]
                            import math
                            h = int(math.sqrt(outputs.shape[1]))
                            self.strides = [size // h if h > 0 else size]
                    
                    self.feature_info = {
                        'channels': self.channels,
                        'reduction': self.strides,
                        'method': 'probed',
                        'input_size': size
                    }
                    
                    print(f"✓ Probed features successfully (input_size={size}):")
                    print(f"  Channels: {self.channels}")
                    print(f"  Strides: {self.strides}")
                    return
                    
                except Exception as e:
                    if size == 512:  # Last attempt failed
                        raise e
                    continue
            
        except Exception as e:
            # Last resort: set reasonable defaults or raise error
            print(f"✗ Error: Could not probe model features: {e}")
            print(f"Model '{self.model_name}' may not support the requested configuration.")
            
            # Set None to trigger error in parse_model with helpful message
            self.feature_info = None
            self.channels = None
            self.strides = None
    
    def _adapt_stem(self, in_chans: int, stem_mode: str):
        """
        Adapt the stem layer for different input channels while preserving pretrained weights.
        
        Args:
            in_chans (int): Target number of input channels.
            stem_mode (str): Method for weight adaptation.
        """
        # Find the first conv layer
        first_conv = None
        for name, module in self.m.named_modules():
            if isinstance(module, nn.Conv2d):
                first_conv = (name, module)
                break
        
        if first_conv is None:
            print(f"Warning: Could not find first conv layer in model '{self.model_name}'")
            print("Model may be a pure transformer or have non-standard architecture")
            return
        
        name, conv = first_conv
        old_weight = conv.weight.data
        
        # Create new conv layer with target input channels
        new_conv = nn.Conv2d(
            in_chans,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=conv.bias is not None
        )
        
        # Handle weight initialization based on stem_mode
        with torch.no_grad():
            if stem_mode == 'repeat' or (stem_mode == 'auto' and in_chans > 3):
                # Repeat weights across channels (good for RGBD, multispectral, etc.)
                repeat_times = (in_chans + 2) // 3
                repeated = old_weight.repeat(1, repeat_times, 1, 1)
                new_conv.weight.data = repeated[:, :in_chans, :, :]
            elif stem_mode == 'mean' or (stem_mode == 'auto' and in_chans < 3):
                # Average weights (good for grayscale)
                new_conv.weight.data = old_weight.mean(dim=1, keepdim=True).repeat(1, in_chans, 1, 1)
            elif stem_mode == 'kaiming':
                # Reinitialize from scratch
                nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
            
            if conv.bias is not None:
                new_conv.bias.data = conv.bias.data
        
        # Replace the conv layer in the model
        parent_name = '.'.join(name.split('.')[:-1]) if '.' in name else ''
        attr_name = name.split('.')[-1]
        
        if parent_name:
            parent = dict(self.m.named_modules())[parent_name]
        else:
            parent = self.m
        
        setattr(parent, attr_name, new_conv)
        print(f"✓ Adapted stem layer '{name}' for {in_chans} input channels (mode: {stem_mode})")
    
    def _freeze_stem(self):
        """Freeze the stem/first few layers only."""
        frozen = False
        for name, param in self.m.named_parameters():
            if 'stem' in name.lower() or 'conv1' in name.lower() or name.startswith('0.'):
                param.requires_grad = False
                frozen = True
            elif frozen:
                break
    
    def _freeze_backbone(self):
        """Freeze all parameters in the backbone."""
        for param in self.m.parameters():
            param.requires_grad = False
        # Set model to eval mode to freeze batch norm statistics
        self.m.eval()
    
    def train(self, mode: bool = True):
        """
        Override train mode to keep backbone in eval if frozen.
        
        Args:
            mode (bool): Whether to set training mode (True) or evaluation mode (False).
        
        Returns:
            self
        """
        super().train(mode)
        # Keep backbone in eval mode if any parameters are frozen
        if any(not p.requires_grad for p in self.m.parameters()):
            self.m.eval()
        return self

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            (torch.Tensor | List[torch.Tensor]): 
                - If features_only=True with native support: List of feature tensors at different scales
                - If features_only=True with hooks: List of feature tensors extracted via hooks
                - If features_only=False: Single output tensor (unchanged from model output)
        """
        if hasattr(self, '_using_hooks') and self._using_hooks:
            # Using hooks for feature extraction
            self.feature_outputs = {}
            _ = self.m(x)
            # Return features in order of out_indices
            sorted_indices = sorted(self.feature_outputs.keys())
            return [self.feature_outputs[idx] for idx in sorted_indices]
        
        # Normal forward pass (either features_only=True with native support, or features_only=False)
        return self.m(x)
    
    def get_channel_info(self) -> Optional[dict]:
        """
        Get information about output channels and strides.

        Returns:
            (dict | None): Dictionary containing 'channels' and 'strides' lists, or None if not available.
        """
        if self.channels is not None and self.strides is not None:
            return {
                'channels': self.channels,
                'strides': self.strides,
                'out_indices': self.out_indices
            }
        return None



# class DinoV3Backbone(nn.Module):
#     """
#     Use to load any dinov3 backbone saved as a .pt file into as a module in the framework. Can be used to extract intermediate features.

#     Attributes:
#         m (nn.module): The loaded .pt file as a torch module
#     """
#     def __init__(self, filepath: str, in_channels: int = 3, out_channels: int = 1024, freeze: bool = True, out_indices: int | list[int] = [23], weights_only: bool = False):
#         """
#         """
#         try:
#             device = 0 if torch.cuda.is_available() else "cpu"
#             self.m = torch.load(filepath, weights_only = weights_only, map_location = torch.device(device))
#             if isinstance(out_indices, int):
#                 self.out_indices = range(out_indices)
#             self.out_indices = out_indices
#             self.out_channels = out_channels
#             if freeze:
#                 for p in self.m.parameters():
#                     p.requires_grad = False
#             self.m.eval()
#         except Exception as e:
#             raise e
    
#     def forward(self, x):
#         feats = self.m.get_intermediate_layers(x, n = self.out_indices, reshape = True, norm = True)
#         # feats_small = [i.view()]]
#         return feats

class MaxMBConv(nn.Module):
    """
    Mobile Inverted Bottleneck with Squeeze-and-Excitation.

    This module implements a mobile inverted bottleneck block with expansion, depthwise convolution,
    SE attention, and projection. Supports residual connection when input and output dimensions match.

    Attributes:
        expand (nn.Module): Expansion layer (1x1 conv or identity).
        dw (DWConv): Depthwise convolution with specified stride.
        se (SE): Squeeze-and-Excitation block.
        project (Conv): Projection layer (1x1 conv with linear activation).
        add (bool): Whether to use residual connection.
        drop (nn.Module): Dropout layer.

    Examples:
        >>> mbconv = MaxMBConv(c1=64, c2=128, s=2, expand=4.0)
        >>> x = torch.randn(1, 64, 56, 56)
        >>> out = mbconv(x)
        >>> print(out.shape)
        torch.Size([1, 128, 28, 28])
    """

    def __init__(self, c1: int, c2: int, s: int = 1, expand: float = 4.0, se_rd: int = 4, drop: float = 0.0):
        """
        Initialize Mobile Inverted Bottleneck block.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            s (int): Stride for depthwise convolution. Default is 1.
            expand (float): Expansion ratio for hidden dimension. Default is 4.0.
            se_rd (int): Reduction ratio for SE block. Default is 4.
            drop (float): Dropout probability. Default is 0.0.
        """
        super().__init__()
        ce = int(round(c1 * expand))
        self.expand = Conv(c1, ce, k=1, s=1) if ce != c1 else nn.Identity()
        self.dw = DWConv(ce, ce, k=3, s=s)  # depthwise + BN + act
        self.se = SE(ce, rd=se_rd)
        # project with linear act (disable activation inside Conv)
        self.project = Conv(ce, c2, k=1, s=1, act=False)
        self.add = (s == 1) and (c1 == c2)
        self.drop = nn.Dropout2d(p=drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        """
        Forward pass through MBConv block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C1, H, W).

        Returns:
            (torch.Tensor): Output tensor of shape (B, C2, H//s, W//s).
        """
        y = self.expand(x)
        y = self.dw(y)
        y = self.se(y)
        y = self.project(y)
        y = self.drop(y)
        return x + y if self.add else y

class WindowSA(nn.Module):
    """
    Windowed Self-Attention using TransformerEncoderLayer.

    Divides input into non-overlapping windows and applies self-attention within each window,
    enabling efficient local attention computation.

    Attributes:
        proj (nn.Module): Input projection layer (1x1 conv or identity).
        win (int): Window size.
        enc (TransformerEncoderLayer): Transformer encoder for attention computation.
        c2 (int): Output channels.

    Examples:
        >>> wsa = WindowSA(c1=128, c2=128, win=7, heads=8)
        >>> x = torch.randn(1, 128, 56, 56)
        >>> out = wsa(x)
        >>> print(out.shape)
        torch.Size([1, 128, 56, 56])
    """

    def __init__(
        self, c1: int, c2: int, win: int = 7, heads: int = 8, mlp: float = 4.0, dropout: float = 0.0, pre_norm: bool = True
    ):
        """
        Initialize Windowed Self-Attention module.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            win (int): Window size for local attention. Default is 7.
            heads (int): Number of attention heads. Default is 8.
            mlp (float): MLP expansion ratio. Default is 4.0.
            dropout (float): Dropout probability. Default is 0.0.
            pre_norm (bool): Whether to use pre-normalization. Default is True.
        """
        super().__init__()
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        self.win = win
        self.enc = TransformerEncoderLayer(
            c1=c2, cm=int(c2 * mlp), num_heads=heads, dropout=dropout, act=nn.GELU(), normalize_before=pre_norm
        )  # batch_first=True in Ultralytics impl
        self.c2 = c2

    def forward(self, x):
        """
        Forward pass with windowed self-attention.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            (torch.Tensor): Output tensor of shape (B, C, H, W) after windowed attention.
        """
        x = self.proj(x)
        B, C, H, W = x.shape
        w = self.win
        # pad to multiple of w
        pad_h = (w - H % w) % w
        pad_w = (w - W % w) % w
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            H, W = H + pad_h, W + pad_w
        # (B, C, H, W) -> windows: (B * (H/w) * (W/w), w*w, C)
        x_ = x.view(B, C, H // w, w, W // w, w).permute(0, 2, 4, 3, 5, 1).contiguous()
        tokens = x_.view(-1, w * w, C)  # batch_first=True: (N, L, C)
        tokens = self.enc(tokens)  # attn + FFN
        # merge back
        x_ = tokens.view(B, H // w, W // w, w, w, C).permute(0, 5, 1, 3, 2, 4).contiguous()
        y = x_.view(B, C, H, W)
        # remove padding
        return y[:, :, : H - pad_h if pad_h else None, : W - pad_w if pad_w else None]


class GridSA(nn.Module):
    """
    Grid Self-Attention for sparse global attention.

    Divides the spatial dimensions into a grid of groups (g x g), where each group attends
    over tokens in the same grid position across all groups, enabling efficient global mixing.

    Attributes:
        proj (nn.Module): Input projection layer (1x1 conv or identity).
        g (int): Grid size (number of groups per dimension).
        enc (TransformerEncoderLayer): Transformer encoder for attention computation.

    Examples:
        >>> gsa = GridSA(c1=128, c2=128, grid=7, heads=8)
        >>> x = torch.randn(1, 128, 56, 56)
        >>> out = gsa(x)
        >>> print(out.shape)
        torch.Size([1, 128, 56, 56])
    """

    def __init__(
        self, c1: int, c2: int, grid: int = 7, heads: int = 8, mlp: float = 4.0, dropout: float = 0.0, pre_norm: bool = True
    ):
        """
        Initialize Grid Self-Attention module.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            grid (int): Grid size for grouping. Default is 7.
            heads (int): Number of attention heads. Default is 8.
            mlp (float): MLP expansion ratio. Default is 4.0.
            dropout (float): Dropout probability. Default is 0.0.
            pre_norm (bool): Whether to use pre-normalization. Default is True.
        """
        super().__init__()
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        self.g = grid
        self.enc = TransformerEncoderLayer(
            c1=c2, cm=int(c2 * mlp), num_heads=heads, dropout=dropout, act=nn.GELU(), normalize_before=pre_norm
        )

    def forward(self, x):
        """
        Forward pass with grid self-attention.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            (torch.Tensor): Output tensor of shape (B, C, H, W) after grid attention.
        """
        x = self.proj(x)
        B, C, H, W = x.shape
        g = self.g
        # pad so H,W are multiples of g
        pad_h = (g - H % g) % g
        pad_w = (g - W % g) % g
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            H, W = H + pad_h, W + pad_w
        # (B, C, H, W) -> (B*g*g, (H/g)*(W/g), C)
        x_ = x.view(B, C, g, H // g, g, W // g).permute(0, 2, 4, 3, 5, 1).contiguous()
        tokens = x_.view(B * g * g, (H // g) * (W // g), C)
        tokens = self.enc(tokens)
        # merge back
        x_ = tokens.view(B, g, g, H // g, W // g, C).permute(0, 5, 1, 3, 2, 4).contiguous()
        y = x_.reshape(B, C, H, W)
        # remove padding
        return y[:, :, : H - pad_h if pad_h else None, : W - pad_w if pad_w else None]


class MaxViTBlock(nn.Module):
    """
    MaxViT block combining MBConv with multi-axis attention.

    This block sequentially applies: Mobile Inverted Bottleneck → Window Self-Attention → Grid Self-Attention,
    providing both local and global feature interactions. Downsampling is applied via stride in MBConv.

    Attributes:
        mb (MaxMBConv): Mobile inverted bottleneck layer.
        win (WindowSA): Window self-attention layer.
        grid (GridSA): Grid self-attention layer.

    Examples:
        >>> block = MaxViTBlock(c1=128, c2=256, s=2, win=7, grid=7, heads=8)
        >>> x = torch.randn(1, 128, 56, 56)
        >>> out = block(x)
        >>> print(out.shape)
        torch.Size([1, 256, 28, 28])
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        win: int = 7,
        grid: int = 7,
        heads: int = 8,
        expand: float = 4.0,
        se_rd: int = 4,
        mlp: float = 4.0,
        s: int = 1,
        dropout: float = 0.0,
        pre_norm: bool = True,
    ):
        """
        Initialize MaxViT block.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            win (int): Window size for local attention. Default is 7.
            grid (int): Grid size for global attention. Default is 7.
            heads (int): Number of attention heads. Default is 8.
            expand (float): Expansion ratio for MBConv. Default is 4.0.
            se_rd (int): Reduction ratio for SE block. Default is 4.
            mlp (float): MLP expansion ratio for attention blocks. Default is 4.0.
            s (int): Stride for downsampling (applied in MBConv). Default is 1.
            dropout (float): Dropout probability. Default is 0.0.
            pre_norm (bool): Whether to use pre-normalization in attention. Default is True.
        """
        super().__init__()
        self.mb = MaxMBConv(c1, c2, s=s, expand=expand, se_rd=se_rd, drop=dropout)
        self.win = WindowSA(c2, c2, win=win, heads=heads, mlp=mlp, dropout=dropout, pre_norm=pre_norm)
        self.grid = GridSA(c2, c2, grid=grid, heads=heads, mlp=mlp, dropout=dropout, pre_norm=pre_norm)

    def forward(self, x):
        """
        Forward pass through MaxViT block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C1, H, W).

        Returns:
            (torch.Tensor): Output tensor of shape (B, C2, H//s, W//s) after MBConv and multi-axis attention.
        """
        x = self.mb(x)
        x = self.win(x)
        x = self.grid(x)
        return x