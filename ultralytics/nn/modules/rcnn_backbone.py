# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Backbone modules used by the native RCNN family.

These wrappers expose multi-scale feature maps plus a `channels` attribute so they
can be consumed by the existing Ultralytics YAML parser in the same way as the
`Timm` backbone wrapper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .block import DropPath

__all__ = ("ResNetBackbone", "UnravelNetBackbone")


def _build_norm(norm_cfg: dict | Callable | None, channels: int) -> nn.Module:
    """Build a small subset of MM-style norm configs used by the reference models."""
    if norm_cfg is None:
        return nn.Identity()
    if isinstance(norm_cfg, dict):
        norm_type = norm_cfg.get("type", "BN").upper()
        if norm_type == "BN":
            layer = nn.BatchNorm2d(channels)
        elif norm_type == "GN":
            groups = int(norm_cfg.get("num_groups", 32))
            layer = nn.GroupNorm(groups, channels)
        else:
            raise ValueError(f"Unsupported norm layer: {norm_type}")
        for p in layer.parameters():
            p.requires_grad = bool(norm_cfg.get("requires_grad", True))
        return layer
    if isinstance(norm_cfg, type):
        return norm_cfg(channels)
    if callable(norm_cfg):
        return norm_cfg(channels)
    raise TypeError(f"Unsupported norm config: {type(norm_cfg)!r}")


class ResNetBackbone(nn.Module):
    """Torchvision ResNet wrapper returning C2-C5 feature maps."""

    def __init__(
        self,
        in_chans: int = 3,
        model: str = "resnet50",
        weights: str | None = "DEFAULT",
        frozen_stages: int = 1,
        norm_eval: bool = True,
        freeze_norm_affine: bool = False,
    ):
        super().__init__()
        import torchvision

        self.channels = [256, 512, 1024, 2048]
        self.norm_eval = norm_eval
        self.freeze_norm_affine = bool(freeze_norm_affine)
        self.frozen_stages = frozen_stages

        weight_enum = None
        if weights not in {None, False, "None"} and hasattr(torchvision.models, "get_model_weights"):
            try:
                weight_enum = torchvision.models.get_model_weights(model).DEFAULT if weights == "DEFAULT" else weights
            except Exception:
                weight_enum = None

        backbone = torchvision.models.get_model(model, weights=weight_enum)
        if in_chans != 3:
            old_conv = backbone.conv1
            new_conv = nn.Conv2d(
                in_chans,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            with torch.no_grad():
                if old_conv.weight.shape[1] == 3:
                    if in_chans > 3:
                        repeats = (in_chans + 2) // 3
                        weight = old_conv.weight.repeat(1, repeats, 1, 1)[:, :in_chans]
                        weight *= 3.0 / float(in_chans)
                    else:
                        weight = old_conv.weight[:, :in_chans].mean(1, keepdim=True).repeat(1, in_chans, 1, 1)
                    new_conv.weight.copy_(weight)
            backbone.conv1 = new_conv

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self._freeze_stages()

    def _freeze_stages(self):
        stages = [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]
        for idx, module in enumerate(stages):
            if idx <= self.frozen_stages:
                module.eval()
                for p in module.parameters():
                    p.requires_grad = False
        if self.freeze_norm_affine:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
                    for parameter in module.parameters():
                        parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_stages()
        if self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c2, c3, c4, c5


class _BlurPool(nn.Module):
    """Minimal anti-aliased blur-pool used by UnravelNet."""

    def __init__(self, channels: int, stride: int = 2):
        super().__init__()
        filt = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float32)
        kernel = (filt[:, None] * filt[None, :])
        kernel = kernel / kernel.sum()
        self.register_buffer("kernel", kernel[None, None].repeat(channels, 1, 1, 1), persistent=False)
        self.channels = channels
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        return F.conv2d(x, self.kernel, stride=self.stride, padding=1, groups=self.channels)


class _DRFD(nn.Module):
    def __init__(self, dim: int, norm_layer: dict | Callable | None, act_layer: type[nn.Module]):
        super().__init__()
        outdim = dim * 2
        self.conv = nn.Conv2d(dim, outdim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.conv_c = nn.Conv2d(outdim, outdim, kernel_size=3, stride=2, padding=1, groups=outdim)
        self.act_c = act_layer()
        self.norm_c = _build_norm(norm_layer, outdim)
        self.max_m = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.norm_m = _build_norm(norm_layer, outdim)
        self.fusion = nn.Conv2d(outdim * 2, outdim, kernel_size=1, stride=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        max_branch = self.norm_m(self.max_m(x))
        conv_branch = self.norm_c(self.act_c(self.conv_c(x)))
        return self.fusion(torch.cat([conv_branch, max_branch], dim=1))


class _PA(nn.Module):
    def __init__(self, dim: int, norm_layer: dict | Callable | None, act_layer: type[nn.Module]):
        super().__init__()
        self.p_conv = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1, bias=False),
            _build_norm(norm_layer, dim * 4),
            act_layer(),
            nn.Conv2d(dim * 4, dim, 1, bias=False),
        )
        self.gate_fn = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gate_fn(self.p_conv(x))


class _LA(nn.Module):
    def __init__(self, dim: int, norm_layer: dict | Callable | None, act_layer: type[nn.Module]):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            _build_norm(norm_layer, dim),
            act_layer(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class _MRA(nn.Module):
    def __init__(self, channel: int, att_kernel: int, norm_layer: dict | Callable | None):
        super().__init__()
        att_padding = att_kernel // 2
        self.channel = channel
        self.gate_fn = nn.Sigmoid()
        self.max_m1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.max_m2 = _BlurPool(channel, stride=3)
        self.h_att1 = nn.Conv2d(channel, channel, (att_kernel, 3), 1, (att_padding, 1), groups=channel, bias=False)
        self.v_att1 = nn.Conv2d(channel, channel, (3, att_kernel), 1, (1, att_padding), groups=channel, bias=False)
        self.h_att2 = nn.Conv2d(channel, channel, (att_kernel, 3), 1, (att_padding, 1), groups=channel, bias=False)
        self.v_att2 = nn.Conv2d(channel, channel, (3, att_kernel), 1, (1, att_padding), groups=channel, bias=False)
        self.norm = _build_norm(norm_layer, channel)

    @staticmethod
    def _h_transform(x: Tensor) -> Tensor:
        shape = x.size()
        x = F.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        return x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)

    @staticmethod
    def _inv_h_transform(x: Tensor) -> Tensor:
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = F.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        return x[..., : shape[-2]]

    @staticmethod
    def _v_transform(x: Tensor) -> Tensor:
        return _MRA._h_transform(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)

    @staticmethod
    def _inv_v_transform(x: Tensor) -> Tensor:
        return _MRA._inv_h_transform(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)

    def forward(self, x: Tensor) -> Tensor:
        x_tem = self.max_m2(self.max_m1(x))
        x_h1 = self.h_att1(x_tem)
        x_w1 = self.v_att1(x_tem)
        x_h2 = self._inv_h_transform(self.h_att2(self._h_transform(x_tem)))
        x_w2 = self._inv_v_transform(self.v_att2(self._v_transform(x_tem)))
        att = self.norm(x_h1 + x_w1 + x_h2 + x_w2)
        return x * F.interpolate(self.gate_fn(att), size=x.shape[-2:], mode="nearest")


class _EdgeEnhance(nn.Module):
    def __init__(self, channel: int, norm_layer: dict | Callable | None, act_layer: type[nn.Module], gaussian=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channel, 64, 1),
            _build_norm(norm_layer, 64),
            act_layer(inplace=True),
            nn.Conv2d(64, 256, 3, stride=1, padding=1, bias=False),
            _build_norm(norm_layer, 256),
            act_layer(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            _build_norm(norm_layer, 256),
            act_layer(inplace=True),
            nn.Conv2d(256, channel, 1),
            _build_norm(norm_layer, channel),
        )
        if gaussian:
            kernel = torch.tensor([[1.0, 0.3, 0.2], [0.3, 2.0, 0.5], [0.2, 0.5, 3.0]], dtype=torch.float32)
        else:
            kernel = torch.tensor([[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]], dtype=torch.float32)
        self.register_buffer("kernel_x", kernel[None, None], persistent=False)
        self.register_buffer("kernel_y", kernel.t()[None, None], persistent=False)
        self.norm = _build_norm(norm_layer, channel)
        self.act = act_layer(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        kx = self.kernel_x.repeat(x.shape[1], 1, 1, 1)
        ky = self.kernel_y.repeat(x.shape[1], 1, 1, 1)
        edges_x = F.conv2d(x, kx, padding=1, groups=x.shape[1])
        edges_y = F.conv2d(x, ky, padding=1, groups=x.shape[1])
        edge = torch.sqrt(edges_x.square() + edges_y.square() + 1e-6)
        edge = self.act(self.norm(edge))
        return self.block(x + edge)


class _BGA(nn.Module):
    def __init__(self, channel: int, norm_layer: dict | Callable | None, act_layer: type[nn.Module]):
        super().__init__()
        kernel = int(abs((torch.log2(torch.tensor(float(channel))) + 1) / 2))
        kernel = int(kernel if kernel % 2 else kernel + 1)
        self.conv2d = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1, bias=False),
            _build_norm(norm_layer, channel),
            act_layer(inplace=True),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=kernel, padding=(kernel - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.norm = _build_norm(norm_layer, channel)

    def forward(self, x: Tensor, att: Tensor) -> Tensor:
        att = self.conv2d(x * att + x)
        wei = self.avg_pool(att)
        wei = self.conv1d(wei.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        wei = self.sigmoid(wei)
        return self.norm(x + att * wei)


class _UnravelBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        stage: int,
        att_kernel: int,
        mlp_ratio: float,
        drop_path: float,
        act_layer: type[nn.Module],
        norm_layer: dict | Callable | None,
    ):
        super().__init__()
        dim_split = dim // 4
        mlp_hidden = int(dim * mlp_ratio)
        self.stage = stage
        self.pa = _PA(dim_split, norm_layer, act_layer)
        self.la = _LA(dim_split, norm_layer, act_layer)
        self.mra = _MRA(dim_split, att_kernel, norm_layer)
        self.bga = _BGA(dim_split, norm_layer, act_layer)
        self.edge = _EdgeEnhance(dim_split, norm_layer, act_layer, gaussian=stage > 0)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mlp_hidden, 1, bias=False),
            _build_norm(norm_layer, mlp_hidden),
            act_layer(),
            nn.Conv2d(mlp_hidden, dim, 1, bias=False),
        )
        self.norm = _build_norm(norm_layer, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        x1 = x1 + self.pa(x1)
        x2 = self.la(x2)
        x3 = self.mra(x3)
        x4 = self.bga(x4, self.edge(x4))
        x = torch.cat((x1, x2, x3, x4), dim=1)
        return shortcut + self.norm(self.drop_path(self.mlp(x)))


class _BasicStage(nn.Module):
    def __init__(
        self,
        dim: int,
        stage: int,
        depth: int,
        att_kernel: int,
        mlp_ratio: float,
        drop_path: list[float],
        norm_layer: dict | Callable | None,
        act_layer: type[nn.Module],
    ):
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                _UnravelBlock(
                    dim=dim,
                    stage=stage,
                    att_kernel=att_kernel,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


class _Stem(nn.Module):
    def __init__(self, in_chans: int, stem_dim: int, norm_layer: dict | Callable | None):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, stem_dim, kernel_size=4, stride=4, bias=False)
        self.norm = _build_norm(norm_layer, stem_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.proj(x))


class UnravelNetBackbone(nn.Module):
    """Standalone UnravelNet backbone ported from the local LEGNet reference."""

    def __init__(
        self,
        in_chans: int = 3,
        stem_dim: int = 64,
        depths: tuple[int, int, int, int] = (1, 4, 4, 2),
        att_kernel: tuple[int, int, int, int] = (11, 11, 11, 11),
        norm_layer: dict | Callable | None = None,
        act_layer: type[nn.Module] = nn.ReLU,
        mlp_ratio: float = 2.0,
        stem_norm: bool = True,
        drop_path_rate: float = 0.1,
        pretrained: str | None = None,
    ):
        super().__init__()
        self.channels = [stem_dim, stem_dim * 2, stem_dim * 4, stem_dim * 8]
        self.stem = _Stem(in_chans=in_chans, stem_dim=stem_dim, norm_layer=norm_layer if stem_norm else None)

        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        stages = []
        start = 0
        for i_stage in range(len(depths)):
            dim = stem_dim * (2**i_stage)
            stage = _BasicStage(
                dim=dim,
                stage=i_stage,
                depth=depths[i_stage],
                att_kernel=att_kernel[i_stage],
                mlp_ratio=mlp_ratio,
                drop_path=dpr[start : start + depths[i_stage]],
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
            start += depths[i_stage]
            stages.append(stage)
            if i_stage < len(depths) - 1:
                stages.append(_DRFD(dim=dim, norm_layer=norm_layer, act_layer=act_layer))
        self.stages = nn.Sequential(*stages)
        self.out_indices = [0, 2, 4, 6]
        for idx, ch in zip(self.out_indices, self.channels):
            self.add_module(f"norm{idx}", _build_norm(norm_layer, ch))

        self.apply(self._init_weights)
        if pretrained:
            self.load_pretrained(pretrained)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            if getattr(m, "weight", None) is not None:
                nn.init.constant_(m.weight, 1.0)
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias, 0)

    def load_pretrained(self, path: str | Path):
        state = torch.load(str(path), map_location="cpu")
        if isinstance(state, dict):
            state = state.get("state_dict", state.get("model", state))
        if any(k.startswith("module.") for k in state):
            state = {k.removeprefix("module."): v for k, v in state.items()}
        self.load_state_dict(state, strict=False)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        x = self.stem(x)
        outputs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.out_indices:
                outputs.append(getattr(self, f"norm{idx}")(x))
        return tuple(outputs)
