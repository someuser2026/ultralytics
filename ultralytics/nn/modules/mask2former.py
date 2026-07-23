# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Mask2Former instance segmentation head.

This module ports the Mask2Former pixel decoder and multi-scale masked transformer
decoder into Ultralytics without a Detectron2 runtime dependency. The backbone is
intentionally external: YAML selects the feature maps and this head consumes them.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.init import constant_, normal_, xavier_uniform_

from .transformer import MLP, MSDeformAttn


def _c2_xavier_fill(module: nn.Module) -> None:
    """Detectron2/fvcore-style Conv2d Xavier init used by Mask2Former."""
    if isinstance(module, nn.Conv2d):
        xavier_uniform_(module.weight)
        if module.bias is not None:
            constant_(module.bias, 0.0)


def _get_norm(norm: str | None, channels: int) -> nn.Module | None:
    """Return the normalization layer used by the reference configs."""
    if norm in {None, ""}:
        return None
    if str(norm).upper() == "GN":
        return nn.GroupNorm(32, channels)
    raise ValueError(f"Unsupported Mask2Former norm={norm!r}. Only ''/None and 'GN' are supported.")


class Conv2dNormActivation(nn.Module):
    """Small Detectron2 Conv2d replacement with optional norm and activation."""

    def __init__(
        self,
        c1: int,
        c2: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        norm: str | None = None,
        activation: Any = None,
    ):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size, stride=stride, padding=padding, bias=bias)
        self.norm = _get_norm(norm, c2)
        self.activation = activation
        _c2_xavier_fill(self.conv)

    @property
    def weight(self) -> nn.Parameter:
        """Expose the wrapped conv weight for compatibility with tests/checkpoint mapping."""
        return self.conv.weight

    @property
    def bias(self) -> nn.Parameter | None:
        """Expose the wrapped conv bias for compatibility with tests/checkpoint mapping."""
        return self.conv.bias

    def forward(self, x: Tensor) -> Tensor:
        """Apply conv, optional norm, and optional activation."""
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class PositionEmbeddingSine(nn.Module):
    """2D sine/cosine positional embedding used by Mask2Former."""

    def __init__(self, num_pos_feats: int = 64, temperature: int = 10000, normalize: bool = False, scale: float | None = None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.scale = 2 * math.pi if scale is None else scale

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Return positional embeddings with shape BCHW."""
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


class Mask2FormerMSDeformAttn(MSDeformAttn):
    """Adapter that keeps the reference Mask2Former MSDeformAttn call signature."""

    def __init__(self, d_model: int = 256, n_levels: int = 4, n_heads: int = 8, n_points: int = 4):
        super().__init__(d_model=d_model, n_levels=n_levels, n_heads=n_heads, n_points=n_points)
        self.im2col_step = 128

    def forward(
        self,
        query: Tensor,
        reference_points: Tensor,
        input_flatten: Tensor,
        input_spatial_shapes: Tensor,
        input_level_start_index: Tensor | None = None,
        input_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Run multi-scale deformable attention."""
        del input_level_start_index
        return super().forward(query, reference_points, input_flatten, input_spatial_shapes, input_padding_mask)


class MSDeformAttnTransformerEncoderLayer(nn.Module):
    """Mask2Former deformable encoder layer."""

    def __init__(
        self,
        d_model: int = 256,
        d_ffn: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        n_levels: int = 4,
        n_heads: int = 8,
        n_points: int = 4,
    ):
        super().__init__()
        self.self_attn = Mask2FormerMSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor: Tensor, pos: Tensor | None) -> Tensor:
        """Add positional embedding when present."""
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src: Tensor) -> Tensor:
        """Apply the reference FFN block."""
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        return self.norm2(src)

    def forward(
        self,
        src: Tensor,
        pos: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward one deformable encoder layer."""
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        return self.forward_ffn(src)


class MSDeformAttnTransformerEncoder(nn.Module):
    """Stack of Mask2Former deformable encoder layers."""

    def __init__(self, encoder_layer: MSDeformAttnTransformerEncoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([_clone_module(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes: Tensor, valid_ratios: Tensor, device: torch.device) -> Tensor:
        """Build normalized reference points exactly as in Mask2Former."""
        reference_points_list = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, h - 0.5, int(h), dtype=torch.float32, device=device),
                torch.linspace(0.5, w - 0.5, int(w), dtype=torch.float32, device=device),
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * h)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * w)
            reference_points_list.append(torch.stack((ref_x, ref_y), -1))
        reference_points = torch.cat(reference_points_list, 1)
        return reference_points[:, :, None] * valid_ratios[:, None]

    def forward(
        self,
        src: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        pos: Tensor | None = None,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward all deformable encoder layers."""
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for layer in self.layers:
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
        return output


class MSDeformAttnTransformerEncoderOnly(nn.Module):
    """Encoder wrapper used inside the Mask2Former pixel decoder."""

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        num_feature_levels: int = 4,
        enc_n_points: int = 4,
    ):
        super().__init__()
        encoder_layer = MSDeformAttnTransformerEncoderLayer(
            d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, enc_n_points
        )
        self.encoder = MSDeformAttnTransformerEncoder(encoder_layer, num_encoder_layers)
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize parameters like the reference implementation."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, Mask2FormerMSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    @staticmethod
    def get_valid_ratio(mask: Tensor) -> Tensor:
        """Compute valid ratios for non-padded feature maps."""
        _, h, w = mask.shape
        valid_h = torch.sum(~mask[:, :, 0], 1)
        valid_w = torch.sum(~mask[:, 0, :], 1)
        return torch.stack([valid_w.float() / w, valid_h.float() / h], -1)

    def forward(self, srcs: list[Tensor], pos_embeds: list[Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """Flatten features and run the deformable encoder."""
        masks = [torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) for x in srcs]
        src_flatten, mask_flatten, lvl_pos_embed_flatten, spatial_shapes = [], [], [], []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, _, h, w = src.shape
            spatial_shapes.append((h, w))
            src_flatten.append(src.flatten(2).transpose(1, 2))
            mask_flatten.append(mask.flatten(1))
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed_flatten.append(pos_embed + self.level_embed[lvl].view(1, 1, -1))

        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)
        return memory, spatial_shapes, level_start_index


class MSDeformAttnPixelDecoder(nn.Module):
    """Faithful Mask2Former multi-scale deformable attention pixel decoder."""

    def __init__(
        self,
        input_channels: list[int],
        feature_strides: list[int],
        *,
        transformer_dropout: float = 0.0,
        transformer_nheads: int = 8,
        transformer_dim_feedforward: int = 1024,
        transformer_enc_layers: int = 6,
        conv_dim: int = 256,
        mask_dim: int = 256,
        norm: str | None = "GN",
        transformer_in_features: list[int] | None = None,
        common_stride: int = 4,
        transformer_n_points: int = 4,
    ):
        super().__init__()
        if len(input_channels) != len(feature_strides):
            raise ValueError("input_channels and feature_strides must have the same length.")
        order = sorted(range(len(feature_strides)), key=lambda i: feature_strides[i])
        self.input_channels = [int(input_channels[i]) for i in order]
        self.feature_strides = [int(feature_strides[i]) for i in order]
        self.input_order = order
        self.in_features = list(range(len(self.input_channels)))

        transformer_in_features = transformer_in_features or self.in_features[-3:]
        self.transformer_in_features = sorted([int(i) for i in transformer_in_features], key=lambda i: self.feature_strides[i])
        transformer_in_channels = [self.input_channels[i] for i in self.transformer_in_features]
        self.transformer_feature_strides = [self.feature_strides[i] for i in self.transformer_in_features]
        self.transformer_num_feature_levels = len(self.transformer_in_features)
        if self.transformer_num_feature_levels < 1:
            raise ValueError("Mask2Former pixel decoder requires at least one transformer feature.")

        self.input_proj = nn.ModuleList()
        for in_channels in transformer_in_channels[::-1]:
            proj = nn.Sequential(nn.Conv2d(in_channels, conv_dim, kernel_size=1), nn.GroupNorm(32, conv_dim))
            xavier_uniform_(proj[0].weight, gain=1)
            constant_(proj[0].bias, 0.0)
            self.input_proj.append(proj)

        self.transformer = MSDeformAttnTransformerEncoderOnly(
            d_model=conv_dim,
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=transformer_enc_layers,
            num_feature_levels=self.transformer_num_feature_levels,
            enc_n_points=transformer_n_points,
        )
        self.pe_layer = PositionEmbeddingSine(conv_dim // 2, normalize=True)
        self.mask_dim = mask_dim
        self.mask_features = nn.Conv2d(conv_dim, mask_dim, kernel_size=1, stride=1, padding=0)
        _c2_xavier_fill(self.mask_features)

        self.maskformer_num_feature_levels = 3
        self.common_stride = int(common_stride)
        stride = min(self.transformer_feature_strides)
        self.num_fpn_levels = int(math.log2(stride) - math.log2(self.common_stride))
        if self.num_fpn_levels < 0:
            raise ValueError(
                f"common_stride={common_stride} must be <= min transformer stride={stride} for Mask2Former."
            )

        lateral_convs, output_convs = [], []
        use_bias = norm in {None, ""}
        for in_channels in self.input_channels[: self.num_fpn_levels]:
            lateral_conv = Conv2dNormActivation(in_channels, conv_dim, 1, bias=use_bias, norm=norm)
            output_conv = Conv2dNormActivation(
                conv_dim, conv_dim, 3, padding=1, bias=use_bias, norm=norm, activation=F.relu
            )
            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        self.lateral_convs = nn.ModuleList(lateral_convs[::-1])
        self.output_convs = nn.ModuleList(output_convs[::-1])

    def forward_features(self, features: list[Tensor]) -> tuple[Tensor, Tensor, list[Tensor]]:
        """Run the pixel decoder and return mask features plus decoder memory features."""
        if len(features) != len(self.input_channels):
            raise ValueError(f"Expected {len(self.input_channels)} feature maps, received {len(features)}.")
        features = [features[i] for i in self.input_order]
        srcs, pos = [], []
        for idx, feature_idx in enumerate(self.transformer_in_features[::-1]):
            x = features[feature_idx].float()
            srcs.append(self.input_proj[idx](x))
            pos.append(self.pe_layer(x))

        y, spatial_shapes, level_start_index = self.transformer(srcs, pos)
        bs = y.shape[0]
        split_sizes = []
        for i in range(self.transformer_num_feature_levels):
            if i < self.transformer_num_feature_levels - 1:
                split_sizes.append(level_start_index[i + 1] - level_start_index[i])
            else:
                split_sizes.append(y.shape[1] - level_start_index[i])
        y = torch.split(y, split_sizes, dim=1)

        out, multi_scale_features = [], []
        for i, z in enumerate(y):
            out.append(z.transpose(1, 2).view(bs, -1, int(spatial_shapes[i][0]), int(spatial_shapes[i][1])))

        for idx, feature_idx in enumerate(self.in_features[: self.num_fpn_levels][::-1]):
            x = features[feature_idx].float()
            cur_fpn = self.lateral_convs[idx](x)
            y = cur_fpn + F.interpolate(out[-1], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False)
            y = self.output_convs[idx](y)
            out.append(y)

        for o in out:
            if len(multi_scale_features) < self.maskformer_num_feature_levels:
                multi_scale_features.append(o)
        return self.mask_features(out[-1]), out[0], multi_scale_features


class SelfAttentionLayer(nn.Module):
    """Mask2Former self-attention layer."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0, activation: str = "relu", normalize_before: bool = False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    @staticmethod
    def with_pos_embed(tensor: Tensor, pos: Tensor | None) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt: Tensor, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None) -> Tensor:
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        return self.norm(tgt)

    def forward_pre(self, tgt: Tensor, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None) -> Tensor:
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        return tgt + self.dropout(tgt2)

    def forward(self, tgt: Tensor, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None) -> Tensor:
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask, tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask, tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):
    """Mask2Former masked cross-attention layer."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0, activation: str = "relu", normalize_before: bool = False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    @staticmethod
    def with_pos_embed(tensor: Tensor, pos: Tensor | None) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt: Tensor, memory: Tensor, memory_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None) -> Tensor:
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)
        return self.norm(tgt)

    def forward_pre(self, tgt: Tensor, memory: Tensor, memory_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None) -> Tensor:
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        return tgt + self.dropout(tgt2)

    def forward(self, tgt: Tensor, memory: Tensor, memory_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None) -> Tensor:
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):
    """Mask2Former feed-forward layer."""

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        activation: str = "relu",
        normalize_before: bool = False,
    ):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward_post(self, tgt: Tensor) -> Tensor:
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        return self.norm(tgt)

    def forward_pre(self, tgt: Tensor) -> Tensor:
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        return tgt + self.dropout(tgt2)

    def forward(self, tgt: Tensor) -> Tensor:
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


class MultiScaleMaskedTransformerDecoder(nn.Module):
    """Faithful Mask2Former multi-scale masked transformer decoder."""

    def __init__(
        self,
        in_channels: int,
        *,
        num_classes: int,
        hidden_dim: int = 256,
        num_queries: int = 100,
        nheads: int = 8,
        dim_feedforward: int = 2048,
        dec_layers: int = 10,
        pre_norm: bool = False,
        mask_dim: int = 256,
        enforce_input_project: bool = False,
    ):
        super().__init__()
        if dec_layers < 1:
            raise ValueError("Mask2Former dec_layers must be >= 1.")
        self.num_heads = nheads
        self.num_layers = int(dec_layers) - 1
        self.pe_layer = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(hidden_dim, nheads, dropout=0.0, normalize_before=pre_norm)
            )
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(hidden_dim, nheads, dropout=0.0, normalize_before=pre_norm)
            )
            self.transformer_ffn_layers.append(
                FFNLayer(hidden_dim, dim_feedforward=dim_feedforward, dropout=0.0, normalize_before=pre_norm)
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.num_queries = num_queries
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
                _c2_xavier_fill(proj)
                self.input_proj.append(proj)
            else:
                self.input_proj.append(nn.Identity())

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    def forward(self, x: list[Tensor], mask_features: Tensor, mask: Tensor | None = None) -> dict[str, Tensor | list[dict[str, Tensor]]]:
        """Forward Mask2Former decoder over the three pixel-decoder feature levels."""
        if len(x) != self.num_feature_levels:
            raise ValueError(f"Expected {self.num_feature_levels} decoder feature levels, received {len(x)}.")
        del mask
        src, pos, size_list = [], [], []
        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        predictions_class, predictions_mask = [], []

        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
            output, mask_features, attn_mask_target_size=size_list[0]
        )
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            output = self.transformer_cross_attention_layers[i](
                output,
                src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,
                pos=pos[level_index],
                query_pos=query_embed,
            )
            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None, tgt_key_padding_mask=None, query_pos=query_embed
            )
            output = self.transformer_ffn_layers[i](output)
            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels]
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        return {
            "pred_logits": predictions_class[-1],
            "pred_masks": predictions_mask[-1],
            "aux_outputs": self._set_aux_loss(predictions_class, predictions_mask),
        }

    def forward_prediction_heads(self, output: Tensor, mask_features: Tensor, attn_mask_target_size: tuple[int, int]) -> tuple[Tensor, Tensor, Tensor]:
        """Run class/mask prediction heads and build the next boolean attention mask."""
        decoder_output = self.decoder_norm(output).transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        attn_mask = (
            attn_mask.sigmoid()
            .flatten(2)
            .unsqueeze(1)
            .repeat(1, self.num_heads, 1, 1)
            .flatten(0, 1)
            < 0.5
        ).bool()
        return outputs_class, outputs_mask, attn_mask.detach()

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class: list[Tensor], outputs_seg_masks: list[Tensor]) -> list[dict[str, Tensor]]:
        """Return auxiliary predictions from all non-final prediction heads."""
        return [{"pred_logits": a, "pred_masks": b} for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])]


class Mask2FormerHead(nn.Module):
    """Mask2Former instance segmentation head with Ultralytics-compatible inference output."""

    def __init__(self, nc: int = 80, cfg: dict[str, Any] | int | None = None, legacy_dim: int | None = None, ch: list[int] | tuple[int, ...] = ()):
        super().__init__()
        if not ch and isinstance(legacy_dim, (list, tuple)):
            ch, legacy_dim = legacy_dim, None
        if not ch:
            raise ValueError("Mask2FormerHead requires input channel metadata from parse_model.")
        cfg = self._normalize_cfg(cfg, legacy_dim, len(ch))
        self.nc = int(nc)
        self.num_queries = int(cfg["num_queries"])
        self.nl = len(ch)
        self.nm = self.num_queries
        self.common_stride = int(cfg["common_stride"])
        self.mask_threshold = float(cfg.get("mask_threshold", 0.5))
        self.max_per_image = int(cfg.get("max_per_image", cfg.get("max_det", 100)))
        self.max_det = self.max_per_image  # Backward-compatible attribute alias.
        self.end2end = False
        self.export = False
        self.format = None
        self.point_rend_source_channels = (int(cfg.get("mask_dim", 256)),)
        self.point_rend_enabled = False
        self.stride = torch.tensor([float(s) for s in cfg["feature_strides"]])
        self.loss_cfg = {
            "class_weight": float(cfg.get("class_weight", 2.0)),
            "mask_weight": float(cfg.get("mask_weight", 5.0)),
            "dice_weight": float(cfg.get("dice_weight", 5.0)),
            "no_object_weight": float(cfg.get("no_object_weight", 0.1)),
            "train_num_points": int(cfg.get("train_num_points", 12544)),
            "oversample_ratio": float(cfg.get("oversample_ratio", 3.0)),
            "importance_sample_ratio": float(cfg.get("importance_sample_ratio", 0.75)),
            "overlap_mask": bool(cfg.get("overlap_mask", True)),
        }

        self.pixel_decoder = MSDeformAttnPixelDecoder(
            list(map(int, ch)),
            cfg["feature_strides"],
            transformer_dropout=float(cfg.get("dropout", 0.0)),
            transformer_nheads=int(cfg.get("nheads", 8)),
            transformer_dim_feedforward=int(cfg.get("encoder_dim_feedforward", 1024)),
            transformer_enc_layers=int(cfg.get("enc_layers", 6)),
            conv_dim=int(cfg.get("conv_dim", 256)),
            mask_dim=int(cfg.get("mask_dim", 256)),
            norm=cfg.get("norm", "GN"),
            transformer_in_features=cfg["transformer_in_features"],
            common_stride=self.common_stride,
            transformer_n_points=int(cfg.get("n_points", 4)),
        )
        self.predictor = MultiScaleMaskedTransformerDecoder(
            int(cfg.get("conv_dim", 256)),
            num_classes=self.nc,
            hidden_dim=int(cfg.get("hidden_dim", cfg.get("conv_dim", 256))),
            num_queries=self.num_queries,
            nheads=int(cfg.get("nheads", 8)),
            dim_feedforward=int(cfg.get("dim_feedforward", 2048)),
            dec_layers=int(cfg.get("dec_layers", 10)),
            pre_norm=bool(cfg.get("pre_norm", False)),
            mask_dim=int(cfg.get("mask_dim", 256)),
            enforce_input_project=bool(cfg.get("enforce_input_project", False)),
        )
        self._last_outputs: dict[str, Any] | None = None

    @staticmethod
    def _normalize_cfg(cfg: dict[str, Any] | int | None, legacy_dim: int | None, n_features: int) -> dict[str, Any]:
        """Normalize modern dict config and tolerate the old `[nc, 32, 256]` placeholder style."""
        if isinstance(cfg, dict):
            out = dict(cfg)
        else:
            out = {}
            if cfg is not None:
                out["num_queries"] = int(cfg)
            if legacy_dim is not None and not isinstance(legacy_dim, (list, tuple)):
                out["mask_dim"] = int(legacy_dim)
                out["conv_dim"] = int(legacy_dim)
        out.setdefault("feature_strides", [4 * (2**i) for i in range(n_features)])
        if len(out["feature_strides"]) != n_features:
            raise ValueError(
                f"Mask2Former feature_strides length {len(out['feature_strides'])} must match input features {n_features}."
            )
        out.setdefault("transformer_in_features", list(range(max(0, n_features - 3), n_features)))
        out.setdefault("common_stride", min(out["feature_strides"]))
        out.setdefault("conv_dim", 256)
        out.setdefault("mask_dim", 256)
        out.setdefault("hidden_dim", out["conv_dim"])
        out.setdefault("num_queries", 100)
        return out

    def forward(self, x: list[Tensor]) -> dict[str, Any] | tuple[Tensor, Tensor]:
        """Return native Mask2Former predictions; family adapters perform reference instance inference."""
        mask_features, _, multi_scale_features = self.pixel_decoder.forward_features(x)
        outputs = self.predictor(multi_scale_features, mask_features, None)
        outputs["feats"] = multi_scale_features
        if self.point_rend_enabled and hasattr(self, "point_rend"):
            outputs["pointrend_features"] = self.point_rend.project_features([mask_features])
        self._last_outputs = outputs
        if self.training:
            return outputs
        if self.export:
            return outputs["pred_logits"], outputs["pred_masks"]
        return outputs


def _get_activation_fn(activation: str):
    """Return the activation function used by the reference transformer code."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")


def _clone_module(module: nn.Module) -> nn.Module:
    """Clone a module via deepcopy without importing copy in hot paths."""
    import copy

    return copy.deepcopy(module)
